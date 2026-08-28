# script adapted from here: https://huggingface.co/blog/train-reranker#data-on-the-hugging-face-hub

import argparse
import json
import logging
import os
import traceback

import torch
from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.evaluation import SequentialEvaluator
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderModelCardData
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderNanoBEIREvaluator,
    CrossEncoderRerankingEvaluator,
)
from sentence_transformers.cross_encoder.losses import (
    BinaryCrossEntropyLoss,
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import (
    CrossEncoderTrainingArguments,
)
from sentence_transformers.util import mine_hard_negatives

from utils import NpEncoder, apply_query_template, dump_jsonl, read_jsonl

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.ERROR)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name")
    parser.add_argument("--output", help="path to output directory", required=True)
    parser.add_argument("--config", required=True, help="path to the json config containing the training parameters")
    args = parser.parse_args()

    args.auth_kwargs = {}
    if os.environ.get("HF_TOKEN"):
        args.auth_kwargs["token"] = os.environ["HF_TOKEN"].strip()
    if not args.model_name:
        args.model_name = args.model_path

    logging.info(f"reading config: {args.config}")
    with open(args.config, "r") as fh:
        config = json.load(fh)

    # 1a. Load a model to finetune with 1b. (Optional) model card data
    # Loading in fp32 is preferred for training if your memory can handle it
    model = CrossEncoder(
        args.model_path,
        model_kwargs={"torch_dtype": config["compute_dtype"]},
    )
    print("Model max length:", model.max_length)
    print("Model num labels:", model.num_labels)

    if config["loss_function"] == "CachedMultipleNegativesRankingLoss":
        loss = CachedMultipleNegativesRankingLoss(model=model)
    elif config["loss_function"] == "BinaryCrossEntropyLoss":
        loss = BinaryCrossEntropyLoss(model=model, pos_weight=torch.tensor(config["hard_train"]["num_negatives"]))
    else:
        raise NotImplementedError(f'loss function not implemented: {config["loss_function"]}')

    # 2a. Load the GooAQ dataset: https://huggingface.co/datasets/sentence-transformers/gooaq
    df = read_jsonl("anchor.jsonl.gz")
    train_df = df[df.partition == "train"].explode("positive").reset_index(drop=True)
    eval_df = df[df.partition == "dev"].explode("positive").reset_index(drop=True)
    train_dataset = Dataset.from_pandas(train_df[["anchor", "positive"]], preserve_index=False)
    eval_dataset = Dataset.from_pandas(eval_df[["anchor", "positive"]], preserve_index=False)

    corpus_eval_partitions = ["dev", "train"] if config.get("hard_eval_corpus_all") else ["dev"]
    corpus_full = (
        df[df.partition.isin(corpus_eval_partitions)]
        .explode("documents")
        .reset_index(drop=True)["documents"]
        .unique()
        .tolist()
    )
    corpus_train = train_df.explode("documents").reset_index(drop=True)["documents"].unique().tolist()
    logging.info(train_dataset)
    logging.info(eval_dataset)

    # 2b. Modify our training dataset to include hard negatives using an embedding model
    embedding_model = SentenceTransformer(config.get("embedding_model", args.model_path))

    logging.info("generating hard training data")
    hard_train_dataset = mine_hard_negatives(
        train_dataset,
        embedding_model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        corpus=corpus_train,
        **config["hard_train"],
        output_format="labeled-pair",  # The output format is (query, passage, label), as required by BinaryCrossEntropyLoss
    )
    logging.info(hard_train_dataset)
    logging.info("finished generating hard training data")

    # 2c. (Optionally) Save the hard training dataset to disk
    # hard_train_dataset.save_to_disk("gooaq-hard-train")
    # Load again with:
    # hard_train_dataset = load_from_disk("gooaq-hard-train")

    # 3. Define our training loss.
    # pos_weight is recommended to be set as the ratio between positives to negatives, a.k.a. `num_hard_negatives`

    # 4a. Define evaluators. We use the CrossEncoderNanoBEIREvaluator, which is a light-weight evaluator for English reranking
    nano_beir_evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["scidocs", "scifact", "climatefever"],
        batch_size=config["train"]["per_device_eval_batch_size"] * config["train"]["gradient_accumulation_steps"],
    )

    # 4b. Define a reranking evaluator by mining hard negatives given query-answer pairs
    # We include the positive answer in the list of negatives, so the evaluator can use the performance of the
    # embedding model as a baseline.
    logging.info("generating hard eval data")
    hard_eval_dataset = mine_hard_negatives(
        eval_dataset,
        embedding_model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        corpus=corpus_full,
        **config["hard_eval"],
        output_format="n-tuple",
    )
    # move the embedding model off the GPU to make space for loading the cross encoder model
    embedding_model.cpu()
    del embedding_model

    logging.info("finished generating hard eval data")
    logging.info(hard_eval_dataset)
    reranking_evaluator = CrossEncoderRerankingEvaluator(
        samples=[
            {
                "query": sample["anchor"],
                "positive": [sample["positive"]],
                "documents": [sample[column_name] for column_name in hard_eval_dataset.column_names[2:]],
            }
            for sample in hard_eval_dataset
        ],
        batch_size=config["train"]["per_device_eval_batch_size"],
        name="civic-dev",
        always_rerank_positives=False,
    )

    # 4c. Combine the evaluators & run the base model on them
    evaluator = SequentialEvaluator([reranking_evaluator, nano_beir_evaluator])
    evaluator(model)

    steps_per_epoch = len(train_dataset) // (
        config["train"]["per_device_train_batch_size"] * config["train"]["gradient_accumulation_steps"]
    )
    # 5. Define the training arguments
    train_args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=args.output,
        **config["train"],
        metric_for_best_model="eval_civic-dev_" + config["best_model_metric"],
        save_steps=steps_per_epoch // config["save_per_epoch"],
        eval_steps=steps_per_epoch // config["eval_per_epoch"],
        logging_steps=steps_per_epoch // config["log_per_epoch"],
    )

    # 6. Create the trainer & start training
    trainer = CrossEncoderTrainer(
        model=model,
        args=train_args,
        train_dataset=hard_train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 7. Evaluate the final model, useful to include these in the model card
    evaluator(model)

    # 8. Save the final model
    model.save_pretrained(os.path.join(args.output, "best_checkpoint"))

    logging.info(f"The best checkpoint is {trainer.state.best_model_checkpoint}")

    # logging.info("saving the model config in the best checkpoint")
    # model.config.to_json_file(os.path.join(OUTPUT_DIR, "best_checkpoint", "config.json"))
    # logging.info(f"memory footprint: {model.get_memory_footprint()}")
    training_params_filename = os.path.join(args.output, "best_checkpoint", "training_params.json")
    logging.info(f"writing: {training_params_filename}")
    try:
        with open(training_params_filename, "w") as fh:
            json.dump(args, fh, sort_keys=True, indent="  ", cls=NpEncoder)
    except Exception as err:
        logging.warning(err)


if __name__ == "__main__":
    main()
    pass
