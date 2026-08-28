import argparse
import json
import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from scipy.stats import pearsonr, spearmanr
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils import apply_query_template, read_training_config


def compute_metrics(pred):
    references = pred.label_ids
    predictions = pred.predictions
    print(references.shape, predictions.shape)

    spearman_corr, spearman_pvalue = spearmanr(references, predictions, axis=None)  # axis=None: ravel arrays

    pearson_corr, pearson_pvalue = pearsonr(references, predictions, axis=None)  # axis=None: ravel arrays

    avg_corr = sum([spearman_corr, pearson_corr]) / 2

    result = {
        "spearman_corr": spearman_corr,
        "spearman_corr_pvalue": spearman_pvalue,
        "pearson_corr": pearson_corr,
        "pearson_corr_pvalue": pearson_pvalue,
        "avg_corr": avg_corr,
    }
    print(result)

    return result


logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("input")  # "/projects/creisle_prj/git/thesis_work/results/sentence_similarity/pairs.jsonl.gz"
parser.add_argument("--model_path", required=True)
parser.add_argument("--model_name")

parser.add_argument("--output", help="path to output directory", required=True)
parser.add_argument("--query_column", default="claim.flat")
parser.add_argument("--passage_column", default="chunk.text")
parser.add_argument("--label_column", default="similarity.score")
parser.add_argument("--config", required=True, help="path to the json config containing the training parameters")
parser.add_argument("--query_template", required=True, type=int)
args = parser.parse_args()

args.auth_kwargs = {}
if os.environ.get("HF_TOKEN"):
    args.auth_kwargs["token"] = os.environ["HF_TOKEN"].strip()
if not args.model_name:
    args.model_name = args.model_path


text1_colname = "sentence1"
text2_colname = "sentence2"
label_colname = "label"

logging.info(f"reading config: {args.config}")
with open(args.config, "r") as fh:
    config = json.load(fh)


logging.info(f"reading: {args.input}")
df = pd.read_json(args.input, lines=True, dtype=False)

for col in [args.query_column, args.passage_column, args.label_column, "partition"]:
    assert col in df.columns, f"missing required column, {col}, in input file"

df["_query_text"] = df.apply(lambda row: apply_query_template(row, args.query_column, args.query_template), axis=1)
df = df.rename(
    columns={"_query_text": text1_colname, args.passage_column: text2_colname, args.label_column: label_colname}
)
df[label_colname] = df[label_colname].astype(float)

# TODO: REMOVE SUBSAMPLING
df_dev = df[df.partition == "dev"]
df_train = df[df.partition == "train"]

# convert to hf style datasets
ds_dev = Dataset.from_pandas(df_dev[[text1_colname, text2_colname, label_colname]], preserve_index=False)
ds_train = Dataset.from_pandas(df_train[[text1_colname, text2_colname, label_colname]], preserve_index=False)
raw_datasets = DatasetDict({"train": ds_train, "dev": ds_dev})

# encode the inputs
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

config = read_training_config(args.config, df_train.shape[0])

logging.info(f"loading config for base model ({args.model_path})")


model = AutoModelForSequenceClassification.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    num_labels=1,
    **args.auth_kwargs,
)

# ValueError: You are trying to save a non contiguous tensor: `bert.encoder.layer.0.attention.self.query.weight` which is not allowed. It either means you are trying to save tensors which are reference of each other in which case it's recommended to save only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to pack it before saving.
# https://github.com/huggingface/transformers/issues/28293#issuecomment-2284567863
for param in model.parameters():
    param.data = param.data.contiguous()

max_length = None
try:
    if model.config.max_position_embeddings:
        max_length = model.config.max_position_embeddings
except AttributeError:
    pass
logging.info(f"truncating to a maximum length of {max_length} tokens")

if max_length == 514:
    max_length -= 2
    logging.info(f"reducing max length to {max_length} tokens")


def preprocess_function(examples):
    return tokenizer(
        examples[text1_colname],
        examples[text2_colname],
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
        max_length=max_length,
    )


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

callbacks = []
if "early_stopping" in config:
    callbacks.append(EarlyStoppingCallback(**config["early_stopping"]))

trainer = Trainer(
    model,
    args=TrainingArguments(
        **config["train"],
        output_dir=args.output,
    ),
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

trainer.train()

trainer.save_model(os.path.join(args.output, "best_checkpoint"))
