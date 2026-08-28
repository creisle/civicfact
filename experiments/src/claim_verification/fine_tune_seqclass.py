import argparse
import json
import logging
import os

import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils import read_training_config


def log_config_settings(config, prefix=''):
    for k, v in config.items():
        if isinstance(v, dict):
            log_config_settings(v, f'{prefix}{k}.')
        else:
            logging.info(f'{prefix}{k} = {v}')

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("input")  # "/projects/creisle_prj/git/thesis_work/results/sentence_similarity/pairs.jsonl.gz"
parser.add_argument("--model_path", required=True)
parser.add_argument("--model_name")
parser.add_argument("--seed", default=None, type=int)

parser.add_argument("--output", help="path to output directory", required=True)
parser.add_argument("--query_column", default="claim.flat")

g = parser.add_mutually_exclusive_group(required=True)
g.add_argument("--passage_column", default="evidence.flat")
g.add_argument("--unpaired", action="store_true")

parser.add_argument("--random_embeddings", action="store_true", default=False)

parser.add_argument("--label_column", default="gold_label_name")
parser.add_argument("--config", required=True, help="path to the json config containing the training parameters")
args = parser.parse_args()

args.auth_kwargs = {}
if os.environ.get("HF_TOKEN"):
    args.auth_kwargs["token"] = os.environ["HF_TOKEN"].strip()
if not args.model_name:
    args.model_name = args.model_path


if args.seed is not None:
    logging.info(f"setting manual seed value for torch: {args.seed}")
    torch.manual_seed(args.seed)

logging.info(f"reading config: {args.config}")
with open(args.config, "r") as fh:
    config = json.load(fh)

log_config_settings(config)


logging.info(f"reading: {args.input}")
df = pd.read_json(args.input, lines=True, dtype=False)

for col in [args.query_column, args.passage_column, args.label_column, "partition"]:
    if args.unpaired and col == args.passage_column:
        continue
    assert col in df.columns, f"missing required column, {col}, in input file"

# TODO: REMOVE SUBSAMPLING
df_dev = df[df.partition == "dev"]
df_train = df[df.partition == "train"]
logging.info(f"there are {df_train.shape[0]} training examples and {df_dev.shape[0]} evaluation examples")

# set up label2id mapping
training_labels = sorted(list(df_train[args.label_column].unique()))
label2id = {l: i for i, l in enumerate(training_labels)}
logging.info(f"label2id: {label2id}")
id2label = {v: k for k, v in label2id.items()}
target_label_names = [id2label[i] for i in sorted(id2label.keys())]


def compute_metrics(pred):
    # global steps
    predicted_labels = pred.label_ids
    predictions = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(predicted_labels, predictions, average="micro")
    acc = accuracy_score(predicted_labels, predictions)
    class_rep = classification_report(predicted_labels, predictions, target_names=target_label_names, output_dict=True)
    # logging.info(f'step: {steps} / {total_steps}')
    # steps += 1
    logging.info(class_rep)
    logging.info(predictions)
    logging.info(f"acc={acc}, recall={recall}, precision={precision}, f1={f1}")
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "class_rep": class_rep,
    }


for df in [df_train, df_dev]:
    df["label"] = df.gold_label_name.apply(lambda l: label2id[l])

# convert to hf style datasets
if args.unpaired:
    logging.info(f"training unpaired mode, will train only on query column: {args.query_column}")
    ds_dev = Dataset.from_pandas(df_dev[[args.query_column, "label"]], preserve_index=False)
    ds_train = Dataset.from_pandas(df_train[[args.query_column, "label"]], preserve_index=False)
else:
    logging.info(
        f"training paired mode, will train on query column: {args.query_column} and passage column: {args.passage_column}"
    )
    ds_dev = Dataset.from_pandas(df_dev[[args.query_column, args.passage_column, "label"]], preserve_index=False)
    ds_train = Dataset.from_pandas(df_train[[args.query_column, args.passage_column, "label"]], preserve_index=False)
raw_datasets = DatasetDict({"train": ds_train, "dev": ds_dev})

# encode the inputs
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)


model = AutoModelForSequenceClassification.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
    **args.auth_kwargs,
)

if args.random_embeddings:
    logging.info("replacing model input embeddings with random embeddings")
    orig_em = model.get_input_embeddings()
    logging.info(f"embedding size: {orig_em.weight.shape}")
    new_em = torch.nn.Embedding(orig_em.weight.shape[0], orig_em.weight.shape[1])
    model.set_input_embeddings(new_em)

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
    if args.unpaired:
        return tokenizer(
            examples[args.query_column],
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            max_length=max_length,
        )
    return tokenizer(
        examples[args.query_column],
        examples[args.passage_column],
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
        max_length=max_length,
    )


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


config = read_training_config(args.config, df_train.shape[0])


callbacks = []
if "early_stopping" in config:
    callbacks.append(EarlyStoppingCallback(**config["early_stopping"]))

trainer = Trainer(
    model,
    args=TrainingArguments(**config["train"], output_dir=args.output),
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

trainer.train()

trainer.save_model(os.path.join(args.output, "best_checkpoint"))
