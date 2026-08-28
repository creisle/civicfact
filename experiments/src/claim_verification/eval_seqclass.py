import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from utils import dump_json_to_str, dump_jsonl, read_jsonl

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("input")  # "/projects/creisle_prj/git/thesis_work/results/sentence_similarity/pairs.jsonl.gz"
parser.add_argument("--model_path", required=True)
parser.add_argument("--model_name")

parser.add_argument("--output", help="path to output file", required=True)
parser.add_argument("--query_column", default="claim.flat")

g = parser.add_mutually_exclusive_group(required=True)
g.add_argument("--passage_column", default="evidence.flat")
g.add_argument("--unpaired", action="store_true")

parser.add_argument("--label_column", default="gold_label_name")

parser.add_argument("--partition", required=True)
args = parser.parse_args()

args.auth_kwargs = {}
if os.environ.get("HF_TOKEN"):
    args.auth_kwargs["token"] = os.environ["HF_TOKEN"].strip()
if not args.model_name:
    args.model_name = args.model_path


data_df = read_jsonl(args.input)
data_df = data_df[data_df.partition == args.partition]

assert data_df.shape[0] > 0

df = data_df.copy()

for col in [args.query_column, args.passage_column, "partition"]:
    assert col in df.columns, f"missing required column, {col}, in input file"

df = df.rename(columns={args.query_column: args.query_column, args.passage_column: args.passage_column})
df[args.label_column] = 1
df = df[[args.query_column, args.passage_column, args.label_column]].drop_duplicates()

logging.info(f"evaluating {df.shape[0]} examples")
logging.info(f'evaluating in {"un" if args.unpaired else ""}paired mode')

# convert to hf style datasets
ds = Dataset.from_pandas(df, preserve_index=False)

# encode the inputs
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_path,
    trust_remote_code=True,
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


logging.info(f"labels: {model.config.label2id}")


tokenized_dataset = ds.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

predictions = trainer.predict(tokenized_dataset)

training_labels = sorted(model.config.label2id, key=lambda k: model.config.label2id[k])

predicted_probability = torch.softmax(torch.from_numpy(predictions.predictions), dim=1)
predicted_index = torch.argmax(predicted_probability, dim=1)
predicted_probability = [
    dump_json_to_str({l: p for l, p in zip(training_labels, prob)}) for prob in predicted_probability
]
ds = ds.add_column("predicted_label_id", predicted_index.numpy())
ds = ds.add_column("predicted_label_probability", predicted_probability)


pred_df = ds.to_pandas().drop(columns=[args.label_column])
pred_df["predicted_label_probability"] = pred_df.predicted_label_probability.apply(json.loads)
pred_df = pred_df.merge(data_df)
pred_df["model_name"] = args.model_name
pred_df["predicted_label_name"] = pred_df.predicted_label_id.apply(lambda label_id: model.config.id2label[label_id])


dump_jsonl(pred_df, args.output)
