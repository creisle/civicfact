import argparse
import logging
import os

import numpy as np
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from utils import apply_query_template, dump_jsonl, read_jsonl


def compute_metrics(pred):
    references = pred.label_ids
    predictions = pred.predictions

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
    return result


logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--model_path", required=True)
parser.add_argument("--model_name")
parser.add_argument("--partition", required=True)
parser.add_argument("--output", help="path to output directory", required=True)
parser.add_argument("--query_column", default="claim.flat")
parser.add_argument("--passage_column", default="chunk.text")
parser.add_argument("--query_template", required=True, type=int)


args = parser.parse_args()

args.auth_kwargs = {}
if os.environ.get("HF_TOKEN"):
    args.auth_kwargs["token"] = os.environ["HF_TOKEN"].strip()
if not args.model_name:
    args.model_name = args.model_path


text1_colname = "sentence1"
text2_colname = "sentence2"
label_colname = "dummy_label"


data_df = read_jsonl(
    args.input, required_columns=[args.query_column, args.passage_column], warning_columns=["partition"]
)

if args.partition != "all":
    data_df = data_df[data_df.partition == args.partition]


data_df["_query_text"] = data_df.apply(
    lambda row: apply_query_template(row, args.query_column, args.query_template), axis=1
)

df = data_df.copy()
df = df.rename(columns={"_query_text": text1_colname, args.passage_column: text2_colname})
df[label_colname] = 1
df = df[[text1_colname, text2_colname, label_colname]].drop_duplicates()

logging.info(f"evaluating {df.shape[0]} examples")

# convert to hf style datasets
ds = Dataset.from_pandas(df, preserve_index=False)

# encode the inputs
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

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


tokenized_dataset = ds.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

predictions = trainer.predict(tokenized_dataset)
ds = ds.add_column("predicted_similarity.score", np.ravel(predictions.predictions))

pred_df = ds.to_pandas().drop(columns=[label_colname])
pred_df = pred_df.rename(columns={text1_colname: "_query_text", text2_colname: args.passage_column})
pred_df = pred_df.merge(data_df)
pred_df["model_name"] = args.model_name
pred_df = pred_df.drop(columns="_query_text")

dump_jsonl(pred_df, args.output)
