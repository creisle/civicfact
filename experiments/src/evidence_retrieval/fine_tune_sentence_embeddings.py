import argparse
import logging
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from transformers import EarlyStoppingCallback
from transformers.utils.logging import disable_progress_bar

from utils import read_training_config

disable_progress_bar()

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--base_model", required=True)
parser.add_argument("--output", help="path to output directory")
parser.add_argument("--query_column", default="claim.flat")
parser.add_argument("--passage_column", default="chunk.text")
parser.add_argument("--label_column", default="similarity.score")
parser.add_argument(
    "--loss_function", default="cosim", choices=["cosim", "cosent", "angle"]
)
parser.add_argument(
    "--config",
    required=True,
    help="path to the json config containing the training parameters",
)
args = parser.parse_args()
text1_colname = "sentence1"
text2_colname = "sentence2"
label_colname = "score"


BASE_MODEL = args.base_model
BASE_MODEL_DIR = BASE_MODEL.replace("/", "__")
logging.info(f"reading: {args.input}")
df = pd.read_json(args.input, lines=True, dtype=False)

for col in [args.query_column, args.passage_column, args.label_column, "partition"]:
    assert col in df.columns, f"missing required column, {col}, in input file"

df = df.rename(
    columns={
        args.query_column: text1_colname,
        args.passage_column: text2_colname,
        args.label_column: label_colname,
    }
)
df_dev = df[df.partition == "dev"]
df_train = df[df.partition == "train"]

config = read_training_config(args.config, df_train.shape[0])

# convert to hf style datasets
ds_dev = Dataset.from_pandas(
    df_dev[[text1_colname, text2_colname, label_colname]], preserve_index=False
)
ds_train = Dataset.from_pandas(
    df_train[[text1_colname, text2_colname, label_colname]], preserve_index=False
)
datasets = DatasetDict({"train": ds_train, "dev": ds_dev})

logging.info(f"loading model: {BASE_MODEL}")
model = SentenceTransformer(BASE_MODEL, trust_remote_code=True)


loss_func_mapping = {
    "cosim": losses.CosineSimilarityLoss,
    "cosent": losses.CoSENTLoss,
    "angle": losses.AnglELoss,
}
logging.info(f"setting loss: {args.loss_function}")
loss = loss_func_mapping[args.loss_function](model)

logging.info(f"will log every {config['train']['logging_steps']} steps")


callbacks = []
if "early_stopping" in config:
    callbacks.append(EarlyStoppingCallback(**config["early_stopping"]))

trainer = SentenceTransformerTrainer(
    model=model,
    loss=loss,
    train_dataset=datasets["train"],
    eval_dataset=datasets["dev"],
    args=SentenceTransformerTrainingArguments(
        **config["train"],
        # Required parameter:
        output_dir=args.output,
    ),
    callbacks=callbacks,
)
logging.info("starting training")
trainer.train()

trainer.save_model(os.path.join(args.output, "best_checkpoint"))
