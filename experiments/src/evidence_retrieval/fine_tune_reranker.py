import argparse
import json
import logging
import math
import os

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from transformers.utils.logging import disable_progress_bar

from utils import apply_query_template

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
parser.add_argument("--query_template", type=int, required=True)
args = parser.parse_args()
text1_colname = "sentence1"
text2_colname = "sentence2"
label_colname = "score"

logging.info(f"reading config: {args.config}")
with open(args.config, "r") as fh:
    config = json.load(fh)

BASE_MODEL = args.base_model
BASE_MODEL_DIR = BASE_MODEL.replace("/", "__")
logging.info(f"reading: {args.input}")
df = pd.read_json(args.input, lines=True, dtype=False)

for col in [args.query_column, args.passage_column, args.label_column, "partition"]:
    assert col in df.columns, f"missing required column, {col}, in input file"

df["_query_text"] = df.apply(
    lambda row: apply_query_template(row, args.query_column, args.query_template),
    axis=1,
)
df = df.rename(
    columns={
        "_query_text": text1_colname,
        args.passage_column: text2_colname,
        args.label_column: label_colname,
    }
)

logging.info(f"loading model: {BASE_MODEL}")
model = CrossEncoder(
    BASE_MODEL, trust_remote_code=True, num_labels=1
)  # always num_labels=1 for non-discrete scores


train_samples = []
dev_samples = []

for _, row in df.iterrows():
    ex = InputExample(
        texts=[row[text1_colname], row[text2_colname]], label=row[label_colname]
    )
    if row["partition"] == "train":
        train_samples.append(ex)
    else:
        dev_samples.append(ex)


# We wrap train_samples, which is a list of InputExample, in a pytorch DataLoader
train_dataloader = DataLoader(
    train_samples,
    shuffle=True,
    batch_size=config["train"]["per_device_train_batch_size"],
)

# During training, we use CESoftmaxAccuracyEvaluator and CEF1Evaluator to measure the performance on the dev set
evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name="train-eval")

warmup_steps = math.ceil(
    len(train_dataloader)
    * config["train"]["num_train_epochs"]
    * config["train"]["warmup_ratio"]
)  # 10% of train data for warm-up


# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=config["train"]["num_train_epochs"],
    evaluation_steps=config["train"]["eval_steps"],
    warmup_steps=warmup_steps,
    output_path=os.path.join(args.output, "best_checkpoint"),
    save_best_model=True,
)

# model.save(os.path.join(args.output, "best_checkpoint"))
