import argparse
import logging
import re

from utils import dump_jsonl, get_predicted_label, read_jsonl

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

df = read_jsonl(args.input, required_columns=["generated_text"])


if "generated_text" in df.columns:
    if "predicted_label_name" not in df.columns:
        df["predicted_label_name"] = None

        df.loc[~df.generated_text.isna(), "predicted_label_name"] = df[~df.generated_text.isna()].generated_text.apply(
            lambda text: get_predicted_label(text, True)
        )


dump_jsonl(df, args.output)
