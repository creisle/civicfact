import argparse
import json
import logging

from utils import collect_filepath_metadata, dump_jsonl, read_jsonl

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()


dfs = []


df = read_jsonl(args.input, add_filename=True, add_filepath_metadata=True)
logging.info(f"loaded {df.shape[0]} rows")
df["ft"] = bool("fine-tuning" in args.input)

df["rep"] = 1


id_columns = ["id", "document.pmid", "claim.flat", "evidence.flat", "gold_label_name"]

if "example.id" in df.columns:
    id_columns[0] = "example.id"

file_columns = ["filename"] + [f for f in df.columns if f.startswith("filepath.")]


def compute_label_stability(series):
    most_freq_label = series.mode()[0]
    return sum([1 for l in series if l == most_freq_label])


id_columns = [c for c in id_columns if c in df.columns]
if "claim.flat" not in id_columns:
    id_columns.append("prompt")

df = (
    df.groupby(id_columns + file_columns, dropna=False)
    .agg({"predicted_label_name": "nunique", "rep": "sum"})
    .reset_index()
)
print(df)
print(df[["predicted_label_name", "rep"]].value_counts())
stab_df = df[df.rep >= 2].copy()

if stab_df.shape[0] == 0:
    logging.warning("cannot compute label stability as individual examples were not repeated. writing NA values")
    stab_df = df[file_columns].drop_duplicates()
    stab_df["stable"] = None
    stab_df["freq"] = None
    stab_df["label_stability"] = None
else:
    stab_df["stable"] = df.predicted_label_name == 1
    stab_df["freq"] = 1
    stab_df = stab_df.groupby(file_columns, dropna=False).agg({"stable": "sum", "freq": "sum"}).reset_index()
    stab_df["label_stability"] = stab_df.stable / stab_df.freq


dump_jsonl(stab_df[["freq", "label_stability"] + file_columns], args.output)
