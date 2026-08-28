import argparse
import logging

import pandas as pd

from prompting import EXAMPLES_COLNAME, TEMPLATES
from utils import dump_jsonl, read_jsonl


def create_prompt(row, template_name="default"):
    template = TEMPLATES[template_name]
    return template.build_chat(row)[0]["content"]


def build_training_label(row, template_name="default"):
    template = TEMPLATES[template_name]
    return template.build_chat(row)[-1]["content"]


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to the JSONL inputs")
    parser.add_argument("output", help="path to a new JSONL which contains the prompts")
    parser.add_argument("--template", choices=TEMPLATES, required=True)
    parser.add_argument(
        "--examples",
        type=int,
        required=True,
        help="number of examples to prepend before the prompt",
    )
    args = parser.parse_args()

    df = read_jsonl(
        args.input,
        required_columns=["claim.flat", "gold_label_name", "id"],
        warning_columns=["document.pmid", "display_id", "evidence.flat"],
    )

    logging.info(f"read {df.shape[0]} lines")
    logging.info(f"limiting to {args.examples} assigned examples")
    df[EXAMPLES_COLNAME] = df[EXAMPLES_COLNAME].apply(lambda examples: examples[: args.examples])

    df["chat"] = df.apply(lambda row: TEMPLATES[args.template].build_chat(row), axis=1)
    df["template"] = args.template
    df["examples"] = args.examples

    df = df.rename(columns={"civic.description": "summary"})

    dump_jsonl(df, args.output)


if __name__ == "__main__":
    main()
