import argparse
import logging
import os

import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.modules import Dense, Module, Transformer
from sentence_transformers.sentence_transformer.modules import Pooling

from utils import apply_query_template, dump_jsonl, read_jsonl


class TemperatureLogitScore(Module):
    config_keys: list[str] = ["true_token_id", "false_token_id", "temperature", "module_input_name"]

    def __init__(
        self,
        true_token_id: int,
        false_token_id: int | None = None,
        temperature: float = 1.0,
        module_input_name: str = "causal_logits",
        **kwargs,
    ) -> None:
        super().__init__()
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        self.temperature = temperature
        self.module_input_name = module_input_name

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        logits = features[self.module_input_name][:, -1]
        # Apply temperature scaling
        logits = logits / self.temperature

        if self.false_token_id is None:
            scores = logits[:, self.true_token_id]
        else:
            scores = logits[:, self.true_token_id] - logits[:, self.false_token_id]

        features["scores"] = scores.unsqueeze(1)
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)

    # The default `load` method reads `config.json` and passes the values
    # (i.e. the `config_keys`) as kwargs to __init__. This works for us,
    # so no need to override it.


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name")
    parser.add_argument("--partition", required=True)
    parser.add_argument("--output", help="path to output file", required=True)
    parser.add_argument("--query_column", default="claim.flat")
    parser.add_argument("--passage_column", default="chunk.text")
    parser.add_argument("--query_template", required=True, type=int)

    args = parser.parse_args()

    args.auth_kwargs = {}
    if os.environ.get("HF_TOKEN"):
        args.auth_kwargs["token"] = os.environ["HF_TOKEN"].strip()
    if not args.model_name:
        args.model_name = args.model_path

    data_df = read_jsonl(
        args.input, required_columns=[args.query_column, args.passage_column], warning_columns=["partition"]
    )
    if args.partition != "all":
        data_df = data_df[data_df.partition == args.partition]

    data_df["_query_text"] = data_df.apply(
        lambda row: apply_query_template(row, args.query_column, args.query_template), axis=1
    )

    df = data_df.copy()

    # Initialize Dense weights to approximate LogitScore: weight = embed("1") - embed("0")
    logging.info(f"loading: {args.model_path}")
    transformer = Transformer(args.model_path, transformer_task="text-generation")
    score_head = TemperatureLogitScore(
        true_token_id=transformer.tokenizer.convert_tokens_to_ids("yes"),
        false_token_id=transformer.tokenizer.convert_tokens_to_ids("no"),
        temperature=2.0,
    )
    pred_df = df[["_query_text", args.passage_column]].drop_duplicates()

    logging.info(f"evaluating {pred_df.shape[0]} examples")
    pairs = list(pred_df.itertuples(index=False))

    model = CrossEncoder(modules=[transformer, score_head])
    scores = model.predict(pairs)

    df = pd.DataFrame(pairs, columns=["_query_text", args.passage_column])
    df["predicted_similarity.score"] = scores

    df = df.merge(data_df)
    dump_jsonl(df.drop(columns=["_query_text"]), args.output)
