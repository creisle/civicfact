import argparse
import json
import logging
import os

import pandas as pd
from transformers import AutoTokenizer

import prompting
from utils import dump_jsonl, file_exists, get_os_release


def get_chat_template(tokenizer):
    for attr in ["chat_template", "default_chat_template"]:
        if hasattr(tokenizer, attr) and getattr(tokenizer, attr):
            return getattr(tokenizer, attr)

    return tokenizer.get_chat_template()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(process)d %(asctime)s %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        help='path to JSONL input file which must as minimum contain a "prompt" field. Any remaining fields will be carried over',
        type=file_exists,
    )
    parser.add_argument(
        "output",
        help='path to JSONL output file, it will produce a "generated" field with the text and model details for the generation',
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the pre-trained pytorch model directory or model name on the huggingface repository",
    )
    parser.add_argument(
        "--model_name",
        help="what to call the model in the output. will default to the model path",
    )
    parser.add_argument(
        "--chat_template",
        default="none",
        help="the chat template to use to build the training/eval prompts",
    )
    args = parser.parse_args()

    auth_kwargs = {}
    if os.environ.get("HF_TOKEN"):
        auth_kwargs["token"] = os.environ["HF_TOKEN"].strip()
    if not args.model_name:
        args.model_name = args.model_path

    # if this is a local model, log the training state so we have a copy to refer back to
    if os.path.exists(os.path.join(args.model_path)):
        with open(os.path.join(args.model_path, "trainer_state.json"), "r") as fh:
            trainer_state = json.load(fh)
            logging.info(f'trainer state: global_step = {trainer_state["global_step"]}')
    os_release = get_os_release().replace("\n", "; ")
    logging.info(f"os-release: {os_release}")
    logging.info(f'HOSTNAME: {os.environ.get("HOSTNAME")}')

    logging.info(f"loading: {args.input}")
    df = pd.read_json(args.input, lines=True)
    for col in ["chat"]:
        assert col in df.columns, f"missing required input columns: {col}"
    input_examples = df.shape[0]

    # # config.init_device = 'meta'  # For fast initialization directly on GPU!
    logging.info(f"loading tokenizer model: {args.model_path}")
    # # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, **auth_kwargs)

    df["model"] = args.model_path
    df["chat_template"] = args.chat_template

    if args.chat_template == "none":
        df["prompt.train"] = df["chat"].apply(lambda chat: "\n".join(line["content"] for line in chat))
        df["prompt.eval"] = df["chat"].apply(lambda chat: "\n".join(line["content"] for line in chat[:-1]))
    else:
        if args.chat_template in prompting.CHAT_TEMPLATES:
            logging.info(f"found chat template ({args.chat_template}). Using custom template")
            template = prompting.CHAT_TEMPLATES[args.chat_template].template
            generation_prompt = prompting.CHAT_TEMPLATES[args.chat_template].generation_prompt
        else:
            logging.warning(
                f"chat template ({args.chat_template}) does not match a custom template, will use chat template given with model"
            )
            template = get_chat_template(tokenizer)
            generation_prompt = prompting.DEFAULT_RESPONSE_TEMPLATE

        df[prompting.RESPONSE_TEMPLATE_FIELD] = generation_prompt
        tokenizer.chat_template = template
        df["prompt.train"] = df["chat"].apply(
            lambda chat: tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                chat_template=template,
                add_generation_prompt=False,
            )
        )
        df["prompt.eval"] = df["chat"].apply(
            lambda chat: tokenizer.apply_chat_template(
                chat[:-1],
                tokenize=False,
                chat_template=template,
                add_generation_prompt=True,
            )
        )

        if args.chat_template in prompting.CHAT_TEMPLATES:
            # if the model defines its own chat template properly we don't need this check
            assert (
                df["prompt.eval"].str.endswith(generation_prompt).all()
            ), f"{df['prompt.eval'].str[-50:].unique()} does not match up to {repr(generation_prompt)}"

    logging.info(f"tokenizing prompts")
    df["expected_response"] = df["chat"].str[-1].str["content"]
    responses = df["expected_response"].unique().tolist()
    tokenized_responses = tokenizer(responses)
    sizes_df = pd.DataFrame(
        data=zip(tokenized_responses["input_ids"], responses),
        columns=["input_ids", "expected_response"],
    )
    sizes_df["expected_response.tokens"] = sizes_df["input_ids"].str.len()
    sizes_df = sizes_df.drop(columns=["input_ids"]).drop_duplicates()

    input_examples = df.shape[0]
    df = df.merge(sizes_df)
    assert df.shape[0] == input_examples

    responses = df["prompt.train"].unique().tolist()
    tokenized_responses = tokenizer(responses)
    sizes_df = pd.DataFrame(
        data=zip(tokenized_responses["input_ids"], responses),
        columns=["input_ids", "prompt.train"],
    )
    sizes_df["prompt.train.tokens"] = sizes_df["input_ids"].str.len()
    sizes_df = sizes_df.drop(columns=["input_ids"]).drop_duplicates()

    input_examples = df.shape[0]
    df = df.merge(sizes_df)
    assert df.shape[0] == input_examples

    dump_jsonl(df, args.output)


if __name__ == "__main__":
    main()
