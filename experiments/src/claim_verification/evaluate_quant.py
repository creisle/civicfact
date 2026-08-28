import argparse
import json
import logging
import os
import subprocess
import time
from datetime import timedelta

import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
from turboquant import TurboQuantCache

from utils import dump_jsonl, load_generation_config
from utils.cache_local import add_many_to_cache, cache_setup, match_to_cache


def get_os_release():
    output = ""
    if os.path.exists("/etc/os-release"):
        output = subprocess.getoutput("cat /etc/os-release")
    return output


def file_exists(filename):
    if os.path.exists(filename):
        return filename
    raise TypeError(f"file {filename} does not exist")


def load_pipeline(model_path, tokenizer_model=None, model_name=None, max_total_tokens=None, generation_config={}):
    auth_kwargs = {}
    if os.environ.get("HF_TOKEN"):
        auth_kwargs["token"] = os.environ["HF_TOKEN"].strip()
    if not model_name:
        model_name = model_path

    os_release = get_os_release().replace("\n", "; ")
    logging.info(f"os-release: {os_release}")
    logging.info(f'HOSTNAME: {os.environ.get("HOSTNAME")}')
    logging.info(f"pid: {os.getpid()}")

    if not tokenizer_model:
        tokenizer_model = model_path

    # Download config
    logging.info(f"getting the model config: {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, **auth_kwargs)

    try:
        logging.info(f"config.max_position_embeddings: {config.max_position_embeddings}")
    except AttributeError as err:
        pass
    try:
        logging.info(f"config.text_config.max_position_embeddings: {config.text_config.max_position_embeddings}")
        if max_total_tokens is not None:
            config.text_config.max_position_embeddings = int(max_total_tokens)
    except AttributeError as err:
        pass

    model_args = {}
    if "model_args" in generation_config:
        model_args.update(generation_config["model_args"])
    if "device_map_type" in generation_config:
        model_args["device_map"] = "balanced_low_0"

    adapter_config = {}
    if os.path.exists(model_path):
        logging.info(f"model path exists: {model_path}")
        adapter_config_file = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_file):
            logging.info(f"reading: {adapter_config_file}")
            with open(adapter_config_file, "r") as fh:
                adapter_config = json.load(fh)

    if "quantization_config" in generation_config:
        # don't need to quantize local models
        print('generation_config["quantization_config"]', generation_config["quantization_config"])
        model_args["quantization_config"] = BitsAndBytesConfig(**generation_config["quantization_config"])

    if adapter_config:
        base_model = AutoModelForCausalLM.from_pretrained(
            adapter_config["base_model_name_or_path"],
            trust_remote_code=True,
            **model_args,
            **auth_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_args,
            **auth_kwargs,
        )

    try:
        logging.info(f"model.config.max_position_embeddings: {model.config.max_position_embeddings}")
    except AttributeError as err:
        pass

    try:
        for key, value in model.hf_device_map.items():
            logging.info(f"device map {key} = {value}")
    except AttributeError:
        pass

    logging.info(f"loading tokenizer model: {tokenizer_model}")
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, **auth_kwargs)

    try:
        logging.info(f"tokenizer.model_max_length: {tokenizer.model_max_length}")
        tokenizer.max_model_length = config.text_config.max_position_embeddings
    except AttributeError as err:
        logging.error(f"error getting tokenizer.model_max_length: {err}")
        pass

    logging.info("setting up the pipeline")
    # Run text-generation pipeline
    tokenizer.pad_token = (
        tokenizer.eos_token
    )  # https://github.com/huggingface/transformers/issues/22546#issuecomment-1764728674

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # device='cuda:0',  # (Optional) to run on GPU 0
    )
    try:
        for key, value in pipe.model.hf_device_map.items():
            logging.info(f"device map {key} = {value}")
    except AttributeError:
        pass
    return pipe


def perform_mid_run_cache(dataset, output, gen_config_json, dataset_text_field, model_path, cache_start=0):
    """
    For jobs that run for a long time they may be killed by time-limit, cache the results partway through to avoid data loss

       id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt VARCHAR NOT NULL,
            generated_text VARCHAR,
            generation_config VARCHAR NOT NULL,
            model_path VARCHAR NOT NULL,
            last_updated datetime default current_timestamp
    """
    rows = []

    for example, out in list(zip(dataset, output))[cache_start:]:
        prompt = example[dataset_text_field]
        rows.append(
            {
                "generated_text": out[0]["generated_text"][len(prompt) :],
                "prompt": prompt,
                "model_path": model_path,
                "generation_config": gen_config_json,
            }
        )
    if len(rows) == 0:
        return

    cache_conn = cache_setup()
    try:
        logging.info(f"adding {len(rows)} rows to the cache")
        add_many_to_cache(cache_conn, rows)
    finally:
        cache_conn.close()


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
        "--tokenizer_model",
        default=None,
        help="Path to the pre-trained pytorch model used for generating tokens. Generally for a fine-tuned sequence classification this will be the base model it was fine-tuned on",
    )
    parser.add_argument(
        "--model_name",
        help="what to call the model in the output. will default to the model path",
    )
    parser.add_argument(
        "--dataset_text_field",
        default="prompt.eval",
        help="the column in the JSONL to use as text data",
    )
    parser.add_argument("--generation_config", required=True)
    parser.add_argument("--token_count_col", default="expected_response.tokens")
    parser.add_argument("--output_column", default="generated_text")

    args = parser.parse_args()

    logging.info(f"reading: {args.input}")
    df = pd.read_json(args.input, lines=True)
    logging.info(f"input columns: {df.columns.tolist()}")
    logging.info(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}')
    df["line_index"] = df.index
    assert args.output_column not in df.columns, f"output column ({args.output_column}) cannot exist as an input column"
    if args.token_count_col in df.columns:
        logging.info(
            f"The max value in the token count column ({args.token_count_col}) is {df[args.token_count_col].max()}"
        )
        max_new_tokens = df[args.token_count_col].max()
    else:
        max_new_tokens = None

    assert args.dataset_text_field in df.columns

    config = load_generation_config(args.generation_config, max_new_tokens)
    with open(args.generation_config, "r") as fh:
        gen_config_json = json.dumps(json.load(fh), sort_keys=True)

    max_total_tokens = None
    if "prompt.train.tokens" in df.columns:
        max_total_tokens = df["prompt.train.tokens"].max() + config["generation"].get("max_new_tokens", 0)
        logging.info(f"the max expected total tokens is {max_total_tokens}")

    cache_conn = cache_setup()
    try:
        cache_df = match_to_cache(cache_conn, df, args.model_path, gen_config_json, args.dataset_text_field)
        assert df.shape[0] == cache_df.shape[0]
    finally:
        cache_conn.close()

    logging.info(f'loaded {cache_df[~cache_df["cache.generated_text"].isna()].shape[0]} previously generated prompts')
    if cache_df[cache_df["cache.generated_text"].isna()].shape[0] > 0:

        if "turboquant" in config:
            tbq_cache = TurboQuantCache(bits=config["turboquant"].get("bits", 4))
            config["generation"]["past_key_values"] = tbq_cache

        dtype = torch.float16

        # setup text-generation pipeline
        generator = load_pipeline(
            args.model_path,
            args.tokenizer_model,
            args.model_name,
            max_total_tokens=max_total_tokens,
            generation_config=config,
        )
        input_df = cache_df[cache_df["cache.generated_text"].isna()]
        dataset = Dataset.from_pandas(input_df[[args.dataset_text_field, "line_index"]])
        print(dataset)

        CACHE_INTERVAL = 60 * 30  # seconds

        with torch.autocast("cuda", dtype=dtype):
            # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            prompts = input_df[args.dataset_text_field].tolist()
            logging.info(f"evaluating {len(prompts)} prompts")
            cache_pos = 0
            hours_cached = set([0])

            start_time = time.time()

            output = []

            for out in tqdm(generator(KeyDataset(dataset, args.dataset_text_field), **config["generation"])):
                current_hour = (time.time() - start_time) // CACHE_INTERVAL  # only cache once per interval
                if current_hour not in hours_cached:
                    hours_cached.add(current_hour)
                    perform_mid_run_cache(
                        dataset=dataset,
                        output=output,
                        gen_config_json=gen_config_json,
                        dataset_text_field=args.dataset_text_field,
                        model_path=args.model_path,
                        cache_start=cache_pos,
                    )
                    cache_pos = len(output)

                output.append(out)

        rows = []

        for example, out in zip(dataset, output):
            example[args.output_column] = out[0]["generated_text"][len(example[args.dataset_text_field]) :]
            rows.append(example)
        result_df = pd.DataFrame.from_records(rows)
        logging.info(f"generated {result_df.shape[0]} texts")
        logging.info(f"generated dataset columns: {result_df.columns.tolist()}")

        result_df = result_df.drop(columns=[c for c in result_df.columns if c.startswith("_")])
        # now add the cached responses as well
        cached_resp_df = cache_df[~cache_df["cache.generated_text"].isna()][
            ["cache.generated_text", "line_index", args.dataset_text_field]
        ].rename(columns={"cache.generated_text": args.output_column})
        result_df = pd.concat([cached_resp_df, result_df]).reset_index(drop=True)

    else:
        result_df = cache_df[["cache.generated_text", "line_index", args.dataset_text_field]].rename(
            columns={"cache.generated_text": args.output_column}
        )

    result_df = result_df.merge(df, on=[args.dataset_text_field, "line_index"]).drop(columns=["line_index"])
    logging.info(
        f"after merging results back onto original dataset (n={df.shape[0]}) there are {result_df.shape[0]} rows"
    )

    if not os.path.exists(os.path.dirname(args.output)) and os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output))
    logging.info(f"output columns: {result_df.columns.tolist()}")
    dump_jsonl(result_df, args.output)


if __name__ == "__main__":
    main()
