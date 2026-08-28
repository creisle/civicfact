import argparse
import json
import logging
import os
import sqlite3
import time

import backoff
import openai
import pandas as pd
import requests
import tqdm

# Regular genai client for uploads & downloads
from google import genai
from google.genai import types as google_types
from openai import OpenAI

from utils import NpEncoder, dump_jsonl, load_generation_config, read_jsonl

tqdm.tqdm.pandas()

GOOGLE_GEN_API = "https://generativelanguage.googleapis.com/v1beta/openai"

CACHE_TABLE = "requests_cache"
LLM_CACHE_DB = os.path.join(os.path.dirname(__file__), "..", "..", ".openai_requests_cache.sqlite")


def file_exists(filename):
    if os.path.exists(filename):
        return filename
    raise TypeError(f"file {filename} does not exist")


def cache_setup():
    conn = sqlite3.connect(LLM_CACHE_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        f"""\
        CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request VARCHAR NOT NULL,
            response VARCHAR,
            last_updated datetime default current_timestamp
        )"""
    )
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{CACHE_TABLE}_request ON {CACHE_TABLE}(request);")
    return conn


def reproducible_encode(input_dict):
    return json.dumps(input_dict, sort_keys=True, cls=NpEncoder)


def expand_response_choices(df):
    df["_choices"] = df["response"].str["choices"]
    df = df.explode("_choices")
    df["generated_text"] = df._choices.str["message"].str["content"]
    df = df.drop(columns=["_choices"])
    return df


def get_from_cache(cache_conn, request) -> dict:
    key = reproducible_encode(request)
    match = cache_conn.execute(f"SELECT * FROM {CACHE_TABLE} WHERE request = :request", {"request": key}).fetchall()

    if not match:
        return pd.DataFrame(columns=["response", "request", "id", "last_updated"])

    df = pd.DataFrame.from_records([dict(r) for r in match])
    df["response"] = df["response"].map(json.loads)
    df["request"] = df["request"].map(json.loads)
    df = expand_response_choices(df)
    logging.debug(f"retrieved from cache: {key}")
    return df


def add_to_cache(cache_conn, request, response):
    cursor = cache_conn.cursor()

    try:
        cursor.execute(
            f"INSERT INTO {CACHE_TABLE}(request, response) VALUES (:request, :response)",
            {"request": reproducible_encode(request), "response": reproducible_encode(response)},
        )
        cache_conn.commit()
        return cursor.lastrowid
    finally:
        cursor.close()


@backoff.on_exception(backoff.expo, requests.exceptions.RequestException)
def openai_request(client, prompt, model, config, n=1, cache_conn=None, return_batch: bool = True) -> pd.DataFrame:
    cache_args = dict()
    cache_args.update(config["generation"])
    cache_args.update(
        dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
    )
    if "system_prompt" in config:
        if len(cache_args["messages"]) > 2:
            raise NotImplementedError("unexpected complex conversation")
        cache_args["messages"] = [{"role": "system", "content": config["system_prompt"]}] + cache_args["messages"]
    cache_args["client.base_url"] = str(client.base_url)

    cache_df = get_from_cache(cache_conn, cache_args)
    if cache_df.shape[0]:
        logging.debug(f"re-using {cache_df.shape[0]} previously cached responses")
    if cache_df.shape[0] >= n:
        if return_batch:
            return pd.DataFrame(columns=["method", "url", "body"])
        return cache_df.head(n)

    curr_n = n - cache_df.shape[0]
    max_n = 8  # google doesn't like 8+, need to split into multiple requests

    rows = []
    batch_rows = []
    while curr_n > 0:
        # remove any invalid params, will be ignored on non-batch endpoint. used for local models sharing this config but will throw an error on batch endpoint
        invalid_params = ["do_sample", "use_cache", "truncation"]
        curr_args = {k: v for k, v in cache_args.items() if k not in invalid_params}
        del curr_args["client.base_url"]
        curr_args["n"] = min(max_n, curr_n)

        if return_batch:
            batch_args = {**curr_args}
            # remove any invalid params, will be ignored on non-batch endpoint. used for local models sharing this config but will throw an error on batch endpoint
            for invalid_param in ["n"]:
                batch_args.pop(invalid_param, None)
            for _ in range(curr_args["n"]):
                batch_rows.append({"method": "POST", "url": "/v1/chat/completions", "body": {**batch_args}})

        else:
            curr_resp = client.chat.completions.create(**curr_args)
            curr_resp = curr_resp.to_dict()
            rows.append({"request": cache_args, "response": curr_resp})
            add_to_cache(cache_conn, cache_args, curr_resp)

        curr_n -= curr_args["n"]

    if return_batch:
        return pd.DataFrame.from_records(batch_rows)
    df = pd.DataFrame.from_records(rows)
    df = expand_response_choices(df)

    if cache_df.shape[0] > 0:
        df = pd.concat([df, cache_df])

    return df


def read_jsonls(content):
    lines = []
    for line in content.split("\n"):
        if not line.strip():
            continue
        lines.append(json.loads(line))
    return pd.DataFrame.from_records(lines)


def batch_request(df, client, model, config, output_filename, make_request, cache_conn, google_client=None):
    batch_filename = output_filename + ".BATCH.jsonl"
    batch_job_filename = output_filename + ".BATCH.meta.json"
    batch_result_file = output_filename + ".BATCH.response.jsonl.gz"

    if os.path.exists(batch_result_file):
        return

    initial_sleep = False

    batches = df["prompt"].progress_apply(make_request).tolist()
    batches_df = pd.concat(batches).reset_index(drop=True)
    if batches_df.shape[0] == 0:
        logging.info("already in cache")
        return

    logging.info(f"missing {batches_df.shape[0]} of {df.shape[0]} rows")

    if not os.path.exists(batch_filename):

        batches_df["custom_id"] = "cf" + batches_df.index.astype(str)

        assert "prompt" not in batches_df.columns
        assert "n" not in batches_df.columns

        # write the batch file

        dump_jsonl(batches_df, batch_filename)
    else:
        batches_df = read_jsonl(batch_filename)

    if not os.path.exists(batch_job_filename):
        initial_sleep = True

        if google_client is not None:
            # Upload the JSONL file in OpenAI input format, using regular genai SDK
            batch_file = google_client.files.upload(
                file=batch_filename,
                config=google_types.UploadFileConfig(mime_type="jsonl"),
            )
            batch_file_id = batch_file.name
        else:
            batch_file = client.files.create(file=open(batch_filename, "rb"), purpose="batch")
            batch_file_id = batch_file.id

        batch_job = client.batches.create(
            input_file_id=batch_file_id, endpoint="/v1/chat/completions", completion_window="24h"
        )

        batch_metadata = {
            "file_id": batch_file_id,
            "job_id": batch_job.id,
            "output_file_id": batch_job.output_file_id,
        }
        logging.info(f"{batch_metadata}")
        logging.info(f"writing: {batch_job_filename}")
        with open(batch_job_filename, "w") as fh:
            json.dump(batch_metadata, fh)
    else:
        logging.info(f"reading: {batch_job_filename}")
        with open(batch_job_filename, "r") as fh:
            batch_metadata = json.load(fh)
            logging.info(f"{batch_metadata}")
    # wait for results
    total_wait_s = 15 * 60
    wait_interval = 30

    if initial_sleep:
        time.sleep(wait_interval)
        total_wait_s -= wait_interval

    if not os.path.exists(batch_result_file):

        result = None
        while total_wait_s > 0:

            total_wait_s -= wait_interval
            batch_job = client.batches.retrieve(batch_metadata["job_id"])
            logging.info(f"current job status: {batch_job.status}")
            if batch_job.output_file_id:
                if google_client is None:
                    result = client.files.content(batch_job.output_file_id).content
                    result = read_jsonls(result.decode())
                else:
                    # Download results in OpenAI output format, using regular genai SDK
                    result = google_client.files.download(file=batch_job.output_file_id)
                    result = read_jsonls(result.decode())

                result = result.merge(batches_df, on=["custom_id"])

                dump_jsonl(result, batch_result_file)

                cache_args = dict()
                cache_args.update(config["generation"])
                cache_args.update({"model": model, "client.base_url": str(client.base_url)})

                logging.info(f"attempting to load {result.shape[0]} rows into the cache")
                for _, row in result.iterrows():
                    body = dict(cache_args)
                    body["messages"] = row["body"]["messages"]
                    add_to_cache(cache_conn, body, row["response"]["body"])

                return
            elif batch_job.status == "completed":
                if batch_job.error_file_id:
                    logging.info(f"Error file found: {batch_job.error_file_id}")

                    # Download and read the errors
                    error_content = client.files.content(batch_job.error_file_id).text
                    logging.error(error_content)
                raise AssertionError(
                    "batch job is complete but there is no output file ID. something has gone wrong"
                )
            time.sleep(wait_interval)
    raise ValueError("timeout: still waiting. check back later")


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
    parser.add_argument("--model", required=True, help="The model to choose from the OpenAI platform. ex. gpt-4o")
    parser.add_argument("--generation_config", required=True)
    parser.add_argument("--output_column", default="generated_text")
    parser.add_argument("--api_key_var", default="OPENAI_API_KEY", choices=["OPENAI_API_KEY", "GEMINI_API_KEY"])
    parser.add_argument("--batch_mode", default=False, action="store_true")
    parser.add_argument("--min_repeat", help="minimum number of times to repeat a prompt", type=int, default=1)

    args = parser.parse_args()

    logging.info(f"reading: {args.input}")
    input_df = read_jsonl(args.input, required_columns=["chat"])
    assert (
        input_df.chat.str.len() <= 2
    ).all(), "chats should be user -> assistant, no complex conversations are currently enabled for caching purposes"
    input_df["prompt"] = input_df.chat.apply(lambda chat: chat[0]["content"])
    df = input_df.copy()

    logging.info(f"input columns: {df.columns.tolist()}")
    df["line_index"] = df.index
    assert args.output_column not in df.columns, f"output column ({args.output_column}) cannot exist as an input column"

    config = load_generation_config(args.generation_config, None)

    if args.model == "gemini-2.5-pro-preview-03-25" and config["generation"]["max_new_tokens"] is not None:
        raise AssertionError(
            "Gemini 2.5pro currently has a bug where if you give it a completion token limit it returns an empty response"
        )

    # do some sanity checks on the model + config match up
    if "gemini" in args.model.lower():
        for openai_option in ["reasoning_effort", "verbosity"]:
            if openai_option in config["generation"]:
                raise AssertionError(
                    f"generation config contains openai specific option ({openai_option}) but a gemini model was given"
                )

        if "gemini" not in args.api_key_var.lower():
            raise AssertionError(f"wrong API key for model: {args.model}, {args.api_key_var}")
    else:
        if "thinking_level" in config["generation"]:
            raise AssertionError(
                "generation config contains gemini specific option (thinking_level) but a openai model was given"
            )
        if "openai" not in args.api_key_var.lower():
            raise AssertionError(f"wrong API key for model: {args.model}, {args.api_key_var}")

    if args.api_key_var == "GEMINI_API_KEY":
        client = OpenAI(api_key=os.environ[args.api_key_var], base_url=GOOGLE_GEN_API)
        google_client = genai.Client(api_key=os.environ[args.api_key_var])
    else:
        client = OpenAI(api_key=os.environ[args.api_key_var])
        google_client = None
    df["request.model"] = args.model
    df["request.api"] = args.api_key_var.split("_")[0]
    df = df[["request.model", "request.api", "prompt"]].drop_duplicates()

    logging.info(f"setting up requests caching to sqlite3 db: {LLM_CACHE_DB}")
    cache_conn = cache_setup()
    try:

        def make_batch_request(prompt):
            try:
                resp_df = openai_request(
                    client,
                    prompt,
                    args.model,
                    config,
                    args.min_repeat,
                    cache_conn=cache_conn,
                    return_batch=args.batch_mode,
                )
                return resp_df
            except openai.BadRequestError as err:
                logging.warning(str(err))
                raise AssertionError(
                    "should not be possible in btch mode since we don't actually request anything yet"
                )

        if args.batch_mode:
            batch_request(
                df,
                client,
                model=args.model,
                config=config,
                output_filename=args.output,
                make_request=make_batch_request,
                cache_conn=cache_conn,
                google_client=google_client,
            )
    finally:
        cache_conn.close()

    cache_conn = cache_setup()
    try:

        def make_request(prompt):
            try:
                resp_df = openai_request(
                    client,
                    prompt,
                    args.model,
                    config,
                    args.min_repeat,
                    cache_conn=cache_conn,
                    return_batch=False,
                )
                resp_df["prompt"] = prompt
                return resp_df
            except openai.BadRequestError as err:
                logging.warning(str(err))
                return pd.DataFrame.from_records([{"error": str(err), "prompt": prompt}])

        dfs = df["prompt"].progress_apply(make_request)
    finally:
        cache_conn.close()
    result_df = pd.concat(dfs.tolist())
    result_df["request.model"] = args.model
    result_df["request.api"] = args.api_key_var.split("_")[0]
    for field, value in config["generation"].items():
        result_df[f"request.{field}"] = value
    result_df["config"] = json.dumps(config, sort_keys=True, cls=NpEncoder)

    if "id" in result_df.columns:
        result_df = result_df.drop(columns=["id"])
    result_df = input_df.merge(result_df)  # don't need ID column from db caching

    dump_jsonl(result_df, args.output)


if __name__ == "__main__":
    main()
