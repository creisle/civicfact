import argparse
import glob
import json
import logging
import os
import subprocess
from typing import TypedDict

import numpy as np
import pandas as pd

# https://www.datacamp.com/tutorial/fine-tuning-llama-2
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from transformers.utils.logging import disable_progress_bar
from trl import SFTConfig, SFTTrainer

from prompting import RESPONSE_TEMPLATE_FIELD
from utils import NpEncoder, read_training_config

disable_progress_bar()


class Config(TypedDict):
    quant: dict
    train: dict
    lora: dict


def log_env_vars(*vars):
    for var in vars:
        logging.info(f"{var}: {os.environ.get(var)}")


def get_os_release():
    output = ""
    if os.path.exists("/etc/os-release"):
        output = subprocess.getoutput("cat /etc/os-release")
    return output


def file_exists(filename):
    if os.path.exists(filename):
        return filename
    raise TypeError(f"file {filename} does not exist")


def process_args():
    logging.basicConfig(
        level=logging.INFO,
        format="%(process)d %(asctime)s %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        required=True,
        help='path to training partition JSONL input file which must as minimum contain a "prompt" field. Any remaining fields will be carried over',
        type=file_exists,
    )
    parser.add_argument(
        "--dev",
        required=True,
        help='path to training dev JSONL input file which must as minimum contain a "prompt" field. Any remaining fields will be carried over',
        type=file_exists,
    )
    parser.add_argument(
        "--output",
        required=True,
        help="path to output directory to save the fine-tuned model in",
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
        "--context_length",
        type=int,
        default=None,
        help="pass an extended context length, can only be used with models which se ALIBI",
    )
    parser.add_argument("--tuning_method", required=True, choices=["full", "completion"])
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    args.auth_kwargs = {}
    if os.environ.get("HF_TOKEN"):
        args.auth_kwargs["token"] = os.environ["HF_TOKEN"].strip()
    if not args.model_name:
        args.model_name = args.model_path

    if not args.tokenizer_model:
        args.tokenizer_model = args.model_path

    os_release = get_os_release().replace("\n", "; ")
    logging.info(f"os-release: {os_release}")
    logging.info(f"pid: {os.getpid()}")
    log_env_vars("HOSTNAME", "LOCAL_RANK", "WORLD_SIZE", "CUDA_VISIBLE_DEVICES")
    return args


def get_max_input_length(config):
    context_length = None
    for field in ["max_position_embeddings", "max_seq_len"]:
        if context_length:
            break
        try:
            context_length = getattr(config, field)
        except AttributeError:
            pass
    assert (
        context_length > 0 and context_length < 1e6
    ), f"check that context length is a reasonable number: {context_length}"
    return context_length


def format_args(defaults, log_with_prefix="", **kwargs) -> dict:
    args = dict()
    args.update(defaults)
    args.update(kwargs)
    if log_with_prefix:
        for arg, val in args.items():
            logging.info(f"{log_with_prefix} setting {arg} = {val}")
    return args


def read_datafile(filename, completion=False) -> "tuple[pd.DataFrame, str]":
    logging.info(f"reading {filename}")
    df = pd.read_json(filename, lines=True, dtype=False)
    for col in ["gold_label_name", "id", "template", "prompt.train", "prompt.eval"]:
        if col not in df.columns:
            raise AssertionError(f"data file is missing a required field: {col}")

    if not completion:
        df[RESPONSE_TEMPLATE_FIELD] = ""
    else:
        if RESPONSE_TEMPLATE_FIELD not in df.columns:
            raise AssertionError(f"data file is missing a required field: {RESPONSE_TEMPLATE_FIELD}")
    assert df[RESPONSE_TEMPLATE_FIELD].nunique() == 1
    response_template = df[RESPONSE_TEMPLATE_FIELD].tolist()[0]

    if "assignment.round" in df.columns:
        original_size = df.shape[0]
        redundancy = df.loc[0, "redundancy"]
        if redundancy > 1 and df["assignment.round"].nunique() > 1:
            assert (
                original_size % redundancy == 0
            ), f"the dataframe (n={df.shape[0]}) is not divisible by the redundancy factor (r={redundancy})"
            assignments = df["assignment.round"].unique().tolist()
            logging.info(f"assigned rounds: {assignments}. Filtering on assignment.round={assignments[0]}")
            new_df = df[df["assignment.round"] == assignments[0]]
            assert (
                new_df.shape[0] == df.shape[0] // redundancy
            ), f"picking first assignment round (n={new_df.shape[0]}) does not appear to have matched the expected redundancy factor (r={redundancy}, n={df.shape[0]})"
            logging.warning(
                f"The input file contains redundancy factor(r={redundancy}), filtering from {df.shape[0]} to {new_df.shape[0]} on the first assignment ({assignments[0]}) to avoid slow-down during training"
            )
            df = new_df.reset_index(drop=True)

    # if completion_only then reformat the dataset
    if completion:
        df["prompt"] = df["prompt.eval"]
        df["completion"] = df.apply(lambda row: row["prompt.train"][len(row["prompt.eval"]) :], axis=1)
        df = df[["prompt", "completion"]]
        print(df)
    else:
        df["text"] = df["prompt.train"]
        df = df[["text"]]
    return df, response_template


def main():
    args = process_args()
    logging.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    logging.info(f"torch.cuda.current_device: {torch.cuda.current_device()}")

    # Standard prompt-completion
    # {"prompt": "The sky is",
    # "completion": " blue."}

    df_train, response_template = read_datafile(args.train, args.tuning_method == "completion")
    print(df_train)

    config = read_training_config(args.config, df_train.shape[0])

    logging.info(f"There are {df_train.shape[0]} training examples")
    ds_train = Dataset.from_pandas(df_train)

    df_dev, dev_response_template = read_datafile(args.dev, args.tuning_method == "completion")
    assert (
        dev_response_template == response_template
    ), f"cannot use different response templates in the training ({response_template}) and dev ({dev_response_template}) partitions"
    logging.info(f"There are {df_dev.shape[0]} evaluation examples")
    ds_dev = Dataset.from_pandas(df_dev)

    datasets = DatasetDict({"train": ds_train, "dev": ds_dev})

    quant_config = BitsAndBytesConfig(
        **format_args(
            config["quant"],
            bnb_4bit_compute_dtype=getattr(torch, config["compute_dtype"]),
            log_with_prefix="bnb",
        )
    )

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = config["train"].get("gradient_accumulation_steps", 2)
    if ddp:
        device_map = {"": torch.cuda.current_device()}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    logging.info(f"loading a quantized version of the model: {args.model_name}")
    logging.info(f"device_map: {device_map}")
    logging.info(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    # TODO: is it possible to do this distributed?
    # https://www.datacamp.com/tutorial/fine-tuning-llama-3-1
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=config["compute_dtype"],
        **args.auth_kwargs,
    )
    model.config.use_cache = False  # not using KV cache as it might require more memory
    model.config.pretraining_tp = 1

    try:
        for key, value in model.hf_device_map.items():
            logging.info(f"device map {key} = {value}")
    except AttributeError:
        logging.warning("unable to log device map")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, trust_remote_code=True)

    peft_params = LoraConfig(**format_args(config["lora"], log_with_prefix="lora"))

    context_length = args.context_length if args.context_length else get_max_input_length(model.config)
    logging.info(f"using context_length: {context_length}")

    # https://github.com/tloen/alpaca-lora/blob/main/finetune.py
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    model.config.use_cache = False

    sft_config = dict(
        gradient_accumulation_steps=gradient_accumulation_steps,
        output_dir=args.output,
        ddp_find_unused_parameters=False if ddp else None,
        log_with_prefix="training",
    )
    if args.tuning_method == "completion":
        sft_config["completion_only_loss"] = True

    steps_per_epoch = len(ds_train) // (
        config["train"]["per_device_train_batch_size"]
        * config["train"]["gradient_accumulation_steps"]
        * torch.cuda.device_count()
    )

    training_params = SFTConfig(
        max_length=context_length,
        packing=False,  # packs multiple examples into the same prompt for efficiency
        **format_args(
            config["train"],
            save_steps=steps_per_epoch // config["save_per_epoch"],
            eval_steps=steps_per_epoch // config["eval_per_epoch"],
            logging_steps=steps_per_epoch // config["log_per_epoch"],
            **sft_config,
        ),
    )

    callbacks = []
    if "early_stopping" in config:
        callbacks.append(EarlyStoppingCallback(**config["early_stopping"]))
    sft_args = dict(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        peft_config=peft_params,
        processing_class=tokenizer,
        args=training_params,
        callbacks=callbacks,
    )
    trainer = SFTTrainer(**sft_args)

    if len(glob.glob(os.path.join(args.output, "checkpoint-*/*.pt"))):
        resume_from_checkpoint = True
        logging.warning(f"found previous training checkpoints. Will resume from last checkpoint")
    else:
        resume_from_checkpoint = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(os.path.join(args.output, "best_checkpoint"))

    logging.info(f"The best checkpoint is {trainer.state.best_model_checkpoint}")

    logging.info("saving the model config in the best checkpoint")
    model.config.to_json_file(os.path.join(args.output, "best_checkpoint", "config.json"))
    logging.info(f"memory footprint: {model.get_memory_footprint()}")
    training_params_filename = os.path.join(args.output, "best_checkpoint", "training_params.json")
    logging.info(f"writing: {training_params_filename}")
    try:
        with open(training_params_filename, "w") as fh:
            json.dump(training_params, fh, sort_keys=True, indent="  ", cls=NpEncoder)
    except Exception as err:
        logging.warning(err)


if __name__ == "__main__":
    main()
