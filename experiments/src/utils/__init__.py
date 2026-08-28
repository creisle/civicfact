import json
import logging
import math
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import torch

# import all the figure helper functions
from .fig import *

EXPECTED_LABELS = {"SUPPORTS", "REFUTES", "NEI", "REVISIONS"}
SIMLIFIED_MODEL_NAMES = {
    "meta-llama__Llama-2-chat-hf": "Llama2-chat",
    "mosaicml__mpt-chat": "MPT-chat",
    "microsoft__Phi-3-medium-128k-instruct": "Phi-3",
}
SELECTION_DELIMITER = " . . . "


def normalize_and_expand_column(df, column):
    normal_df = pd.json_normalize(df[column])
    normal_df = normal_df.rename(columns={c: f"{column}.{c}" for c in normal_df.columns})
    return pd.concat([df, normal_df], axis=1).drop(columns=[column])


def get_os_release():
    output = ""
    if os.path.exists("/etc/os-release"):
        output = subprocess.getoutput("cat /etc/os-release")
    return output


def file_exists(filename):
    if os.path.exists(filename):
        return filename
    raise TypeError(f"file {filename} does not exist")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        import torch

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, torch.Tensor):
            return obj.item()
        if isinstance(obj, torch.dtype):
            return str(obj)
        if isinstance(obj, pd.arrays.ArrowStringArray):
            return obj.tolist()
        try:
            if np.isnan(obj):
                return None
        except TypeError:
            pass
        try:
            return obj.to_json()
        except AttributeError:
            pass
        try:
            return dict(obj)
        except TypeError:
            pass
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super(NpEncoder, self).default(obj)


def dump_records(df) -> "list[dict]":
    df = df.astype(object).where(pd.notnull(df), None)
    records = df.to_dict("records")
    return records


def dump_json_to_str(obj):
    return json.dumps(obj, sort_keys=True, cls=NpEncoder)


def dump_jsonl(df, filename):
    """
    pandas does some weird escaping if you use its built-in method
    """
    records = dump_records(df)
    if filename.endswith(".gz"):
        jsonl_output = filename[:-3]
    else:
        jsonl_output = filename
    logging.info(f"writing: {jsonl_output}")
    with open(jsonl_output, "w") as fh:
        rows = []
        for record in records:
            try:
                rows.append(dump_json_to_str(record))
            except TypeError as err:
                logging.error(f"failed to JSON serialize the following record: {record}")
                raise err
            except ValueError as err:
                values = {k: type(v) for k, v in record.items()}
                logging.error(f"failed to JSON serialize the following record: {values}")
                raise err
        fh.write("\n".join(rows) + "\n")
    if filename.endswith(".gz"):
        if os.path.exists(filename):
            os.remove(filename)
        logging.info(f"compressing {jsonl_output}")
        subprocess.check_call(["gzip", "-n", jsonl_output])


def load_models(model_path, tokenizer_model=None, model_name=None, max_seq_len=None):
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # device_index = (
    #     Accelerator().process_index
    # )  # https://github.com/huggingface/accelerate/issues/1840#issuecomment-1683105994
    # device_map = {"": device_index}
    auth_kwargs = {}
    if os.environ.get("HF_READ_TOKEN"):
        auth_kwargs["token"] = os.environ["HF_READ_TOKEN"]
    if not model_name:
        model_name = model_path

    # if this is a local model, log the training state so we have a copy to refer back to
    if os.path.exists(os.path.join(model_path)):
        with open(os.path.join(model_path, "trainer_state.json"), "r") as fh:
            trainer_state = json.load(fh)
            logging.info(f'trainer state: global_step = {trainer_state["global_step"]}')
    os_release = get_os_release().replace("\n", "; ")
    logging.info(f"os-release: {os_release}")
    logging.info(f'HOSTNAME: {os.environ.get("HOSTNAME")}')

    if not tokenizer_model:
        tokenizer_model = model_path

    # Download config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, **auth_kwargs)

    if max_seq_len and "mpt-" in model_path.lower():
        config.max_seq_len = max_seq_len
    # config.init_device = 'meta'  # For fast initialization directly on GPU!

    dtype = torch.bfloat16  # or torch.float32
    # Download model source and weights
    logging.info(f"loading model: {model_path}")

    logging.info("download checkpoint file")
    checkpoint_file = snapshot_download(repo_id=model_path, repo_type="model", **auth_kwargs)

    # see: https://huggingface.co/docs/accelerate/usage_guides/big_modeling
    logging.info("initializing the empty model")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_file,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            **auth_kwargs,
        )

    for key, value in model.hf_device_map.items():
        logging.info(f"device map {key} = {value}")
    # logging.info('setting up the device mapping')
    # device_map = infer_auto_device_map(model, dtype=dtype, verbose=True)

    logging.info("loading the model checkpoint")
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_file,
        device_map=model.hf_device_map,
        dtype=dtype,
    )

    logging.info(f"loading tokenizer model: {tokenizer_model}")
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, **auth_kwargs)

    return model, tokenizer


def is_plausible(text):
    return text.strip().upper() in EXPECTED_LABELS | {"NOT ENOUGH INFORMATION"}


def cleanup_generated_text(generated_text):
    # preprocess to remove non-word characters
    subs = [(r"[^A-Za-z0-9.:]", " "), (r"\s+", " "), (r"\n+", " "), (r"\*", ""), (r"\s+:", ":")]
    for sub, repl in subs:
        generated_text = re.sub(sub, repl, generated_text, re.MULTILINE)
    generated_text = generated_text.strip()
    return generated_text


def get_all_predicted_labels(generated_text, strict=False) -> list[tuple[str, int, str]]:
    # preprocess to remove non-word characters

    generated_text = cleanup_generated_text(generated_text)
    labels = []
    if generated_text.lower().startswith("not enough information"):
        labels.append(("NEI", 0, "not enough information"))

    for label in EXPECTED_LABELS:
        if generated_text.upper().startswith(label):
            labels.append((label, 0, None))

    for label in EXPECTED_LABELS:
        m = re.search(r"\b" + label + r"\b", generated_text)
        if m:
            labels.append((label, m.start(), None))

    if strict:
        return sorted(labels, key=lambda x: x[1])

    for substring, label in [
        ("evidence does not directly support or refute the claim", "NEI"),
        ("evidence does not support the claim", "REFUTES"),
        ("evidence is supportive", "SUPPORTS"),
        ("evidence provided does not directly support or refute the claim", "NEI"),
        ("evidence provided support", "SUPPORTS"),
        ("evidence supports the claim", "SUPPORTS"),
        ("insufficient information", "NEI"),
        ("neither supports nor refutes the claim", "NEI"),
        ("not enough information to determine the stance", "NEI"),
        ("refutes", "REFUTES"),
        ("supports", "SUPPORTS"),
    ]:
        if substring in generated_text.lower():
            index = generated_text.lower().find(substring)
            labels.append((label, index, substring))

    hypothetical = r"(?P<hypothetical>((if\s[^.]*)|([cw]ould be considered))\s[^.]*)?"
    negation = r"(?P<negated>(doesn'?t)|not|nor)?\s*(((?!(doesn'?t)|not|nor)).)*"
    label_patt = r"\s*(?P<label>\w+|not enough information)\b"

    patterns = {
        "NEI": [r"does not (mention|specify|address)", r"does not (provide|have) enough information"],
        "SUPPORTS": [],
        "REFUTES": [
            r"(?P<label>opposes|contradicts) (the|this) claim",
        ],
    }
    for label in patterns:
        for pattern in patterns[label]:
            m = re.search(hypothetical + pattern, generated_text, re.IGNORECASE)
            if m:
                labels.append((label, m.start(), pattern))

    for pattern in [
        negation + r"\"?(?P<label>refutes?|supports?)\"?( (the|this))? ((given|underlying) )?(claim|statement)",
        r"(?P<negated>not be )?labeled as " + label_patt,
        r"(stance|label) of (the|this) evidence (towards? the claim )?[^.]*(is|appears to be):? \W?" + label_patt,
        r"\blabel (this|the) evidence as:?" + label_patt,
        r"^(?P<label>(refut|support)[a-z]*)\b",
        r"^(label|answer|reply|teacher|response|respond)s?( as)?\s*: " + label_patt,
        r"^\s*(evidence|label|stance): " + label_patt,
        r"based on the evidence, the label is:? " + label_patt,
        r"claim (that [^.]+ )?(is|would be|should be|might be) (?P<label>supported|refuted)",
        r"evidence has (?P<label>\w+|not enough information) label",
        r"evidence is (?P<label>\w+|not enough information) to the claim",
        r"evidence is labell?ed as \s*\W?" + label_patt,
        r"evidence labels the claim as " + label_patt,
        r"I (would )?choose the label " + label_patt,
        r"i conclude that the evidence (?P<label>\w+) the claim.",
        r"label (for this ((claim|evidence|case) )?)?(is|would be|should be|might be) \W?" + label_patt,
        r"label for this (claim|evidence) (and (claim|evidence) )?(is|would be|should be|might be) \W?" + label_patt,
        r"label: \s*(?P<label>\w+)\b",
        r"label( (is|would be))?( indeed)?:? " + label_patt,
        r"stance (of the evidence )?(toward the claim )?is:? " + label_patt,
        r"stance toward claim:\s*(?P<label>\w+)\b",
        r"the (evidence )?label for the (\w+ )?claim is:? " + label_patt,
        r"the evidence is (?P<label>supports|refutes|nei|not enough information)",
        r"the label for the relationship between them is " + label_patt,
        r"the stance (toward the claim )?is\s*:? " + label_patt,
        r"would (?P<negated>not )?label (it|the stance) (of the evidence )?(toward the claim )?as\s*:?" + label_patt,
    ]:
        m = re.search(hypothetical + pattern, generated_text, re.IGNORECASE)
        if m:
            label = m.group("label").upper().strip()
            if m.group("hypothetical"):
                # the explanation is likely postulating about a hypothetical scenario and not talking about the current state of the evidence
                continue
            # if not label.startswith("N") and not label.endswith("S"):
            #     label = label + "S"
            negated = ""
            try:
                negated = m.group("negated")
            except IndexError:
                pass
            if negated:
                continue
            if label.startswith("SUPPORT"):
                label = "SUPPORTS"
            if label.startswith("REFUT"):
                label = "REFUTES"
            if label == "NOT ENOUGH INFORMATION" or label.startswith("NEI") or label.startswith("uncertain"):
                label = "NEI"
            labels.append((label, m.span("label")[0], pattern))

    return sorted(labels, key=lambda x: x[1])


def get_predicted_label(generated_text, strict=False) -> str:

    labels = get_all_predicted_labels(generated_text, strict)

    filtered_labels = [l for l in labels if l[0] in EXPECTED_LABELS]
    if filtered_labels:
        return filtered_labels[0][0]
    if not labels:
        return ""
    return labels[0][0]


def pop_model_size(model_name) -> "tuple[str, float]":
    assert not pd.isnull(model_name), f"null input for pop_model_size"
    if "stanford-crfm__BioMedLM" == model_name:
        return model_name, 3
    if "microsoft__Phi-3-medium-128k-instruct" == model_name:
        return model_name, 14
    if "google-bert__bert-large-uncased" == model_name:
        return model_name, 0.336
    if "google-bert__bert-base-uncased" == model_name:
        return model_name, 0.1
    if "intfloat__multilingual-e5-large-instruct" == model_name:
        return model_name, None
    if "microsoft__BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" == model_name:
        return model_name, None
    if "microsoft__MediPhi-PubMed" == model_name:
        return model_name, 3.8 * 7
    if model_name.startswith("openai__") or model_name.startswith("google__"):
        return model_name, None

    patt = r"([-_]|(?P<copies>\d+)x)(?P<model_size>\d+)[bB](-|_|\b|$)"
    m = re.search(patt, model_name)
    assert m, f"failed to find size from model name: {model_name}"
    size = float(m.group("model_size"))

    if m.group("copies"):
        size = size * int(m.group("copies"))
    return re.sub(patt, "-", model_name), size


def process_prompting_result(df, groupby: "Optional[list[str]]" = None, agg=None):
    df["freq"] = 1
    if not groupby:
        groupby = ["filename"]
    agg_funcs = {
        "freq": "sum",
        "correct": "sum",
        "plausible": "sum",
        **{col: "first" for col in df.columns if col not in {"freq", "correct", "plausible", *groupby}},
    }
    agg_funcs.update(agg or {})
    totals_df = df.groupby(groupby, dropna=False).agg(agg_funcs).reset_index()
    totals_df["accuracy"] = 100 * totals_df.correct / totals_df.freq
    totals_df["plausibility"] = 100 * totals_df.plausible / totals_df.freq
    # assert totals_df.shape[0] == 1
    return totals_df


def read_prompting_result(filename):
    _, quant_status, model, dataset, chat_template, base_filename = filename.rsplit("/", 5)
    _, template, shots, partition, _ = base_filename.split(".")
    shots = int(shots.replace("-shot", ""))
    logging.info(f"reading: {filename}")
    with open(filename, "r") as fh:
        df = pd.read_json(fh, lines=True, dtype=False)
    df["filename"] = filename
    df["shots"] = shots

    df["dataset"] = dataset
    df["dataset_version"] = df["dataset"].str.split("-").str[1]
    df["dataset_name"] = df["dataset"].str.split("-").str[0]
    df["dataset_variant"] = df["dataset"].str.extract(r"^civicfact(\w+)-").fillna("")
    df.loc[df.dataset_variant == "ft", "dataset_variant"] = ""
    df["dataset"] = df.apply(lambda row: row["dataset"].replace(row["dataset_variant"] + "-", "-"), axis=1)

    df["quant"] = quant_status.replace("_evaluations", "")

    df["model"] = model
    df["template"] = template
    df["partition"] = partition
    df["chat_template"] = chat_template

    # {model_dir}--{dataset}--prompts.{template}.{examples}-shot.train.jsonl--{tuning_method}--config-{learning_config}
    for col in [
        "base_model",
        "ft_dataset",
        "ft_prompts",
        "ft_tuning_method",
        "ft_learning_config",
    ]:
        df[col] = "none"

    base_model, parameters = pop_model_size(model)
    df["base_model"] = base_model
    df["parameters"] = parameters
    df["predicted_label_name"] = df["generated_text"].apply(get_predicted_label)
    df["plausible"] = df["predicted_label_name"].apply(is_plausible)
    df["correct"] = df.predicted_label_name == df.gold_label_name
    df["input_file"] = filename
    df["freq"] = 1
    return df


def read_ft_prompting_result(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    fields = [
        "ft_learning_config",
        "ft_tuning_method",
        "model",
        "ft_dataset",
        "ft_chat_template",
        "ft_prompts",
        "",
        "dataset",
        "chat_template",
        "prompts_file",
    ]

    data = {}

    for field, value in zip(fields[::-1], filename.split("/")[::-1]):
        if field:
            data[field] = value

    try:
        _, template, shots, partition, _ = re.sub(r"\.bk$", "", data["prompts_file"]).split(".")
    except ValueError as err:
        logging.error(f'failed to split prompts file: {data["prompts_file"]}')
        raise err
    shots = int(shots.replace("-shot", ""))
    logging.info(f"reading: {filename}")
    with open(filename, "r") as fh:
        df = pd.read_json(fh, lines=True, dtype=False)
    df["filename"] = filename

    for field, value in data.items():
        df[field] = value
    df["shots"] = shots
    df["template"] = template
    df["partition"] = partition
    base_model, parameters = pop_model_size(data["model"])
    df["base_model"] = base_model
    df["parameters"] = parameters
    df["predicted_label_name"] = df["generated_text"].apply(get_predicted_label)
    df["plausible"] = df["predicted_label_name"].apply(is_plausible)
    df["correct"] = df.predicted_label_name == df.gold_label_name
    df["input_file"] = filename
    df["freq"] = 1

    # post-process dataset name
    df["dataset_version"] = df["dataset"].str.split("-").str[1]
    df["dataset_variant"] = df["dataset"].str.extract(r"^civicfact(\w+)-").fillna("")
    df.loc[df.dataset_variant == "ft", "dataset_variant"] = ""
    df["dataset"] = df.apply(lambda row: row["dataset"].replace(row["dataset_variant"] + "-", "-"), axis=1)
    df["dataset_name"] = df["dataset"].str.split("-").str[0]

    df["ft_dataset"] = df.apply(lambda row: row["ft_dataset"].replace(row["dataset_variant"] + "-", "-"), axis=1)

    return df


def read_dataset(filename):
    _, dataset, base_filename = filename.rsplit("/", 2)
    _, template, shots, partition, _ = base_filename.split(".")
    shots = int(shots.replace("-shot", ""))
    logging.info(f"reading: {filename}")
    df = pd.read_json(filename, lines=True, dtype=False)
    df["shots"] = shots
    df["dataset"] = dataset
    df["template"] = template
    df["partition"] = partition
    return df


def read_jsonl_file_from_tarball(tarball, filename):
    tar = tarfile.open(tarball)
    with tempfile.TemporaryDirectory() as tmpdirname:
        tar.extract(filename, path=tmpdirname)
        df = pd.read_json(os.path.join(tmpdirname, os.path.basename(filename)), lines=True, dtype=False)
        return df


def calculate_optimal_step_size(training_dataset_size: int, batch_size: int, min_splits: int, max_splits: int):
    """
    check/save stuff min min_splits times per epoch and up to max_splits times per epoch. Prefer the smallest size
    """
    assert (
        min_splits < max_splits and min_splits > 0 and max_splits > 0
    ), f"min_splits: {min_splits}; max_splits: {max_splits}"
    total_steps_per_epoch = math.ceil(training_dataset_size / batch_size)
    max_step_size = total_steps_per_epoch // min_splits
    min_step_size = int(round(total_steps_per_epoch / max_splits))
    assert min_step_size < max_step_size
    return min_step_size


def read_training_config(config_path, training_dataset_size: int):
    logging.info(f"reading: {config_path}")
    with open(config_path, "r") as fh:
        data = json.load(fh)

    for section in ["train", "lora", "quant"]:
        data.setdefault(section, {})

    # if "saves_per_epoch" in data:
    #     step_size = calculate_optimal_step_size(
    #         training_dataset_size,
    #         data["train"]["per_device_train_batch_size"],
    #         data["saves_per_epoch"]["min"],
    #         data["saves_per_epoch"]["max"],
    #     )
    #     logging.info(f"calculated an optimal step size of: {step_size}")
    #     data["train"].setdefault("logging_steps", step_size)
    #     data["train"].setdefault("eval_steps", step_size)
    #     data["train"].setdefault("save_steps", step_size)
    #     data["train"].setdefault("eval_strategy", "steps")
    #     data["train"].setdefault("save_strategy", "steps")
    return data


@dataclass
class Text:
    content: Union[str, list[tuple[str, str]]]
    type: Literal["table", "unstructured"] = "unstructured"
    meta: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(json.dumps(self))

    def to_json(self):
        return {"type": self.type, "content": self.content, "meta": self.meta}

    def concat(self, relation_delim=" is ", column_delim="; "):
        if self.type == "table":
            result = []
            for col, val in self.content:
                if col == "evidenceType":
                    val = convert_enum_to_words(val)
                if val:
                    result.append(f"{convert_col_to_words(col)}{relation_delim}{val}")
            return column_delim.join(result)
        return self.content


def convert_col_to_words(col_name) -> str:
    words = re.sub(
        r"([a-z])([A-Z])",
        lambda pat: pat.group(1) + " " + pat.group(2).lower(),
        col_name,
    )
    return words


def convert_enum_to_words(enum):
    if re.match(r"^([A-Z]|/|_|[0-9])+$", enum) and "_" in enum:
        result = re.sub(r"_", " ", enum)
        result = result.lower()
        return result
    return enum


def concat_text(text_list: list[Text], delimiter=SELECTION_DELIMITER, **kwargs):
    text_list = [Text(**t) if not isinstance(t, Text) else t for t in text_list]
    elements = [t.concat(**kwargs) for t in text_list]
    return delimiter.join([e.strip("\n") for e in elements])


def parse_filepath_metadata(filename):
    subsampling_pattern = r"_f(?P<subsample_fraction>[10]\.\d+)(_(?P<subsample_filter>hard|long\d+|nonabstract|realclaims))?_s(?P<subsample_seed>\d+)"
    retreival_model_patt = (
        r"-(?P<ret_model>(([a-z]+-)*[a-z]+)|random\d+|oracle)-(?P<ret_length>L\d+)-qt(?P<ret_query_template>\d+)"
    )
    subpatterns = {
        "train_data": r"(?P<train_dataset>[^-_]+)-(?P<train_data_version>[^/]+)",
        "eval_data": r"(?P<eval_dataset>([^/_-]+))("
        + retreival_model_patt
        + r")?("
        + subsampling_pattern
        + r")?-(?P<data_version>[^/-]+)",
    }
    patterns = [
        (
            r"results/(?P<task>claim_verification)/fine-tuning/config-(?P<config>\d+)/(?P<method>seqclass)/(?P<embedding_mode>default|random)(?P<embedding_seed>\d+)?/(?P<pairing_mode>paired|unpaired)/"
            + subpatterns["train_data"]
            + r"/(?P<model_dir>[^/]+)/evaluations/"
            + subpatterns["eval_data"]
            + r"/data.(?P<partition>\w+)\.[^/]+$"
        ),
        (
            r"results/(?P<task>claim_verification)/fine-tuning/config-(?P<config>\d+)/(?P<method>\w+)/"
            + subpatterns["train_data"]
            + r"/chat-(?P<train_chat_template>\w+)/prompts.(?P<train_prompt_template>\w+).(?P<train_examples>\d+)-shot.train.jsonl/(?P<model_dir>[^/]+)/evaluations/"
            + subpatterns["eval_data"]
            + r"/chat-(?P<chat_template>\w+)/(?P<generation_config>gen-\d+)/prompts.(?P<prompt_template>\w+).(?P<examples>\d+)-shot.(?P<partition>\w+)\.[^/]+$"
        ),
        (
            r"results/(?P<task>claim_verification)/(?P<compute_location>api/)?evaluations(?P<quantized>_noquant)?/(?P<model_dir>[^/]+)/"
            + subpatterns["eval_data"]
            + r"/chat-(?P<chat_template>\w+)/(?P<generation_config>gen-\d+)/prompts.(?P<prompt_template>\w+).(?P<examples>\d+)-shot.(?P<partition>\w+)\.[^/]+$"
        ),
        (
            r"results/(?P<task>claim_verification)/fine-tuning/config-(?P<config>\d+)/(?P<method>seqclass)/(?P<embedding_mode>default|random)(?P<embedding_seed>\d+)?/(?P<pairing_mode>paired|unpaired)/"
            + subpatterns["train_data"]
            + r"/(?P<model_dir>[^/]+)/checkpoints/"
        ),
        (
            r"results/(?P<task>claim_verification)/fine-tuning/config-(?P<config>\d+)/(?P<method>\w+)/"
            + subpatterns["train_data"]
            + r"/chat-(?P<train_chat_template>\w+)/prompts.(?P<train_prompt_template>\w+).(?P<train_examples>\d+)-shot.train.jsonl/(?P<model_dir>[^/]+)/checkpoints/"
        ),
        (
            r"results/(?P<task>claim_verification)/pretokenize/(?P<model_dir>[^/]+)/"
            + subpatterns["eval_data"]
            + r"/chat-(?P<chat_template>\w+)/prompts\.(?P<prompt_template>\w+)\.(?P<examples>\d+)-shot\.(?P<partition>\w+)\."
        ),
        # retrieve AND verify
        # ex. results/claim_verification/api/evaluations/openai__gpt-5.2/civicfactft_f0.01_s42-2026.03.03/chat-default/gen-8/fulltextretrieveandverify.0-shot/qt0/data.dev.retrieval_accuracy.jsonl.gz
        (
            r"results/(?P<task>claim_verification)/(?P<compute_location>api/)?evaluations(?P<quantized>_noquant)?/(?P<model_dir>[^/]+)/"
            + subpatterns["eval_data"]
            + r"/chat-(?P<chat_template>\w+)/(?P<generation_config>gen-\d+)/(?P<prompt_template>\w+).(?P<examples>\d+)-shot/qt(?P<query_template>\d+)/data.(?P<partition>[^./]+)"
        ),
        # bi-encoder FT eval
        (
            r"results/(?P<task>evidence_retrieval)/(?P<encoding>bi)/fine-tuning/(?P<train_data_format>[^/]+)/config-(?P<config>\d+)/"
            + subpatterns["train_data"]
            + r"/qt(?P<train_query_template>\d+)/(?P<model_dir>[^/]+)"
            + r"("
            + r"/evaluations/"
            + subpatterns["eval_data"]
            + r"/qt(?P<query_template>\d+)/[^.]+.(?P<partition>[^.]+)"
            + r")?"
        ),
        ## crossencoder FT eval
        (
            r"results/(?P<task>evidence_retrieval)/(?P<encoding>cross)/fine-tuning/(?P<train_data_format>[^/]+)/config-(?P<config>\d+)/"
            + subpatterns["train_data"]
            + r"/qt(?P<train_query_template>\d+)/(?P<model_dir>[^/]+)"
            + r"("
            + r"/evaluations/"
            + subpatterns["eval_data"]
            + r"/qt(?P<query_template>\d+)/[^.]+.(?P<partition>[^.]+)"
            + r")?"
        ),
        # cross/bi encoder eval
        (
            r"results/(?P<task>evidence_retrieval)/(?P<encoding>cross|bi)/evaluations/(?P<model_dir>[^/]+)(/"
            + subpatterns["eval_data"]
            + r"/qt(?P<query_template>\d+)/[^.]+.(?P<partition>[^.]+))?"
        ),
        # explanation data results/explanations/civicfact-2026.03.03/n5000_s42/openai__gpt-5.2/gen-8/prompts.jsonl.gz
        (
            r"results/explanations/(?P<eval_dataset>\w+)-(?P<data_version>\d+\.\d+\.\d+\w*)/f(?P<subsample_ratio>\d+\.\d+)_s(?P<subsample_seed>\d+)/(?P<model_dir>[^/]+)/(?P<generation_config>gen-\d+)/prompts.jsonl.gz"
        ),
    ]

    groups = None
    for patt in patterns:
        m = re.search(patt, filename)
        if m:
            groups = m.groupdict()
            break
    if groups is None:
        raise AssertionError(f"failed to match metadata for filename: {filename}")

    groups = dict(groups)
    if "eval_dataset" in groups:
        if groups["eval_dataset"] == "civicfactnl":
            groups["eval_dataset_claim_format"] = "nl"
            groups["eval_dataset"] = "civicfact"
        elif groups["eval_dataset"] == "civicfactisa":
            groups["eval_dataset_claim_format"] = "isa"
            groups["eval_dataset"] = "civicfact"

    if "subsample_fraction" in groups and groups["subsample_fraction"] is not None:
        groups["subsample_fraction"] = groups["subsample_fraction"].rstrip("0").rstrip(".")
    return groups


def collect_filepath_metadata(filename):

    metadata = parse_filepath_metadata(filename)
    if metadata.get("ret_model"):
        if not re.match(r"^random\d+$", metadata["ret_model"]):
            path_nicknames_file = os.path.join(os.path.dirname(__file__), "../../data/pathnames.json")
            logging.debug(f"loading path nicknames: {path_nicknames_file}")
            with open(path_nicknames_file, "r") as fh:
                path_nicknames = json.load(fh)
            try:
                ret_meta = parse_filepath_metadata(path_nicknames[metadata["ret_model"]])
                metadata.update({f"ret_{k}": v for k, v in ret_meta.items()})
            except KeyError as err:
                print(metadata)
                raise err
        else:
            metadata["ret_model_dir"] = "random"
            metadata["ret_model_random_seed"] = metadata["ret_model"][len("random") :]
    return metadata


def read_jsonl(
    *files,
    add_filename=False,
    add_filepath_metadata=False,
    required_columns: list[str] = [],
    warning_columns: list[str] = [],
    unique_columns: list[str] = [],
    static_columns: list[str] = [],
    **kwargs,
):
    for filename in files:
        assert os.path.exists(filename)

    dfs = []
    for filename in files:
        logging.info(f"reading: {filename}")
        df = pd.read_json(filename, lines=True, dtype=False, **kwargs)
        for col in required_columns:
            assert col in df.columns, f"missing required column: {col}"
        for col in warning_columns:
            if col not in df.columns:
                logging.warning(f"missing column: {col}")

        for col in unique_columns:
            assert df[col].nunique() == df.shape[0], f"{col} is not a unique index"

        for col in static_columns:
            assert df[col].nunique() == 1, f"{col} has more than 1 value"

        if add_filename:
            df["filename"] = filename
        if add_filepath_metadata:
            meta = collect_filepath_metadata(filename)
            for k, v in meta.items():
                df[f"filepath.{k}"] = v
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def load_generation_config(filename, dynamic_max_new_tokens: int = None):
    logging.info(f"reading: {filename}")
    with open(filename, "r") as fh:
        config = json.load(fh)

    if "system_prompt_file" in config:
        with open(os.path.join(os.path.dirname(filename), config["system_prompt_file"]), "r") as fh:
            config["system_prompt"] = fh.read().strip()

    if "max_new_tokens" not in config["generation"]:
        if "min_new_tokens" not in config or config["min_new_tokens"] is not None:
            max_new_tokens = max([config.get("min_new_tokens", 1), dynamic_max_new_tokens or 1])
            config["generation"]["max_new_tokens"] = max_new_tokens

    if "quantization_config" in config:
        if "bnb_4bit_compute_dtype" in config["quantization_config"]:
            config["quantization_config"]["bnb_4bit_compute_dtype"] = getattr(
                torch, config["quantization_config"]["bnb_4bit_compute_dtype"]
            )

    if "torch_dtype" in config:
        config["torch_dtype"] = getattr(torch, config["torch_dtype"])

    for k, v in config["generation"].items():
        logging.info(f"generation config: {k}={v}")

    return config


def redundant_columns(df, col1, col2):
    return df[[col1, col2]].dropna(how="all")[df[col1] != df[col2]].shape[0] == 0


def fix_bad_df_merge(df):
    df = df.copy()
    x_cols = [c for c in df.columns if c.endswith("_x")]
    drops = []
    renames = {}
    for x_col in x_cols:
        y_col = x_col[:-2] + "_y"
        if y_col not in df.columns:
            continue
        if redundant_columns(df, x_col, y_col):
            # always the same, should have merged
            drops.append(y_col)
            col = x_col[:-2]

            if col in df.columns:
                if redundant_columns(df, col, x_col):
                    drops.append(x_col)
                else:
                    print("cannot merge, columns have diff values in some cases")
                    print(df[df[x_col] != df[col]][[x_col, col]].drop_duplicates())
            else:
                renames[x_col] = col
        else:
            print("cannot merge, columns have diff values in some cases")
            print(df[df[x_col] != df[y_col]][[x_col, y_col]].drop_duplicates())

    df = df.rename(columns=renames)
    # for any renames, if the target column already exists, repeat the previous sameness check
    df = df.drop(columns=drops)
    return df


def convert_with_template(claim: dict) -> str:
    evidence_type = claim["evidenceType"].lower()
    evidence_direction = claim["evidenceDirection"].upper()
    assert evidence_direction in {
        "SUPPORTS",
        "DOES_NOT_SUPPORT",
        "N/A",
    }, f"unexpected evidenceDirection: {evidence_direction}"

    irr_verb = "is"
    negated = evidence_direction == "DOES_NOT_SUPPORT"

    molecular_profile = (
        claim["molecularProfile"].replace(" OR ", " or ").replace(" AND ", " and ").replace(" NOT ", " not ")
    )
    template = ""
    significance = claim["significance"]

    if " or " in molecular_profile:
        irr_verb = "are"
        if "variantOrigin" in claim:
            if claim["variantOrigin"] == "RARE_GERMLINE":
                molecular_profile += " rare germline variants"
            if claim["variantOrigin"] == "COMMON_GERMLINE":
                molecular_profile += " common germline variants"
    elif " and " in molecular_profile:
        molecular_profile = f"a molecular profile consisting of {molecular_profile}"
        if "variantOrigin" in claim:
            if claim["variantOrigin"] == "RARE_GERMLINE":
                molecular_profile += " rare germline variants"
            if claim["variantOrigin"] == "COMMON_GERMLINE":
                molecular_profile += " common germline variants"
    elif "variantOrigin" in claim:
        if claim["variantOrigin"] == "RARE_GERMLINE":

            molecular_profile = f"The rare germline variant, {molecular_profile},"
        if claim["variantOrigin"] == "COMMON_GERMLINE":
            molecular_profile = f"The common germline variant, {molecular_profile},"

    drugs = ""
    if "therapies" in claim and claim["therapies"]:
        if len(claim["therapies"]) == 1:
            drugs = claim["therapies"][0]
        elif len(claim["therapies"]) > 1:
            assert "therapyInteractionType" in claim
            interaction_type = claim["therapyInteractionType"]
            if interaction_type == "SUBSTITUTES":
                if len(claim["therapies"]) > 2:
                    drugs = "any of "
                drugs += ", ".join(claim["therapies"][:-1]) + ", or " + claim["therapies"][-1]
            if interaction_type == "COMBINATION":
                drugs = "a combination of " + ", ".join(claim["therapies"][:-1]) + ", and " + claim["therapies"][-1]

    disease = claim.get("disease", "")
    negation_term = "" if not negated else "not "
    if evidence_type == "predictive":
        template += (
            f"{molecular_profile} {irr_verb} {negation_term}predictive of {significance} to {drugs} in {disease}"
        )
    elif evidence_type == "functional":
        if negated:
            action = "does not result in"
        else:
            action = "results in"
        if claim["significance"].endswith("ic"):
            template += f"{molecular_profile} {action} {significance} functionality"
        elif claim["significance"].startswith("u"):
            template += f"{molecular_profile} {action} {significance}"
        else:
            template += f"{molecular_profile} {action} a {significance}"
    elif claim["significance"] == "na":
        template += f"{molecular_profile} {irr_verb} {negation_term}a {evidence_type} indicator in {disease}"
    elif evidence_type == "diagnostic":
        template += (
            f"{molecular_profile} {irr_verb} {negation_term}a {significance} {evidence_type} indicator in {disease}"
        )
    elif evidence_type == "prognostic":
        template += (
            f"{molecular_profile} {irr_verb} {negation_term}associated with a prognosis of {significance} in {disease}"
        )
    elif claim["significance"] == "predisposition":
        template += f"{molecular_profile} {irr_verb} {negation_term}associated with {significance} to {disease}"
    elif evidence_type == "predisposing":
        if claim["significance"].startswith("uncertain"):
            if negated:
                irr_verb = "does not have"
            else:
                irr_verb = "has"
        template += f"{molecular_profile} {irr_verb} {significance} in {disease}"
    else:
        template += f"{molecular_profile} {irr_verb} {negation_term}associated with {significance} in {disease}"

    if "phenotypes" in claim:
        phenotypes = ", ".join(claim["phenotypes"])
        template += f" accompanied by the {phenotypes}"
        if len(claim["phenotypes"]) > 1:
            template += " phenotypes"
        else:
            template += " phenotype"

    return template


NUM_QUERY_TEMPLATES = 8


def apply_query_template(row, query_column="claim.flat", template: int = 0):
    if template == 0:
        return row[query_column]
    if template == 1:  # col: val\n
        lines = []
        for col, values in sorted(row["claim"]["content"].items()):
            if isinstance(values, list):
                values = "; ".join(values)
            lines.append(f"{col}: {values}")
        return "\n".join(lines)
    if template == 2:  # val\nval
        lines = []
        for _, values in sorted(row["claim"]["content"].items()):
            if isinstance(values, list):
                values = "; ".join(values)
            lines.append(values)
        return "\n".join(lines)
    if template == 3:  # col\tcol\nval\tval
        header = []
        table_row = []
        for col, values in sorted(row["claim"]["content"].items()):
            if isinstance(values, list):
                values = "; ".join(values)
            header.append(col)
            table_row.append(values)
        return "\t".join(header) + "\n" + "\t".join(table_row)
    if template == 4:  # natural language
        return convert_with_template(row["claim"]["content"])
    if template == 5:  # col is value;
        lines = []
        for col, values in sorted(row["claim"]["content"].items()):
            if isinstance(values, list):
                values = ", ".join(values)
            lines.append(f"{col} is {values}")
        return "; ".join(lines)
    if template == 6:  # val; val
        lines = []
        for _, values in sorted(row["claim"]["content"].items()):
            if isinstance(values, list):
                values = ", ".join(values)
            lines.append(values)
        return "; ".join(lines)
    if template == 7:  # molecularProfile only
        return row["claim"]["content"]["molecularProfile"]
    raise NotImplementedError(f"do not recognize template number: {template}")


def compute_overlap(range1, range2):
    overlap = min(range1[1], range2[1]) - max(range1[0], range2[0])
    if overlap < 0:
        return 0
    return overlap


def smart_merge(df1, df2, merge_cols=None, how="inner"):
    """
    pandas is an idiot for types when one column is default null. fix this by precasting shared columns
    """
    df1 = df1.copy()
    df2 = df2.copy()
    if merge_cols is None:
        merge_cols = list(set(df1.columns) & set(df2.columns))

    for col in merge_cols:
        dtype1 = str(df1[col].dtype)
        dtype2 = str(df2[col].dtype)
        cast_type = list({dtype1, dtype2} - {"object", "str"})
        if len(cast_type) == 1:
            cast_type = cast_type[0]
            df1[col] = df1[col].astype(cast_type)
            df2[col] = df2[col].astype(cast_type)
    return df1.merge(df2, on=merge_cols, how=how)


def pick_largest_subsampling(df):
    df = df.copy()
    df["filepath.subsample_fraction"] = df["filepath.subsample_fraction"].astype(float)

    key_count_cols = [
        "filepath.model_dir",
        "filepath.train_data_version",
        "filepath.data_version",
        "filepath.eval_dataset",
        "filepath.train_dataset",
    ]
    key_count_cols = [c for c in key_count_cols if c in df.columns]
    previous_count = df[key_count_cols].drop_duplicates().shape[0]
    # pick the largest subsmaple when all else is equal
    filepath_columns = [
        c
        for c in df.columns
        if c.startswith("filepath.") and c not in ["filepath.subsample_fraction", "filepath.subsample_seed"]
    ]

    print("filepath_columns:", filepath_columns)
    sub_df = df.sort_values(["filepath.subsample_fraction"]).drop_duplicates(filepath_columns, keep="last")
    temp_df = df.merge(sub_df[["filename"]])
    assert previous_count == temp_df[key_count_cols].drop_duplicates().shape[0]
    print("reduced rows from", df.shape[0], "to", temp_df.shape[0])
    return temp_df


def load_json_as_kv_lines(json_str):
    claim = json.loads(json_str)
    claim_lines = []
    for key in claim:
        value = json.dumps(claim[key], sort_keys=True)
        claim_lines.append(f'"{key}": {value}')
    return claim_lines


def str_diff_revision(ref, alt):
    if alt == "none" or not alt:
        return None, None
    claim = load_json_as_kv_lines(ref)
    revision = load_json_as_kv_lines(alt)

    all_lines = sorted(set(claim) | set(revision))

    diff = []
    for line in all_lines:
        if line in claim and line in revision:
            diff.append("    " + line)
        elif line in claim:
            diff.append("-   " + line)
        elif line in revision:
            diff.append("+   " + line)
        else:
            raise AssertionError("line must be in claim or revision")
    diff = ",\n".join(diff)
    return "{\n" + diff + "\n}"


pass
