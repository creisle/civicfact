import json
import logging
import os
import subprocess
from uuid import UUID

import numpy as np
import pandas as pd

INVALID_PUBMED_IDS = {'27577079': '28947956', '27556863': '28978004', '22713161': '25998099'}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)

        try:
            if np.isnan(obj):
                return None
        except TypeError:
            pass
        try:
            return obj.to_json()
        except AttributeError:
            pass
        return super(NpEncoder, self).default(obj)


def dump_records(df) -> list[dict]:
    df = df.astype(object).where(pd.notnull(df), None)
    records = df.to_dict('records')
    return records


def dump_json_to_str(obj):
    return json.dumps(obj, sort_keys=True, cls=NpEncoder)


def dump_jsonl(df, filename):
    """
    pandas does some weird escaping if you use its built-in method
    """
    records = dump_records(df)
    if filename.endswith('.gz'):
        jsonl_output = filename[:-3]
    else:
        jsonl_output = filename
    if not os.path.exists(os.path.dirname(filename)) and os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename))
    logging.info(f'writing: {jsonl_output}')
    with open(jsonl_output, 'w') as fh:
        rows = []
        for record in records:
            try:
                rows.append(dump_json_to_str(record))
            except TypeError as err:
                logging.error(f'failed to JSON serialize the following record: {record}')
                raise err
        fh.write('\n'.join(rows) + '\n')
    if filename.endswith('.gz'):
        if os.path.exists(filename):
            os.remove(filename)
        logging.info(f'compressing {jsonl_output}')
        subprocess.check_call(['gzip', '-n', jsonl_output])


def read_jsonl(
    *filenames, lines=True, dtype=False, add_filename: bool = False, required_columns=None, **kwargs
):
    dfs = []
    for filename in filenames:
        logging.info(f'reading: {filename}')
        try:
            df = pd.read_json(filename, lines=lines, dtype=dtype, **kwargs)
        except UnicodeDecodeError as err:
            rows = []
            with open(filename, 'r') as fh:
                logging.error(f'reading line by line to find source of decoding error: {err}')
                for i, line in enumerate(fh.readlines()):
                    try:
                        rows.append(json.loads(line))
                    except UnicodeDecodeError as sub_err:
                        logging.error(f'error on line index: {i}')
                        raise sub_err
            df = pd.DataFrame.from_records(rows)

            if df.shape[0] and required_columns:
                for col in required_columns:
                    if col not in df.columns:
                        logging.error(f'missing required column: {col}')

        if add_filename:
            df['filename'] = filename
        dfs.append(df)
    return pd.concat(dfs)


def normalize_and_expand_column(df, column):
    normal_df = pd.json_normalize(df[column])
    normal_df = normal_df.rename(columns={c: f'{column}.{c}' for c in normal_df.columns})
    return pd.concat([df, normal_df], axis=1).drop(columns=[column])


def listify(item):
    if isinstance(item, list):
        return item
    if pd.isnull(item):
        return []
    if isinstance(item, str) and item == '':
        return []
    raise ValueError(f'dont know what to do with this type: {type(item)} repr({repr(item)})')


def redundant_columns(df, col1, col2):
    return df[[col1, col2]].dropna(how='all')[df[col1] != df[col2]].shape[0] == 0


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
                    print('cannot merge, columns have diff values in some cases')
                    print(df[df[x_col] != df[col]][[x_col, col]].drop_duplicates())
            else:
                renames[x_col] = col
        else:
            print('cannot merge, columns have diff values in some cases')
            print(df[df[x_col] != df[y_col]][[x_col, y_col]].drop_duplicates())

    df = df.rename(columns=renames)
    # for any renames, if the target column already exists, repeat the previous sameness check
    df = df.drop(columns=drops)
    return df
