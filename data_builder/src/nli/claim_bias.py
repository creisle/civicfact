import warnings
import itertools
import math
import pandas as pd

import numpy as np

from .util import CIVIC_CLAIM_COL_ORDER, Text


def get_ngrams(claim: Text, gram_size: int = 2):
    if isinstance(claim, list):
        warnings.warn(f'deprecated claim format, older claims were given as lists')
        text = ' '.join(itertools.chain.from_iterable(claim[0]['content']))
    else:
        text = []
        for col, val in sorted(
            claim['content'].items(), key=lambda key: CIVIC_CLAIM_COL_ORDER.index(key[0])
        ):
            text.append(f'{col} {val}')
        text = ' '.join(text)
    terms = [t for t in text.split(' ') if t.strip()]

    grams = []

    for i in range(0, max([0, len(terms) - gram_size])):
        grams.append(tuple(terms[i + j] for j in range(gram_size)))
    return set(grams)


def compute_lmi_fast(input_df, grams_col='bigram', label_col='gold_label_name'):
    df = input_df.copy()

    if 'alpha' not in df.columns:
        df['alpha'] = 0.0
    df['alpha'] = df['alpha'].fillna(0.0)

    # Track original rows so we can count "rows containing gram" (pairs_count) efficiently
    df = df.reset_index(drop=False).rename(columns={'index': '_row_index'})

    # One row per (original_row, label, gram)
    ex = df[['_row_index', label_col, 'alpha', grams_col]].explode(grams_col)
    ex['_w'] = 1.0 + ex['alpha'].astype(float)  # your weighting

    # Weighted counts per (label, gram)
    lg = ex.groupby([label_col, grams_col], dropna=False)['_w'].sum()

    # Weighted counts per gram and per label
    g = lg.groupby(level=1).sum()
    l = lg.groupby(level=0).sum()
    D = float(l.sum())

    # Presence-based counts:
    # - bigram_label_rows: number of original rows with this (label, gram)
    # - label_freq: number of labels where gram appears at least once (presence, not weight sign)
    pres_lg = (
        ex.drop_duplicates(['_row_index', label_col, grams_col])
        .groupby([label_col, grams_col])
        .size()
    )
    label_freq = pres_lg.reset_index().groupby(grams_col)[label_col].nunique()

    labels = lg.index.get_level_values(0).unique()
    grams = g.index
    full_index = pd.MultiIndex.from_product([labels, grams], names=[label_col, grams_col])

    lg_full = lg.reindex(full_index, fill_value=0.0)
    pres_full = pres_lg.reindex(full_index, fill_value=0)

    mat = lg_full.to_numpy().reshape(len(labels), len(grams))
    count_gram = g.reindex(grams).to_numpy()
    count_gram_mat = np.tile(count_gram, (len(labels), 1))

    count_label = l.reindex(labels).to_numpy()
    count_label_mat = np.tile(count_label.reshape(-1, 1), (1, len(grams)))

    with np.errstate(divide='ignore', invalid='ignore'):
        # your "bias" is count(label,gram) / count(gram)
        bias = np.where(count_gram_mat != 0, mat / count_gram_mat, np.nan)

        ratio = np.where(
            (mat != 0) & (count_gram_mat != 0) & (count_label_mat != 0) & (D != 0),
            (mat / count_gram_mat) / (count_label_mat / D),
            np.nan,
        )

        lmi = np.where(np.isfinite(ratio) & (ratio > 0), (mat / D) * np.log(ratio), np.nan)

        gram_prev = np.where(D != 0, count_gram_mat / D, np.nan)

    out = pd.DataFrame(
        {
            label_col: np.repeat(labels, len(grams)),
            grams_col: np.tile(grams.to_numpy(), len(labels)),
            'lmi': (lmi.reshape(-1) * 1e4),
            'bias': bias.reshape(-1),
            'p(l|w) - p(l)': bias.reshape(-1) - 1 / len(labels),
            'p(w)': gram_prev.reshape(-1),
            'bigram_label_rows': pres_full.to_numpy(),
            'bigram_freq': (count_gram_mat.reshape(-1) / df.shape[0]),
            'label_freq': np.tile(label_freq.reindex(grams, fill_value=0).to_numpy(), len(labels)),
        }
    )

    return out.sort_values(['lmi'], ascending=False)


def compute_lmi(input_df, grams_col='bigram', label_col='gold_label_name'):
    input_df = input_df.copy()
    if 'alpha' not in input_df.columns:
        input_df['alpha'] = 0.0
    grams_by_label = {l: {} for l in input_df[label_col].unique()}

    input_df['alpha'] = input_df['alpha'].fillna(0)
    for _, row in input_df.iterrows():
        hist = grams_by_label[row[label_col]]

        for gram in row[grams_col]:
            hist[gram] = hist.get(gram, 0) + 1.0 + row['alpha']

    D = sum([sum(g.values()) for g in grams_by_label.values()])
    all_grams = set().union(*grams_by_label.values())

    def _compute_lmi(gram, label):
        count_gram_and_label = grams_by_label[label].get(gram, 0)
        count_gram = sum([grams_by_label[l].get(gram, 0) for l in grams_by_label])
        count_label = sum(grams_by_label[label].values())

        try:
            lmi = (count_gram_and_label / D) * math.log(
                (count_gram_and_label / count_gram) / (count_label / D)
            )
        except ValueError:

            lmi = None
        return (count_gram_and_label / count_gram), lmi, count_gram / D

    rows = []

    for label in grams_by_label:
        for gram in all_grams:
            p_wl, lmi, gram_prev = _compute_lmi(gram, label)
            pairs_count = input_df[
                (input_df[label_col] == label)
                & (input_df[grams_col].apply(lambda grams: gram in grams))
            ].shape[0]
            rows.append(
                {
                    label_col: label,
                    grams_col: gram,
                    'lmi': lmi * 1e4 if lmi is not None else None,
                    'bias': p_wl,
                    'p(l|w) - p(l)': p_wl - 1 / len(grams_by_label),
                    'p(w)': gram_prev,
                    'bigram_label_rows': pairs_count,
                    'bigram_freq': (sum([grams_by_label[l].get(gram, 0) for l in grams_by_label]))
                    / input_df.shape[0],
                    'label_freq': sum([gram in grams_by_label[l] for l in grams_by_label]),
                }
            )

    lmi_df = pd.DataFrame.from_records(rows).sort_values(['lmi'], ascending=False)
    return lmi_df


def compute_bias(df, grams_col='bigram', label_col='gold_label_name'):
    """
    df: three column dataframe with bigram, gold_label_name, and alpha values
    """
    df = df.copy()
    df['row_index'] = df.index.tolist()
    df = df.explode(grams_col).reset_index()
    df = df.drop_duplicates([grams_col, label_col, 'row_index'])
    if 'alpha' not in df.columns:
        df['alpha'] = 0
    if 'freq' not in df.columns:
        df['freq'] = 1
    df['alpha'] = (
        df['alpha'] * df['freq'] + df['freq']
    )  # number of times the claim appears (to ensure alpha is identical per claim/label pair)
    num = (
        df.groupby([grams_col, label_col])
        .agg({'alpha': 'sum'})
        .reset_index()
        .rename(columns={'alpha': 'b_num'})
    )
    denom = (
        df.groupby([grams_col])
        .agg({'alpha': 'sum', 'row_index': 'nunique', label_col: 'nunique'})
        .reset_index()
        .rename(columns={'alpha': 'b_denom', label_col: 'bigram_unique_labels'})
    )
    df = num.merge(denom)
    df['bias'] = df['b_num'] / df['b_denom']
    df['bigram_rows'] = df['row_index']
    df['bigram_freq'] = df['row_index'] / df.shape[0]
    df = df.merge(
        df.groupby(grams_col)
        .agg({label_col: 'nunique'})
        .reset_index()
        .rename(columns={label_col: 'label_freq'})
    )

    return df.drop(columns=['row_index'])


def l2_norm(series):
    return np.linalg.norm(series, ord=2)  # lambda * L2 norm


def objective(df, lambda_coeff):
    """
    df: three column dataframe with bigram, gold_label_name, and alpha values
    """
    df = df.copy()
    if 'alpha' not in df.columns:
        df['alpha'] = 0

    bias_df = compute_bias(df)

    # take max label bias per class
    bias_df = bias_df.groupby(['bigram']).agg({'bias': 'max'}).reset_index()
    return bias_df.bias.sum() + lambda_coeff * l2_norm(df['alpha'])
