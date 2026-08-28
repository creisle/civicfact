"""
This was written for clarity and is not optimized for speed. If using to train regularly or on your own dataset this should be further optimized to use numpy
"""

import numpy as np
from article_utils.util import read_jsonl, dump_jsonl
from figutils import savefig
import tqdm
import logging
import pandas as pd
import os
import seaborn as sns
import re
import argparse
import itertools
import glob
import random


def get_ngrams(claim: str, gram_size: int, include_headers: bool = True):
    if not include_headers:
        claim = claim.strip().split('\n')[-1]
    claim = claim.replace('\n', ' ').replace('\t', ' ')

    claim = re.sub(r'\s+', ' ', claim, re.MULTILINE)
    assert '\t' not in claim
    terms = [t for t in claim.split(' ') if t.strip()]

    grams = []

    for i in range(0, max([0, len(terms) - gram_size])):
        grams.append(tuple(terms[i + j] for j in range(gram_size)))
    return set(grams)


def compute_bias(df):
    """
    df: three column dataframe with bigram, gold_label_name, and alpha values
    """
    df = df.explode('bigram').reset_index()
    df['alpha'] = (
        df['alpha'] * df['freq'] + df['freq']
    )  # number of times the claim appears (to ensure alpha is identical per claim/label pair)
    num = (
        df.groupby(['bigram', 'gold_label_name'])
        .agg({'alpha': 'sum'})
        .reset_index()
        .rename(columns={'alpha': 'b_num'})
    )
    denom = (
        df.groupby(['bigram'])
        .agg({'alpha': 'sum'})
        .reset_index()
        .rename(columns={'alpha': 'b_denom'})
    )
    df = num.merge(denom)
    df['bias'] = df['b_num'] / df['b_denom']

    return df


def l2_norm(series):
    return np.linalg.norm(series, ord=2)  # lambda * L2 norm


def objective(df, lambda_coeff):
    """
    df: three column dataframe with bigram, gold_label_name, and alpha values
    """
    bias_df = compute_bias(df)

    # take max label bias per class
    bias_df = bias_df.groupby(['bigram']).agg({'bias': 'max'}).reset_index()
    return bias_df.bias.sum() + lambda_coeff * l2_norm(df['alpha'])


def training_loop(input_df, max_steps, eps, learning_rate, min_alpha, lambda_coeff, early_stopping):

    curr_state_df = input_df.copy()

    curr_state_df = (
        curr_state_df.groupby(['claim.flat', 'gold_label_name', 'alpha'])
        .agg({'bigram': 'first', 'row_i': 'nunique'})
        .reset_index()
        .rename(columns={'row_i': 'freq'})
    )
    curr_state_df['grad'] = 0.0

    history = [{'loss': objective(curr_state_df, lambda_coeff)}]
    best_alpha = curr_state_df
    best_step = 0
    best_loss = None

    for step_i in range(max_steps):
        base = objective(curr_state_df, lambda_coeff)
        logging.info(f'step: {step_i + 1}/{max_steps} loss={base}')
        step_df = curr_state_df.copy()

        for i, _ in step_df.iterrows():
            # calculate diff for each alpha value change independent of other changes
            row_df = step_df.copy()
            row_df.at[i, 'alpha'] += eps
            step_df.at[i, 'grad'] = (objective(row_df, lambda_coeff) - base) / eps

        step_df['alpha'] -= learning_rate * step_df['grad']
        step_df.loc[step_df.alpha < min_alpha, 'alpha'] = min_alpha

        new_base = objective(step_df, lambda_coeff)

        if best_loss is not None and new_base >= best_loss:
            logging.warning(f'loss is not decreasing. {new_base - base}')
        else:
            best_alpha = step_df
            best_step = step_i
            best_loss = new_base

        history.append({'loss': new_base, 'keep': best_loss is None or new_base < best_loss})
        if (step_i - best_step) > early_stopping:
            logging.warning(f'Early stopping. Last {early_stopping} steps did not improve loss')
            break
        curr_state_df = step_df
    history_df = pd.DataFrame.from_records(history)
    history_df['step'] = history_df.index.tolist()
    history_df['lambda_coeff'] = lambda_coeff
    history_df['eps'] = eps
    history_df['learning_rate'] = learning_rate
    history_df['min_alpha'] = min_alpha
    history_df['early_stopping'] = early_stopping
    return best_alpha, history_df


def pick_initial_alpha_state(df, method='zeros'):
    if method == 'inverse':
        bias_df = (
            df.groupby(['claim.flat', 'gold_label_name'])
            .agg({'bigram': 'first', 'row_i': 'nunique'})
            .reset_index()
            .rename(columns={'row_i': 'freq'})
        )
        bias_df['alpha'] = 0.0
        bigram_df = compute_bias(bias_df)
        bias_df = (
            bias_df.explode('bigram')
            .merge(bigram_df, on=['bigram', 'gold_label_name'])
            .groupby(['claim.flat', 'gold_label_name'])
            .agg({'bias': 'sum'})
            .reset_index()
        )
        bias_df['alpha'] = 1 / bias_df['bias']
        df = df.merge(bias_df[['claim.flat', 'gold_label_name', 'alpha']])

        return df
    elif method == 'random':
        df['alpha'] = df.apply(lambda _: random.uniform(0, 1), axis=1)
    elif method == 'zeros':
        df['alpha'] = 0.0
    else:
        raise NotImplementedError(method)
    return df


def learn(
    input_file,
    output_folder,
    continue_training=False,
    downsample=None,
    random_seed=42,
    include_headers=True,
    **kwargs,
):

    DATA_OUTPUT_FILE = os.path.join(output_folder, 'data.with_bias.jsonl.gz')
    HISTORY_OUTPUT_FILE = os.path.join(output_folder, 'training_history.jsonl.gz')

    init_method = kwargs.pop('init_method', 'zeros')

    sampled = False

    if os.path.exists(DATA_OUTPUT_FILE) and os.path.exists(HISTORY_OUTPUT_FILE):
        if not continue_training:
            logging.warning(f'Must select contiunue training or else we expect an empty directory')
            return
        raise NotImplementedError('need to update to truncate history to last kept alpha state')
        previous_history_df = read_jsonl(HISTORY_OUTPUT_FILE)
        df = read_jsonl(DATA_OUTPUT_FILE)
        for col in ['alpha', 'bigram', 'row_i']:
            assert col in df.columns
        df['bigram'] = df.bigram.apply(
            lambda grams: [tuple(gram) for gram in grams]
        )  # dumping to json makes them lists which we can't hash
        assert 'alpha'
    else:
        previous_history_df = None
        df = read_jsonl(input_file)
        df = df[df.partition.isin({'train', 'dev'})]
        if downsample is not None:
            sampled = True
            df = df.sample(n=downsample, random_state=random_seed)
        df['row_i'] = df.index.tolist()
        df['bigram'] = df['claim.flat'].apply(
            lambda claim: get_ngrams(claim, 2, include_headers=include_headers)
        )

        df = pick_initial_alpha_state(df, init_method)
        logging.info('initializing alpha to 0')

    input_df = df.copy()

    df, history_df = training_loop(input_df, **kwargs)

    if sampled:
        history_df['downsample'] = downsample
        history_df['random_seed'] = random_seed

    if previous_history_df is not None:
        history_df['step'] += previous_history_df.step.max()
        history_df = pd.concat([previous_history_df, history_df])
        assert (
            history_df.groupby(['step']).agg({'loss': 'nunique'}).reset_index()['loss'].max() == 1
        )
        # history_df = history_df.drop_duplicates(['step'], keep='first')
    else:
        # started from sratch
        history_df['init_method'] = init_method
        history_df['include_headers'] = include_headers

    df = input_df.drop(columns=['alpha']).merge(df[['alpha', 'claim.flat', 'gold_label_name']])
    if df.shape[0] != input_df.shape[0]:
        logging.warning(f'number of claims changed: {df.shape[0]} vs {input_df.shape[0]}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sns.set_style('whitegrid')
    fig = sns.histplot(df[df.alpha > 0], x='alpha', hue='gold_label_name', multiple='stack')
    savefig(fig, os.path.join(output_folder, 'hist.png'))

    fig = sns.lineplot(history_df, x='step', y='loss')
    savefig(fig, os.path.join(output_folder, 'training_loss.png'))

    dump_jsonl(df, DATA_OUTPUT_FILE)
    dump_jsonl(history_df, HISTORY_OUTPUT_FILE)


def parameter_search(input_file, output_folder, max_steps, early_stopping):

    conditions = {
        'min_alpha': [0],
        'learning_rate': [0.5, 0.2, 0.1],
        'lambda_coeff': [0.01],  # , 0.001],
        'eps': [0.05, 0.01],  # , 0.001],
        'init_method': ['zeros'],
        'random_seed': [1, 2, 42],
        'include_headers': [False],
    }
    variables = sorted(conditions)

    tests = list(itertools.product(*[conditions[c] for c in variables]))
    for i in range(len(tests)):
        tests[i] = {c: v for c, v in zip(variables, tests[i])}
    random.shuffle(tests)
    print(f'runnning {len(tests)} tests')

    for test_state in tests:
        print('test_state:', test_state)
        # max_steps = int(round(MAX_STEPS * max(learning_rates) / lr))
        output_folder = f'{output_folder}/test-{test_state["learning_rate"]}-{test_state["lambda_coeff"]}-{test_state["eps"]}-{test_state["min_alpha"]}-{early_stopping}-{test_state["random_seed"]}-{test_state["init_method"]}'
        if not test_state['include_headers']:
            output_folder += '--noheaders'
        learn(
            input_file,
            output_folder,
            max_steps=max_steps,
            early_stopping=early_stopping,
            downsample=100,
            **test_state,
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='builds/civicfact-2025.03.25/data.jsonl.gz')
    parser.add_argument('--max_steps', default=1000, type=int)
    parser.add_argument('output_folder')
    parser.add_argument(
        '--early_stopping',
        default=100,
        type=int,
        help='number of steps (epochs) to run without getting a better loss before quitting',
    )
    parser.add_argument('--parameter_search', default=False, action='store_true')

    args = parser.parse_args()

    if args.parameter_search:
        parameter_search(args.input, args.output_folder, args.max_steps, args.early_stopping)
    else:
        learn(
            args.input,
            args.output_folder,
            max_steps=args.max_steps,
            early_stopping=args.early_stopping,
            lambda_coeff=0.001,
            eps=0.05,
            learning_rate=0.2,
            init_method='random',
            min_alpha=0,
            include_headers=False,
        )
