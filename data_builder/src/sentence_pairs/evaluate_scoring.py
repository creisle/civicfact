import argparse
import logging
import math

import pandas as pd
import seaborn as sns

from article_utils.util import read_jsonl
from figutils import apply_to_ax, savefig

K_VALUE = 15


def listify(item):
    try:
        None in item
        return item
    except TypeError:
        return []


def compute_overlap(range1, range2):
    overlap = min(range1[1], range2[1]) - max(range1[0], range2[0])
    if overlap < 0:
        return 0
    return overlap


def is_expected_chunk(row):
    for segment in row['evidence']:
        try:
            if compute_overlap(
                (segment['meta']["start"], segment['meta']["end"]),
                (row['chunk.start'], row['chunk.end']),
            ):
                return True
        except KeyError as err:
            print(row)
            print(segment)
            raise err
    return False


def _calc_kprob(N, k, s, x):
    """
    https://www.britannica.com/topic/hypergeometric-distribution
    P(X=x) = comb(s, x) * comb(N - s, k - x) / comb(N, k)
    - s = size of the desired set (prev k)
    - x = number of elements drawn from desired set
    - k = number of draws ( prev n)
    - N = size of population to be drawn from
    """
    try:
        if k == 0 or s > N or s > k or x > s:
            return 0
        if k > N:
            return 1
        return math.comb(s, x) * math.comb(N - s, k - x) / math.comb(N, k)

    except AssertionError as err:
        raise err
    except ValueError as err:
        logging.error(f"calc_kprob(N={N}, k={k}, s={s}, x={x})")
        raise err


def calc_kprob(N, k, s, strict=True):
    if strict:
        return _calc_kprob(N, k, s, s)

    if k > N:
        return 1
    return 1 - math.comb(N - s, k) / math.comb(N, k)

    # return max([_calc_kprob(N, k, s, x) for x in range(1, s + 1)])


def compute_accuracy(k_df, correct_function, accuracy_criteria):
    instance_sums_df = (
        k_df.groupby(["id", "claim.flat", "document.pmid", 'partition'])
        .agg({"chunk_correct": "sum", "is_expected_chunk": "sum"})
        .reset_index()
    )
    instance_sums_df["correct"] = instance_sums_df.apply(correct_function, axis=1)
    instance_sums_df['freq'] = 1

    dfs = []

    temp_df = (
        instance_sums_df.groupby(['partition']).agg({'correct': 'sum', 'freq': 'sum'}).reset_index()
    )
    temp_df['accuracy'] = temp_df.correct / temp_df.freq
    temp_df['accuracy_level'] = 'instance'
    dfs.append(temp_df)

    claim_sums_df = (
        instance_sums_df.groupby(["claim.flat", "document.pmid", 'partition'])
        .agg({'correct': 'any'})
        .reset_index()
    )
    claim_sums_df['freq'] = 1

    temp_df = (
        claim_sums_df.groupby(['partition']).agg({'correct': 'sum', 'freq': 'sum'}).reset_index()
    )
    temp_df['accuracy'] = temp_df.correct / temp_df.freq
    temp_df['accuracy_level'] = 'claim'
    dfs.append(temp_df)

    df = pd.concat(dfs)
    df['accuracy_criteria'] = accuracy_criteria
    return df


def compute_k_ranks(ranks_df, ranking_method, k_value):
    k_df = ranks_df.copy()
    k_df['freq'] = 1

    dup_check = (
        k_df.groupby(["id", "claim.flat", "document.pmid", 'chunk.id'])
        .agg({'freq': 'sum'})
        .reset_index()
    )
    assert dup_check[dup_check.freq > 1].shape[0] == 0

    k_df["similarity.index"] = k_df.groupby(["id", "claim.flat", "document.pmid"])[
        "similarity.score"
    ].rank(method=ranking_method, ascending=False)
    k_df["similarity.index"] = k_df["similarity.index"] - 1

    assertion_failure = k_df["similarity.index"] >= k_df["document.total_chunks"]

    if assertion_failure.sum() > 0:
        print(k_df[assertion_failure])
        raise AssertionError(
            f'One or more chunks was ranked higher ({k_df["similarity.index"].max()}) than the maximum expected rank ({k_df["document.total_chunks"].max()})'
        )

    k_df["chunk_correct"] = (k_df["similarity.index"] < k_value) & (k_df["is_expected_chunk"])

    df = pd.concat(
        [
            compute_accuracy(
                k_df,
                lambda row: row['chunk_correct'] > 0
                and row['chunk_correct'] == row['is_expected_chunk'],
                'strict',
            ),
            compute_accuracy(k_df, lambda row: row['chunk_correct'] > 0, 'partial'),
            compute_accuracy(
                k_df, lambda row: row['chunk_correct'] / row['is_expected_chunk'], 'fractional'
            ),
        ]
    )

    df['ranking_method'] = ranking_method
    df['k'] = k_value

    return df[
        ['accuracy', 'accuracy_criteria', 'accuracy_level', 'k', 'ranking_method', 'partition']
    ]


def compute_ranks(ranks_df):
    ranks_df = ranks_df.copy()
    ranks_df['is_expected_chunk'] = ranks_df.apply(is_expected_chunk, axis=1)

    dfs = []

    # calculate the accuracy at different k values
    for k_value in range(K_VALUE):
        for ranking_method in ['min', 'max']:
            dfs.append(compute_k_ranks(ranks_df, ranking_method, k_value))

    df = pd.concat(dfs)
    df['model'] = 'similarity'
    return df


def compute_prob_ranks(df):
    df = df.copy()
    df['is_expected_chunk'] = df.apply(is_expected_chunk, axis=1)
    df = (
        df.groupby(["id", "claim.flat", "document.pmid", 'partition'])
        .agg({"is_expected_chunk": "sum", "document.total_chunks": "first"})
        .reset_index()
    )

    k_dfs = []

    # calculate the probable accuracy at different k values
    for k_value in range(K_VALUE):
        prob_dfs = []
        k_prob_df = df.copy()
        k_prob_df["k"] = k_value
        k_prob_df["accuracy"] = k_prob_df.apply(
            lambda row: calc_kprob(
                row['document.total_chunks'],
                k_value,
                row['is_expected_chunk'],
                True,
            ),
            axis=1,
        )
        k_prob_df["accuracy_criteria"] = "strict"
        k_prob_df["accuracy_level"] = "instance"
        prob_dfs.append(k_prob_df.copy())

        k_prob_df["accuracy"] = k_prob_df.apply(
            lambda row: calc_kprob(
                row['document.total_chunks'], k_value, row['is_expected_chunk'], False
            ),
            axis=1,
        )
        k_prob_df["accuracy_criteria"] = "partial"
        k_prob_df["accuracy_level"] = "instance"
        prob_dfs.append(k_prob_df)

        k_prob_df = (
            k_prob_df.groupby(["claim.flat", "document.pmid", 'partition'])
            .agg({"is_expected_chunk": "max", "document.total_chunks": "first"})
            .reset_index()
        )
        k_prob_df["k"] = k_value
        k_prob_df["accuracy"] = k_prob_df.apply(
            lambda row: calc_kprob(
                row['document.total_chunks'], k_value, row['is_expected_chunk'], True
            ),
            axis=1,
        )
        k_prob_df["accuracy_criteria"] = "strict"
        k_prob_df["accuracy_level"] = "claim"
        prob_dfs.append(k_prob_df.copy())

        k_prob_df["accuracy"] = k_prob_df.apply(
            lambda row: calc_kprob(
                row['document.total_chunks'], k_value, row['is_expected_chunk'], False
            ),
            axis=1,
        )
        k_prob_df["accuracy_criteria"] = "partial"
        k_prob_df["accuracy_level"] = "claim"
        prob_dfs.append(k_prob_df)

        prob_df = (
            pd.concat(prob_dfs)
            .groupby(["k", 'partition', "accuracy_criteria", "accuracy_level"])
            .agg({"accuracy": "mean"})
            .reset_index()
        )
        k_dfs.append(prob_df)

    df = pd.concat(k_dfs)
    df['ranking_method'] = 'n/a'
    df["model"] = "baseline probability"
    return df


sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--document_chunks",
    required=True,
    help="jsonl file containing the definitions for every document chunk",
)
parser.add_argument("--data", required=True, help='claim/evidence pairs')
parser.add_argument('--scores', required=True, help='claim/chunk scores')
parser.add_argument("--output", required=True)
args = parser.parse_args()

chunks_df = read_jsonl(args.document_chunks).rename(columns={'pmid': 'document.pmid'})


df = read_jsonl(args.data)
print(df.partition.unique())

df = df[(df.generated == False) & (df['evidence.source'] != 'civic')][
    ['id', 'display_id', 'claim.flat', 'evidence', 'document.pmid']
]

# evidence_chunks_overlaps, claim_evidence_lookup = create_lookups(df, chunks_df)
# df =

scores_df = read_jsonl(args.scores)

ranks_df = df.merge(scores_df)
ranks_df = ranks_df.merge(
    chunks_df.drop(columns=['passages', 'pubtator', 'selections', 'acronyms'])
)
print(ranks_df.columns)
print(ranks_df.partition.value_counts(dropna=False))
print('document.total_chunks:', ranks_df['document.total_chunks'].max())

dfs = [compute_ranks(ranks_df), compute_prob_ranks(ranks_df)]


data_df = pd.concat(dfs)
print(data_df[data_df.k < 10])

# print(data_df.partition.unique())

# claim_acc_df = (
#     data_df.groupby(
#         ["k", "claim.flat", "pmid", 'partition', 'ranking_method'],
#         dropna=False,
#     )
#     .agg({"correct": "any"})
#     .reset_index()
#     .rename(columns={"correct": "claim_correct"})
# )
# claim_acc_df["claim_freq"] = 1
# claim_acc_df = (
#     claim_acc_df.groupby(["k", "partition", 'ranking_method'], dropna=False)
#     .agg({"claim_correct": "sum", "claim_freq": "sum"})
#     .reset_index()
# )

# df = (
#     data_df.groupby(["k", 'partition', 'ranking_method'])
#     .agg({"correct": "sum", "freq": "sum"})
#     .reset_index()
# ).merge(claim_acc_df)
# df["example"] = df.correct / df.freq
# df["claim"] = df.claim_correct / df.claim_freq
# print(df)

# df = pd.melt(
#     df,
#     id_vars=['k', 'partition', 'ranking_method', 'model'],
#     value_vars=['example', 'claim'],
#     var_name='accuracy_type',
#     value_name='accuracy',
# )
chunk_size = chunks_df['chunk.size'].mode()[0]
chunk_overlap = chunks_df['chunk.overlap'].mode()[0]
data_df['k_characters'] = data_df.k * chunk_size

fig = sns.relplot(
    data=data_df,
    y="accuracy",
    x="k_characters",
    kind="line",
    style='model',
    hue='partition',
    col='accuracy_criteria',
)
fig.figure.suptitle(f'chunk_size={chunk_size} & chunk_overlap={chunk_overlap}', y=1.05, va='bottom')

apply_to_ax(fig.axes, lambda ax: ax.grid(True, which="both", axis="x"))


savefig(fig, args.output)
