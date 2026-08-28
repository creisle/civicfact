import itertools

import pandas as pd
from irrCAC.raw import CAC
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

MINOR_PROBLEMS = {
    'Incomplete',
    'Misleading Or Overstated',
    'Extra Information',
    'Incoherent, Poor Wording, Or Poor Grammar',
}
MAJOR_PROBLEMS = {'External Inconsistency', 'Internal Inconsistency', 'Logic Error'}


def classify_problem(tags):
    if tags & MAJOR_PROBLEMS:
        return 'Major Issues'
    tags -= {'Untagged'}
    if tags & {"Helpful"}:
        tags = tags - {"Helpful"}
        if tags & MINOR_PROBLEMS:
            return 'Helpful (Minor Issues)'
        elif not tags:
            return "Helpful"
    tags -= MINOR_PROBLEMS
    if tags:
        raise NotImplementedError(f'unexpected tags: {tags}')
    return 'Unhelpful'


def categorize_tag(tag):
    if tag in MAJOR_PROBLEMS:
        return 'Major Issue'
    if tag in MINOR_PROBLEMS:
        return 'Minor Issue'
    if tag == 'Helpful':
        return tag
    if tag == 'Untagged':
        return 'Unhelpful'
    raise ValueError(tag)


def compute_pa(df: pd.DataFrame, equality_function=lambda x, y: x == y):
    """
    Ratings dataframe where instances are by row and each column represents a user/rater. Values are expected to be directly comparable
    """

    users = df.columns.tolist()
    agreements = []
    used_users = set()

    for user1, user2 in itertools.combinations(users, 2):
        subset_df = df[[user1, user2]].dropna(how='any')
        if subset_df.shape[0]:
            agreements.append(
                (
                    user1,
                    user2,
                    subset_df[
                        subset_df.apply(
                            lambda row: equality_function(row[user1], row[user2]), axis=1
                        )
                    ].shape[0],
                    subset_df.shape[0],
                )
            )
            used_users.update({user1, user2})

    agg_df = pd.DataFrame(agreements, columns=['user1', 'user2', 'agree', 'total'])
    agg_df['rate'] = agg_df.agree / agg_df.total
    agg_df['weighted_rate'] = agg_df.rate * agg_df.total / agg_df.total.sum()
    return {
        'mean': agg_df.rate.mean(),
        'users': len(used_users),
        'min_examples': agg_df.total.min(),
        'avg_examples': agg_df.total.mean(),
        'std': agg_df.rate.std(),
        'weighted_mean': agg_df.weighted_rate.sum(),
    }


def batch_compute_pa(df: pd.DataFrame, batch_col: str):
    batches = []
    for _, subset_df in df.groupby(batch_col):
        subset_df = subset_df[[c for c in subset_df.columns if c != batch_col]]
        batches.append(compute_pa(subset_df))
    batch_df = pd.DataFrame.from_records(batches)
    return {
        'mean': batch_df['mean'].mean(),
        'std': batch_df['std'].mean(),
        'weighted_mean': batch_df.weighted_mean.mean(),
        'n_batches': df[batch_col].nunique(),
    }


def batch_compute_fleiss(df: pd.DataFrame, batch_col: str) -> pd.DataFrame:
    batches = []
    for _, subset_df in df.groupby(batch_col):
        subset_df = subset_df[[c for c in subset_df.columns if c != batch_col]]
        batches.append(CAC(subset_df).fleiss()['est'])
    batch_df = pd.DataFrame.from_records(batches)
    return batch_df


def binarize_multilabel_fleiss(df, label_col, index_cols, user_cols):

    binarized_df = df.copy()

    mlb = MultiLabelBinarizer()

    binarized_df['tag_class'] = list(mlb.fit_transform(binarized_df[label_col].tolist()))
    binarized_df['tag_class'] = binarized_df['tag_class'].apply(lambda x: tuple(x))

    tag_id_mapping = {
        tag_array: i for i, tag_array in enumerate(binarized_df['tag_class'].unique())
    }
    binarized_df['tag_id'] = binarized_df['tag_class'].apply(lambda t: tag_id_mapping[t])
    binarized_df

    weights_df = pd.DataFrame(
        itertools.product(binarized_df['tag_class'].unique(), binarized_df['tag_class'].unique()),
        columns=['tag1', 'tag2'],
    )
    weights_df['distance'] = weights_df.apply(
        lambda row: jaccard_score(row['tag1'], row['tag2'], zero_division=1), axis=1
    )
    weights_df['tag_id1'] = weights_df['tag1'].apply(tag_id_mapping.get)
    weights_df['tag_id2'] = weights_df['tag2'].apply(tag_id_mapping.get)
    weights_df = pd.pivot_table(weights_df, index='tag_id1', columns='tag_id2', values='distance')

    temp_df = pd.pivot_table(
        binarized_df, values=['tag_id'], aggfunc='first', index=index_cols, columns=user_cols
    )  # shouldn't be dups
    cac = CAC(temp_df, categories=list(tag_id_mapping.values()), weights=weights_df)
    fk = cac.fleiss()
    return fk
