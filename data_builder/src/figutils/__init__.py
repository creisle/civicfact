import argparse
import os

import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

# convenience exports
from .util import savefig

__all__ = ['savefig']

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MIN_COMMENT_SIZE = 1000
BC_CANCER_PALETTE = sns.color_palette(
    ['#2ec7d6', '#9e4dab', '#3b4486', '#F6AE2D', '#F2542D', '#3A3B3C']
)


def add_bar_labels(fig, **kwargs):
    if isinstance(fig, Axes):
        for c in range(len(fig.axes.containers)):
            fig.axes.bar_label(fig.axes.containers[c], **kwargs)
    else:
        for i in range(len(fig.axes)):
            try:
                for j in range(len(fig.axes[i])):
                    for c in range(len(fig.axes[i][j].containers)):
                        fig.axes[i][j].bar_label(fig.axes[i][j].containers[c], **kwargs)
            except TypeError:
                for c in range(len(fig.axes[i].containers)):
                    fig.axes[i].bar_label(fig.axes[i].containers[c], **kwargs)


def package_path(*paths):
    return os.path.join(BASE_DIR, *paths)


def filepath(path):
    if not os.path.exists(path):
        raise TypeError(f'File not found: {path}')
    return path


def basic_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--output',
        '-o',
        help='path to the output file to write the results to',
        required=True,
    )
    parser.add_argument(
        '--input',
        '-i',
        help='path to the input file',
        required=True,
        type=filepath,
    )

    return parser


def plot_confusion_matrix(df, rows_col, cols_col, **kwargs):
    fig_df = df.copy().sort_values([rows_col, cols_col])
    fig_df['freq'] = 1
    fig_df = fig_df.groupby([cols_col, rows_col]).agg({'freq': 'sum'}).reset_index()
    sums_df = (
        fig_df.groupby([rows_col])
        .agg({'freq': 'sum'})
        .reset_index()
        .rename(columns={'freq': 'totals'})
    )
    fig_df = fig_df.merge(sums_df, on=[rows_col])
    # fig_df['proportion'] = fig_df['freq'] / fig_df['totals']
    fig_df[rows_col] = fig_df[rows_col] + ' (n=' + fig_df['totals'].astype('str') + ')'

    # add column totals to the column labels
    col_sums_df = df.copy().sort_values([rows_col, cols_col])
    col_sums_df['freq'] = 1
    col_sums_df = (
        col_sums_df.groupby([cols_col])
        .agg({'freq': 'sum'})
        .reset_index()
        .rename(columns={'freq': 'col_totals'})
    )
    fig_df = fig_df.merge(col_sums_df)
    fig_df[cols_col] = fig_df[cols_col] + ' (n=' + fig_df['col_totals'].astype('str') + ')'

    fig_df = fig_df.pivot(index=rows_col, columns=cols_col, values='freq')

    fig = sns.heatmap(data=fig_df, square=True, annot=True, robust=True, **kwargs)
    return fig


def compute_heatmap_linkage_with_nulls(df):
    # https://stackoverflow.com/questions/62044918/how-to-fix-seaborn-clustermap-condensed-distance-matrix-must-contain-only-finit
    # replicates the default way that seaborn does it but allows nan
    import scipy.cluster.hierarchy as hc
    import scipy.spatial as sp

    try:
        row_linkage = hc.linkage(df.T.values, method='average', metric='euclidean')
        col_linkage = hc.linkage(df.values, method='average', metric='euclidean')
    except ValueError as err:
        import warnings

        warnings.warn(
            'cannot compute distance linkage across raw values since there are nan in matrix. Will fallback to clustering by correlation'
        )
        row_dism = 1 - df.T.corr()
        row_linkage = hc.linkage(row_dism, method='average', metric='euclidean')
        col_dism = 1 - df.corr()
        col_linkage = hc.linkage(col_dism, method='average', metric='euclidean')
    return dict(row_linkage=row_linkage, col_linkage=col_linkage)


def plot_heatmap(df, rows_col, cols_col, data_col, row_labels=[], agg_func='mean', **kwargs):
    # metric = euclidean
    # method = average
    labels_df = df.copy().sort_values([rows_col])[[rows_col] + row_labels].drop_duplicates()
    fig_df = df.copy()
    if agg_func:
        fig_df = (
            fig_df.groupby([rows_col, cols_col, *row_labels])
            .agg({data_col: agg_func})
            .reset_index()
        )
    fig_df = fig_df.pivot(index=rows_col, columns=cols_col, values=data_col)
    plt.figure(figsize=(10, 16))
    for label in row_labels:
        color_map = dict(zip(labels_df[label].unique(), sns.color_palette('cubehelix')))
        labels_df[label] = labels_df[label].map(color_map)

    labels_df = labels_df.set_index(rows_col)

    linkage_args = compute_heatmap_linkage_with_nulls(fig_df)

    fig = sns.clustermap(
        data=fig_df,
        square=True,
        annot=True,
        robust=True,
        row_colors=labels_df[row_labels] if row_labels else None,
        **linkage_args,
        **kwargs,
    )
    return fig


def stacked_bar_chart(df, cat_var, num_var, stack_var, cat_axis='x'):
    # set the figure size
    fig = plt.figure(figsize=(14, 14))

    legend_patches = []
    all_groups = set(df[stack_var].unique().tolist())
    used_groups = set()

    default_df = df[[cat_var]].drop_duplicates()

    for i, group in enumerate(all_groups):
        group_df = default_df.merge(
            (
                df[df[stack_var].isin(all_groups - used_groups)]
                .groupby(cat_var)
                .agg({num_var: 'sum'})
                .reset_index()
            ),
            on=cat_var,
            how='left',
        )
        group_df[num_var] = group_df[num_var].fillna(0)
        group_df = group_df.sort_values([cat_var])

        sns.barplot(
            x=cat_var if cat_axis == 'x' else num_var,
            y=num_var if cat_axis == 'x' else cat_var,
            data=group_df,
            color=BC_CANCER_PALETTE[i % len(BC_CANCER_PALETTE)],
        )
        legend_patches.append(
            mpatches.Patch(color=BC_CANCER_PALETTE[i % len(BC_CANCER_PALETTE)], label=group)
        )
        used_groups.add(group)
    plt.legend(handles=legend_patches)

    return fig


def apply_to_ax(fig, func) -> None:
    if isinstance(fig, Axes):
        func(fig)
    else:
        for i in range(len(fig)):
            apply_to_ax(fig[i], func)
