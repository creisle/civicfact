import logging
import os

import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def ci_95(series):
    return (series.quantile(0.025), series.quantile(0.975))


def apply_ax_func(fig, func_name, *args, **kwargs):
    for i in range(len(fig.axes)):
        try:
            for j in range(len(fig.axes[i])):
                getattr(fig.axes[i][j], func_name)(*args, **kwargs)
        except TypeError:
            getattr(fig.axes[i], func_name)(*args, **kwargs)


def apply_to_ax(fig, func) -> None:
    if isinstance(fig, Axes):
        func(fig)
    else:
        for i in range(len(fig)):
            apply_to_ax(fig[i], func)


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


def savefig(ax, plot_name, bbox_inches="tight", **kwargs):
    logging.info(f"writing: {plot_name}")
    default_kwargs = {}
    if bbox_inches is not None:
        default_kwargs["bbox_inches"] = bbox_inches

    if not os.path.exists(os.path.dirname(plot_name)) and os.path.dirname(plot_name):
        os.makedirs(os.path.dirname(plot_name))

    try:
        ax.figure.savefig(plot_name, **default_kwargs, **kwargs)
    except AttributeError:
        ax.savefig(plot_name, **default_kwargs, **kwargs)
    plt.close()


def generate_matching_palette(df: pd.DataFrame, hue_map: dict[str, str], hue_column: str) -> list[str]:
    # get the order of these columns in the current dataframe
    palette = []
    for label in df[hue_column].unique():
        palette.append(hue_map[label])
    return palette


def generate_hue_map(values, palette_name, offset: int = 0, **kwargs):
    palette = sns.color_palette(palette_name, n_colors=len(values) + offset, **kwargs)[offset:]
    return {m: p for m, p in zip(values, palette)}


def generate_cont_hue_map(values, palette_name, **kwargs):
    palette = sns.color_palette(palette_name, as_cmap=True, **kwargs)
    max_value = max(values)
    min_value = min(values)
    return {v: palette((v - min_value) / (max_value - min_value)) for v in sorted(values)}


def plot_confusion_matrix(df, rows_col, cols_col, normalize_counts=False, add_n_labels=False, **kwargs):
    fig_df = df.copy().sort_values([rows_col, cols_col])
    fig_df["freq"] = 1
    total_rows = fig_df.shape[0]
    fig_df = fig_df.groupby([cols_col, rows_col]).agg({"freq": "sum"}).reset_index()
    sums_df = fig_df.groupby([rows_col]).agg({"freq": "sum"}).reset_index().rename(columns={"freq": "totals"})
    fig_df = fig_df.merge(sums_df, on=[rows_col])
    # fig_df['proportion'] = fig_df['freq'] / fig_df['totals']
    if add_n_labels:
        fig_df[rows_col] = fig_df[rows_col] + " (n=" + fig_df["totals"].astype("str") + ")"

    # add column totals to the column labels
    col_sums_df = df.copy().sort_values([rows_col, cols_col])
    col_sums_df["freq"] = 1
    col_sums_df = (
        col_sums_df.groupby([cols_col]).agg({"freq": "sum"}).reset_index().rename(columns={"freq": "col_totals"})
    )
    fig_df = fig_df.merge(col_sums_df)
    if add_n_labels:
        fig_df[cols_col] = fig_df[cols_col] + " (n=" + fig_df["col_totals"].astype("str") + ")"

    fig_df = fig_df.pivot(index=rows_col, columns=cols_col, values="freq")

    if normalize_counts:
        fig_df = fig_df.fillna(0).map(lambda freq: round(100 * freq / total_rows))

    fig = sns.heatmap(data=fig_df, square=True, annot=True, robust=True, **kwargs)
    return fig


def add_huemap_legend(hue_map, fig=None, **kwargs):
    handles = []
    for key, color in hue_map.items():
        if color is None:
            continue
        handles.append(mpatches.Patch(color=color, label=key))
    if fig:
        fig.legend(handles=handles, **kwargs)
    else:
        plt.figlegend(handles=handles, **kwargs)
