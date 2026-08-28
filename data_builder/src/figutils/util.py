import logging

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import patches
from matplotlib import pyplot as plt

NATURE_FONTSIZE_SM = 5
NATURE_FONTSIZE_MD = 6
NATURE_FONTSIZE_LG = 7

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.size'] = NATURE_FONTSIZE_MD

MM_TO_INCH = 1 / (10 * 2.54)  # millimetres in inches

CUSTOM_BREWER_DARK2 = [
    "#04C1C8",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#D95F02",
    "#E6AB02",
    "#A6761D",
    "#666666",
]


def set_fontsize(ax, axis_fontsize, label_fontsize):
    ax.set_ylabel(ax.get_ylabel(), fontsize=axis_fontsize)
    ax.set_xlabel(ax.get_ylabel(), fontsize=axis_fontsize)
    ax.tick_params(axis='both', labelsize=label_fontsize)


def scale_figure_to_width(fig, width_in_mm, legend_patch=None):
    curr_width, curr_height = fig.get_size_inches()

    new_width = width_in_mm * MM_TO_INCH
    new_height = new_width * (curr_height / curr_width)
    fig.set_size_inches(new_width, new_height)


def savefig(ax, plot_name, bbox_inches='tight', **kwargs):
    logging.info(f'writing: {plot_name}')
    default_kwargs = {}
    if bbox_inches is not None:
        default_kwargs['bbox_inches'] = bbox_inches

    try:
        ax.figure.savefig(plot_name, **default_kwargs, **kwargs)
    except AttributeError:
        ax.savefig(plot_name, **default_kwargs, **kwargs)
    plt.close()


def create_legend_handles(hue_map, order, hatch_map={}, **kwargs):
    chunks = []
    for label in order:
        chunks.append(
            patches.Patch(
                facecolor=hue_map[label], hatch=hatch_map.get(label), label=label, **kwargs
            )
        )
    return chunks


def generate_matching_palette(
    df: pd.DataFrame, hue_map: dict[str, str], hue_column: str
) -> list[str]:
    # get the order of these columns in the current dataframe
    palette = []
    for label in df[hue_column].unique():
        palette.append(hue_map[label])
    return palette


def generate_hue_map(values, palette_name, offset=0, **kwargs):
    palette = sns.color_palette(palette_name, n_colors=len(values) + offset, **kwargs)
    return {m: p for m, p in zip(values, palette[offset:])}


def add_barplot_labels(
    ax,
    df,
    label_column,
    data_column=None,
    plot_baseline: str = 'west',
    color='black',
    fontsize='small',
    text_offset_points=(0, 2),
    format_label=str,
    skip_na=False,
):
    """
    Args:
        plot_baseline: the position of the bottom of the bars, the direction they are being drawn from
    """
    if data_column is None:
        data_column = label_column

    def skip_patch(patch):
        if not skip_na:
            return False
        if plot_baseline in ['east', 'west']:
            if pd.isnull(patch._width):
                return True
        elif pd.isnull(patch._height):
            return True
        return False

    rows = [r for _, r in df.iterrows()]

    for patch, row in zip([p for p in ax.patches if not skip_patch(p)], rows):
        # patches should default iterate in the order the dataframe was passed to the plot function
        if plot_baseline in ['east', 'west']:
            if skip_na and pd.isnull(patch._width):
                continue
            assert row[data_column] == patch._width or pd.isnull(patch._width)
            xpos = row[data_column]
            try:
                ypos = (patch._y1 + patch._y0) / 2
            except AttributeError:
                ypos = patch._y0 + patch._height / 2
            va = 'center'

            if plot_baseline == 'east':
                ha = 'right'
            else:
                ha = 'left'
        else:
            if skip_na and pd.isnull(patch._height):
                continue
            if not (row[data_column] == patch._height or pd.isnull(patch._height)):
                raise AssertionError(
                    f'data point value ({row[data_column]}) does not match bar height ({patch._height})'
                )

            try:
                xpos = (patch._x1 + patch._x0) / 2
            except AttributeError:
                xpos = patch._x0 + patch._width / 2
            ypos = row[data_column]
            ha = 'center'

            if plot_baseline == 'south':
                va = 'bottom'
            else:
                va = 'top'

        ax.annotate(
            format_label(row[label_column]),
            (xpos, ypos),
            color=color,
            va=va,
            ha=ha,
            fontsize=fontsize,
            xytext=text_offset_points,
            textcoords='offset points',
        )


def add_boxplot_median_labels(
    medians_df,
    ax,
    median_column,
    xytext=(0, 2),
    color='black',
    fontsize='x-small',
    hide_zeroes=True,
):
    LINES_PER_BOX = 6  # 6 pieces
    MEDIAN_LINE = 4  # median line is the 4th line involved. You can check this yourself by printing the xy coordinates of all the line and comparing them

    _, ymax = ax.get_ylim()

    for i, row in medians_df.iterrows():
        median_line = ax.lines[i * LINES_PER_BOX + MEDIAN_LINE]
        yvalue = median_line.get_ydata()[0]  # y values are same, just pick one
        xvalue = sum(median_line.get_xdata()) / 2
        if row[median_column] != yvalue:
            raise AssertionError(
                f'failed sanity check median={row[median_column]}, yvalue={yvalue}'
            )  # sanity check we didn't mess up the ordering somewhere
        if hide_zeroes and row[median_column] == 0:
            continue
        ax.annotate(
            str(round(row[median_column], 2)),
            (xvalue, row[median_column]),
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=fontsize,
            color=color,
            xytext=xytext,  # put it above the median line a bit
            textcoords='offset points',
        )


def _convert_df_to_sankey(df, levels, add_node_counts=True):

    node_mapping = {}

    for level in levels:
        for value in sorted(df[level].dropna().unique().tolist()):
            node_mapping[(level, value)] = len(node_mapping)

    dfs = []
    for level1, level2 in zip(levels, levels[1:]):
        tmp_df = df[[level1, level2]].value_counts().reset_index(name='value')
        tmp_df[level1] = tmp_df[level1].apply(lambda value: node_mapping[(level1, value)])
        tmp_df[level2] = tmp_df[level2].apply(lambda value: node_mapping[(level2, value)])
        tmp_df = tmp_df.rename(columns={level1: 'source', level2: 'target'})
        dfs.append(tmp_df)

    links_df = pd.concat(dfs)

    node_labels = []
    node_groups = []
    for level, value in sorted(node_mapping, key=lambda t: node_mapping[t]):
        count = df[df[level] == value].shape[0]
        label = f'{value}'
        if add_node_counts:
            label += f' (n={count})'
        node_labels.append(label)
        node_groups.append(level)

    return dict(label=node_labels, groups=node_groups), links_df


def plot_sankey_diagram(df, levels, **kwargs):
    import plotly.graph_objects as go

    nodes, links = _convert_df_to_sankey(df, levels)
    fig = go.Figure(
        data=[
            go.Sankey(
                valueformat='%d',
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    # color = ["blue", "red"],
                    **nodes,
                ),
                link=links,
                **kwargs,
            )
        ]
    )

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    return fig
