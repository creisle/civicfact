import itertools
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import hex2color

from .util import (
    NATURE_FONTSIZE_MD,
    NATURE_FONTSIZE_SM,
    add_barplot_labels,
    create_legend_handles,
    set_fontsize,
)


def fancy_upset_plot(
    member_sets: dict[str, set],
    total_items: Optional[int] = None,
    min_intersection_size: int = 1,
    text_offset_points=3,
    matrix_circle_size=100,
    lower_axis_label='',
    upper_axis_label='Intersection Size',
    matrix_connecting_linewidth=2,
    legend_kwargs: dict = {},
    percent: bool = False,
    format_label=str,
    figure=None,
    subplot_spec=None,
    hide_legend: bool = False,
    plot_union_line: bool = False,
    hue_map: Optional[dict[str, list[list[float]]]] = None,
    data_label_fontsize=NATURE_FONTSIZE_SM,
    fontsize=NATURE_FONTSIZE_MD,
    upper_hue=hex2color('#2b2b2b'),
    gridspec_kwargs={},
    linewidth=0.5,
):
    """
    gs00 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[0])

    Args:
        member_sets: the sets of items that we want to compare intersections for
        matrix_circle_size: the size of the dots in the presence/absence matrix
    """
    member_names = [s for s in member_sets]
    total_union = len(set.union(*member_sets.values()))
    total_items = total_items if total_items else total_union

    fig = plt.figure() if figure is None else figure
    if subplot_spec:
        spec = gridspec.GridSpecFromSubplotSpec(
            hspace=0.0,
            wspace=0.0,
            width_ratios=[1, 3],
            **gridspec_kwargs,
            ncols=2,
            nrows=2,
            subplot_spec=subplot_spec,
        )
    else:
        spec = gridspec.GridSpec(
            hspace=0.0,
            wspace=0.0,
            width_ratios=[1, 3],
            **gridspec_kwargs,
            ncols=2,
            nrows=2,
            figure=fig,
        )
    matrix_ax = fig.add_subplot(spec[1, 1])
    upper_bar_ax = fig.add_subplot(spec[0, 1], sharex=matrix_ax)
    left_bar_ax = fig.add_subplot(spec[1, 0], sharey=matrix_ax)

    # count the intersections
    set_members = []

    if hue_map is None:
        palette = sns.cubehelix_palette(
            n_colors=len(member_names),
            start=0.5,
            rot=-1.5,
            hue=1,
            dark=0.1,
            light=0.8,
            reverse=True,
        )
        hue_map = {m: p for m, p in zip(member_names, palette)}

    member_order = {
        m: i
        for i, m in enumerate(
            sorted(
                member_names,
                key=lambda name: (
                    len(member_sets[name]),
                    len(member_names) - member_names.index(name),
                ),
                reverse=True,
            )
        )
    }
    # calculate the initial intersections
    set_order: dict[tuple[str, ...], int] = {}

    initial_intersects: dict[tuple[str, ...], set[str]] = {}
    for size in range(1, len(member_sets) + 1):
        for combo in itertools.combinations(member_names, size):
            intersect = set.intersection(*[member_sets[m] for m in combo])
            # remove any elements present in the excluded members
            for other_member in member_names:
                if other_member not in combo:
                    intersect = intersect - member_sets[other_member]

            if len(intersect) >= min_intersection_size:
                initial_intersects[combo] = intersect

    set_order = {
        m: i
        for i, m in enumerate(
            sorted(initial_intersects, key=lambda s: len(initial_intersects[s]), reverse=True)
        )
    }

    for combo, intersect in initial_intersects.items():
        for member_label in combo:
            set_members.append(
                (
                    set_order[combo],
                    member_label,
                    member_order[member_label],
                    len(intersect),
                    len(combo),
                )
            )

    set_members_df = pd.DataFrame(
        set_members,
        columns=['set_pos', 'member_label', 'member_pos', 'intersection_count', 'member_count'],
    )
    set_members_df['intersection_percent'] = (
        set_members_df['intersection_count'] / total_items
    ) * 100
    set_members_df = set_members_df.sort_values(
        ['intersection_count', 'member_pos'], ascending=False
    ).reset_index()
    member_sizes = pd.DataFrame(
        [(m, len(member_sets[m]), member_order[m]) for m in member_names],
        columns=['member_label', 'intersection_count', 'member_pos'],
    )
    member_sizes['intersection_percent'] = (member_sizes['intersection_count'] / total_items) * 100
    member_sizes = member_sizes.sort_values(
        ['intersection_count', 'member_pos'], ascending=False
    ).reset_index()

    sns.scatterplot(
        data=set_members_df,
        s=matrix_circle_size,
        x='set_pos',
        y='member_pos',
        hue='member_label',
        palette=hue_map,
        ax=matrix_ax,
        edgecolor='#000000',
        linewidth=linewidth,
    )

    # add the vertical lines to the scatter plot
    for set_pos in set_order.values():
        max_member = set_members_df[set_members_df.set_pos == set_pos].member_pos.max()
        min_member = set_members_df[set_members_df.set_pos == set_pos].member_pos.min()

        if max_member > min_member:
            matrix_ax.vlines(
                x=[set_pos],
                ymin=min_member,
                ymax=max_member,
                color='#000000',
                zorder=-1,
                linewidth=matrix_connecting_linewidth,
            )

    left_bar_ax.invert_xaxis()
    sns.barplot(
        data=member_sizes,
        x='intersection_count' if not percent else 'intersection_percent',
        y='member_pos',
        ax=left_bar_ax,
        palette=hue_map,
        hue='member_label',
        orient='h',
        dodge=False,
        edgecolor='#000000',
        linewidth=linewidth,
    )

    if plot_union_line:
        trans = mtransforms.blended_transform_factory(left_bar_ax.transData, left_bar_ax.transAxes)
        left_bar_ax.fill_between(
            (0, total_union), 0, 1, facecolor=upper_hue, alpha=0.25, transform=trans
        )
        left_bar_ax.axvline(total_union, color='black', linestyle='dashed', linewidth=0.5)
        left_bar_ax.annotate(
            f'Union\n{format_label(total_union)}',
            (total_union, left_bar_ax.get_ylim()[1]),
            va='baseline',
            ha='center',
            fontsize=data_label_fontsize,
            textcoords='offset points',
            xytext=(0, text_offset_points),
        )
        # add just the total items line as the xtick (keep the zero to be removed later down)
        left_bar_ax.set_xticks([0, total_items])
        left_bar_ax.set_xticklabels(['0', f'Total\n{format_label(total_items)}'])

    upper_fig_df = set_members_df[
        ['intersection_count', 'intersection_percent', 'set_pos', 'member_count']
    ].drop_duplicates()

    sns.barplot(
        data=upper_fig_df,
        y='intersection_count' if not percent else 'intersection_percent',
        x='set_pos',
        ax=upper_bar_ax,
        color=upper_hue,
        alpha=0.9,
        linewidth=linewidth,
    )

    # remove the built-in legends (so we can add a figure level one later)
    for ax in [upper_bar_ax, matrix_ax, left_bar_ax]:
        if ax.get_legend():
            ax.get_legend().remove()

    set_fontsize(left_bar_ax, fontsize, data_label_fontsize)
    set_fontsize(upper_bar_ax, fontsize, data_label_fontsize)

    # remove all the extra crap we don't need
    plt.setp(matrix_ax.get_xticklabels(), visible=False)
    plt.setp(matrix_ax.get_yticklabels(), visible=False)
    plt.setp(upper_bar_ax.get_xticklabels(), visible=False)
    plt.setp(left_bar_ax.get_yticklabels(), visible=False)

    # first tick on upper plot gets cut off, remove it
    # upper plot behaves weird when you just hide the first tick, hides a couple top ones too
    plt.setp(upper_bar_ax.get_yticklabels()[:1], visible=False)
    plt.setp(left_bar_ax.get_xticklabels()[:1], visible=False)

    matrix_ax.set_ylabel(None)
    matrix_ax.set_xlabel(None)
    left_bar_ax.set_ylabel(None)
    upper_bar_ax.set_xlabel(None)
    sns.despine(ax=upper_bar_ax, right=True, top=True, left=False, bottom=False)
    sns.despine(ax=matrix_ax, right=True, top=True, left=True, bottom=True)
    sns.despine(ax=left_bar_ax, right=True, top=True, left=True, bottom=False)
    matrix_ax.grid(True, which='major', axis='both')
    upper_bar_ax.grid(True, which='major', axis='y')

    add_barplot_labels(
        left_bar_ax,
        member_sizes,
        label_column='intersection_count' if not percent else 'intersection_percent',
        plot_baseline='east',
        fontsize=data_label_fontsize,
        format_label=format_label,
        text_offset_points=(-1 * text_offset_points, 0),
    )
    add_barplot_labels(
        upper_bar_ax,
        upper_fig_df,
        fontsize=data_label_fontsize,
        label_column='intersection_count' if not percent else 'intersection_percent',
        plot_baseline='south',
        format_label=format_label,
        text_offset_points=(0, text_offset_points),
    )

    upper_bar_ax.set_ylabel(upper_axis_label)

    left_bar_ax.set_xlabel(lower_axis_label)

    # add a legend below the whole plot
    if not hide_legend:
        legend_rows = len(member_names) / legend_kwargs.get('ncol', 2)
        default_legend_kwargs = {
            'loc': 'lower left',
            'bbox_to_anchor': (0.25, -0.01 * (legend_rows - 1)),
            'frameon': False,
            'ncol': 2,
            'fontsize': fontsize,
        }
        default_legend_kwargs.update(legend_kwargs)
        labels = member_sizes.member_label.tolist()

        fig.legend(
            create_legend_handles(hue_map, labels, edgecolor='#000000', linewidth=linewidth),
            labels,
            **default_legend_kwargs,
        )
    return fig
