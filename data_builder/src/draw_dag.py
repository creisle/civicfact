import argparse
import json
import subprocess
import tempfile

import networkx as nx

grouping_substrings = ["augment_", "pubtator"]
build_files = [
    "balance_dataset",
    "chunk_documents",
    "merge_sentence_pairs",
    "generate_civic_claims",
    "download_civic",
]


def convert_to_dot(graph):
    nodes = []
    graph = nx.convert_node_labels_to_integers(graph)
    groups = {}

    all_node = None
    for node_id, label in graph.nodes(data="label"):
        if label == "all":
            all_node = node_id
        for prefix in grouping_substrings:
            if prefix in label:
                groups.setdefault(prefix, []).append(node_id)

    node_declarations = {}

    for node_id, label in graph.nodes(data="label"):
        if node_id == all_node:
            continue
        job_label = label.split("\n")[0].strip()
        termnial_node = "peripheries=2" if job_label in build_files else ""
        node_declarations[node_id] = (
            f'{node_id}[label = {json.dumps(label)}, color = "0.53 0.6 0.85", style = "rounded" {termnial_node}];'
        )
        nodes.append(
            f'{node_id}[label = {json.dumps(label)}, color = "0.53 0.6 0.85", style = "rounded" {termnial_node}];'
        )  # 	2[label = "ablation_ft_nli\nanalysis: ablation\nft_dataset: civic1.4\nmodel_dir: bert-base-uncased", color = "0.13 0.6 0.85", style="rounded,dashed"];

    if all_node is not None:
        graph.remove_node(all_node)
    edges = []

    for src_node, dst_node, weight in graph.edges(data="weight"):
        edges.append(f"{src_node} -> {dst_node}")

    nodes = "\n    ".join(nodes)
    edges = "\n    ".join(edges)

    clusters = []

    for i, (group, group_nodes) in enumerate(groups.items()):
        group_nodes = "\n    ".join([node_declarations[n] for n in group_nodes])
        clusters.append(
            f'''\
subgraph cluster_{i} {{
    style=filled;
    color=lightgrey;
    {group_nodes}
    label = "{group}";
}}'''
        )
    clusters = "\n    ".join(clusters)
    return f"""digraph snakemake_dag {{
    graph[bgcolor=white, margin=0];
    node[shape=box, style=rounded, fontname=sans, fontsize=10, penwidth=2];
    edge[penwidth=2, color=grey];
    {clusters}
    {nodes}
    {edges}
}}
"""


def squash_job_rules(graph, keep_wildcards=["analysis"]):
    squash_mapping = {}
    squash_attrs = {}

    for node_id, data in graph.nodes(data=True):
        wkey = tuple(
            sorted(
                (k, v)
                for k, v in data.get("wildcards", {}).items()
                if k in keep_wildcards
            )
        )
        node_key = (data.get("rule"), wkey)
        squash_mapping.setdefault(node_key, []).append(node_id)
        squash_attrs.setdefault(node_key, {})

        for k, v in data.get("wildcards", {}).items():
            squash_attrs[node_key].setdefault(k, []).append(v)

    labels = {}
    for squash_node, nodes in squash_mapping.items():
        label = squash_node[0]
        if squash_node[1]:
            label += "\n" + "\n".join(f"{k}: {v}" for k, v in squash_node[1])

        if len(nodes) > 1:
            labels[squash_node] = f"{label}\nn = {len(nodes)}"
        else:
            labels[squash_mapping[squash_node][0]] = label

    squash_mapping = {k: v for k, v in squash_mapping.items() if len(v) > 1}

    for squash_node in squash_mapping:
        graph.add_node(squash_node, **squash_attrs[squash_node])

    nx.set_node_attributes(graph, labels, "label")

    reverse_squash_mapping = {}
    for squash_node in squash_mapping:
        for node in squash_mapping[squash_node]:
            reverse_squash_mapping[node] = squash_node

    for src_node, dst_node in list(graph.edges()):
        if (
            src_node not in reverse_squash_mapping
            and dst_node not in reverse_squash_mapping
        ):
            continue

        new_src = reverse_squash_mapping.get(src_node, src_node)
        new_dst = reverse_squash_mapping.get(dst_node, dst_node)

        graph.add_edge(new_src, new_dst)

    # drop all the squashed nodes
    for node in reverse_squash_mapping:
        graph.remove_node(node)


def simplify_paths(g):
    # only keep longer path
    keep = set()
    all = set()
    for src, tgt in g.edges():
        all.add((src, tgt))
        paths = nx.all_simple_paths(g, source=src, target=tgt)
        paths = sorted(paths, key=lambda x: len(x), reverse=True)
        for new_src, new_tgt in zip(paths[0], paths[0][1:]):
            keep.add((new_src, new_tgt))

    drop = all - keep
    print(f"{len(drop)} possible edges could be dropped")
    for src, tgt in drop:
        g.remove_edge(src, tgt)
    return g


def parse_snakemake_attributes(graph):
    attrs = {}
    all_wildcards = set()
    for node_id, data in graph.nodes(data=True):
        label = data["label"].strip('"').split("\\n")
        wildcards = {}
        for wildcard in label[1:]:
            name, value = wildcard.split(": ")
            wildcards[name] = value
        attrs[node_id] = {"rule": label[0], "wildcards": wildcards, **data}
        all_wildcards.update(wildcards)

    print("possible wildcards:", all_wildcards)
    return attrs


def convert_multigraph_to_digraph(multi_graph):
    graph = nx.DiGraph()
    graph.add_nodes_from(multi_graph.nodes(data=True))

    for u, v, data in multi_graph.edges(data=True):
        w = data.get("weight", 1)
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += w
        else:
            graph.add_edge(u, v, weight=w)
    return graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_prefix")
    snake_group = parser.add_mutually_exclusive_group()
    snake_group.add_argument("--snakemake_dot", help="snakemake dag dot file to redraw")
    snake_group.add_argument(
        "--snakemake_command",
        help="command to run to generate the snakemake dot file to stdout",
        default="snakemake --dag -F",
    )
    parser.add_argument(
        "--wildcards",
        default=[],
        nargs="+",
        help="wildcards to expand, all others will be collapsed",
    )

    args = parser.parse_args()

    if not args.snakemake_dot:
        with tempfile.NamedTemporaryFile() as temp_fp:
            subprocess.run(f"{args.snakemake_command} > {temp_fp.name}", shell=True)
            dot_graph = nx.nx_pydot.read_dot(temp_fp.name)
    else:
        dot_graph = nx.nx_pydot.read_dot(args.snakemake_dot)

    attrs = parse_snakemake_attributes(dot_graph)
    nx.set_node_attributes(dot_graph, attrs)

    squash_job_rules(dot_graph, args.wildcards)
    graph = convert_multigraph_to_digraph(dot_graph)

    labels = {}
    for node, label in graph.nodes(data="label"):
        labels[node] = label or node

    nx.relabel_nodes(graph, labels)
    graph = simplify_paths(graph)

    with open(f"{args.output_prefix}.dot", "w") as fh:
        print(f"writing: {args.output_prefix}.dot")
        dot_data = convert_to_dot(graph)
        fh.write(dot_data)

    print(f"writing: {args.output_prefix}.png")

    subprocess.run(
        f"cat {args.output_prefix}.dot | dot -Tsvg > {args.output_prefix}.svg; convert {args.output_prefix}.svg {args.output_prefix}.png",
        shell=True,
    )


if __name__ == "__main__":
    main()
