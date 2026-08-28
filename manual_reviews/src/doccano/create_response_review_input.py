import argparse
import logging
import uuid

import pandas as pd
from article_utils.util import dump_jsonl, read_jsonl
from nli.util import (
    SELECTION_DELIMITER,
    Text,
    convert_col_to_words,
    convert_enum_to_words,
)

LABEL_COL = "gold_label_name"


def build_alias_table(claim) -> str:
    aliases = []

    for component in claim:
        if "meta" not in component:
            logging.debug(f"claim component has no metadata: {component}")
            continue
        if "synonyms" not in component["meta"] or not component["meta"]["synonyms"]:
            logging.debug(f"claim metadata has no synonyms: {component}")
            continue
        for synonym in component["meta"]["synonyms"]:
            if synonym["name"] and synonym["synonyms"]:
                aliases.append(
                    {
                        "term": synonym["name"],
                        "term_type": synonym["type"],
                        "term_aliases": set(synonym["synonyms"]) - {synonym["name"]},
                    }
                )
    if not aliases:
        return ""

    aliases_df = pd.DataFrame.from_records(aliases)
    aliases_df["term_type"] = aliases_df.term_type.apply(
        lambda tt: tt if tt != "molecularProfile" else "variant"
    )
    aliases_df = aliases_df.drop_duplicates(["term", "term_type"]).fillna("")
    aliases_df = aliases_df[aliases_df.term != ""]
    aliases_df = aliases_df[aliases_df.term_aliases.apply(lambda aliases: bool(aliases))]

    rows = []
    for _, row in aliases_df.iterrows():
        rows.append(
            f'<li><em>{row.term} ({row.term_type})</em>: {"; ".join([a for a in row.term_aliases])}</li>'
        )
    if len(rows) < 2:
        logging.debug(f"no non-name aliases found: {aliases}")
        return ""
    return "<p>Alias Terms</p><ul>" + " ".join(rows) + "</ul>"


def build_acronym_html(acronyms):
    if len(acronyms) == 0:
        return ""
    html = ['<ul class="acronyms">']
    df = pd.DataFrame.from_records(acronyms).drop_duplicates()
    for _, row in df.iterrows():
        html.append(f"<li>{row['name']}: {row['defn']}</li>")
    html.append("</ul>")
    return "<br/><h3>Acronyms</h3>" + "".join(html)


def build_html_component(selections: list[Text], selection_delimiter=SELECTION_DELIMITER) -> str:
    paragraphs = []
    for selection in selections:
        if selection["type"] == "table":
            # new paragraph
            table = []
            for col, val in selection["content"]:
                table.append(
                    f"<tr><th>{convert_col_to_words(col)}</th><td>{convert_enum_to_words(val)}</td></tr>"
                )

            paragraphs.append("<table>" + " ".join(table) + "</table>")
        elif paragraphs:
            paragraphs[-1] += selection_delimiter + selection["content"]
        else:
            paragraphs.append(selection["content"])
    return " ".join([f'<p style="white-space: pre-wrap">{p}</p>' for p in paragraphs])


def unflatten_claim(flat_claim):
    header, row = flat_claim.split('\n')
    content = []
    for col, val in zip(header.split('\t'), row.split('\t')):
        content.append([col, val])
    return [dict(content=content, type='table')]


def build_html(df):
    df = df.copy()

    if 'evidence' not in df.columns:
        df['evidence'] = df['evidence.flat'].apply(
            lambda ev: [Text(e) for e in ev.split(SELECTION_DELIMITER)]
        )
    df["evidence_html"] = df.evidence.apply(build_html_component)
    if 'claim' not in df.columns:
        df['claim'] = df['claim.flat'].apply(unflatten_claim)
    df["claim_html"] = df.claim.apply(build_html_component)
    df["info_html"] = df["claim"].apply(build_alias_table)

    df["info_html"] = df.info_html.fillna("")

    for field in ["claim_html", "evidence_html", "info_html"]:
        if df[df[field].fillna("") == ""].shape[0] != 0:
            logging.warning(
                f'some rows ({df[df[field].fillna("") == ""].shape[0]}) do not have {field}'
            )

    df.loc[df.info_html != "", "info_html"] = (
        '<div class="ad_info"><h2>Additional Information</h2>'
        + df[df.info_html != ""].info_html
        + "</div>"
    )

    df["html"] = (
        '<div class="example">'
        + '<div class="question"><h2>Given a claim and some evidence, determine the stance of the evidence toward the claim.</h2></div>'
        + '<div class="claim"><h2>Claim</h2>'
        + df.claim_html
        + '</div><div class="evidence" style="white-space: pre-wrap;"><h2>Evidence</h2>'
        + df.evidence_html
        + "</div>"
        + '<br><div class="question"><h2>Does the evidence support the claim? Use of the following labels: SUPPORTS, REFUTES, NEI. Give your reasoning</h2></div>'
        + df.info_html
        + "</div>"
    )

    assert df[df.html.fillna("") == ""].shape[0] == 0, "some rows do not have html"

    return df


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--subset_selection', help='filter all respones on this')
parser.add_argument(
    '--inputs',
    nargs='+',
    help='files to choose claim/evidence pairs from',
    required=True,
    action='append',
)
parser.add_argument('--ref_data', help='reference dataset to get review stats from')
parser.add_argument(
    '--response_per_file',
    type=int,
    help='number of response to keep for each prompt from each file',
)
parser.add_argument(
    '--output', required=True, nargs=2, metavar=('detailed_output', 'doccano_input')
)
parser.add_argument('--num_expected_responses', default=None, type=int)
args = parser.parse_args()

dfs = []
for input_files_group in args.inputs:
    df = read_jsonl(*input_files_group, add_filename=True)
    df['filenames'] = ';'.join(sorted(input_files_group))
    df = df.sample(frac=1)
    dfs.append(df)

df = pd.concat(dfs)
df['document.pmid'] = df['document.pmid'].astype(str)
MERGE_COLS = ['claim.flat', 'evidence.flat', 'document.pmid']

if args.subset_selection:
    subset_df = read_jsonl(args.subset_selection)
    df = df.drop(columns=[c for c in df.columns if c in subset_df.columns and c not in MERGE_COLS])
else:
    subset_df = df[MERGE_COLS].drop_duplicates()

subset_df['review_id'] = subset_df.apply(lambda _: uuid.uuid4(), axis=1)
df = df.merge(subset_df)
df = df.drop_duplicates(MERGE_COLS + ['filenames', 'generated_text'])

# only keep the first n responses from each model
df['_prompt_index'] = df.groupby(['filenames'] + MERGE_COLS, dropna=False).cumcount()
if args.response_per_file is not None:
    logging.info(f'filtering to {args.response_per_file} responses per input file group')
    df = df[df._prompt_index < args.response_per_file]
dump_jsonl(df, args.output[0])


total_examples = df.shape[0]
logging.info(f"{total_examples} input examples")

# dump an html version of examples for loading in doccano
html_df = build_html(df)
assert html_df[html_df.html.fillna("") == ""].shape[0] == 0

assert (
    total_examples == html_df.shape[0]
), f"examples in {total_examples} does not match examples out {html_df.shape[0]}"


html_df = html_df.sample(frac=1)  # randomize response order
html_df = (
    html_df.groupby(['claim.flat', 'evidence.flat', 'document.pmid', 'generated_text'])
    .agg({'html': 'first', 'gold_label_name': 'first', 'civic.id': 'unique'})
    .reset_index()
)
html_df['response_index'] = html_df.groupby(['evidence.flat', 'claim.flat', 'document.pmid'])[
    'generated_text'
].cumcount()
html_df = html_df.merge(
    html_df[['evidence.flat', 'claim.flat', 'document.pmid']]
    .value_counts()
    .reset_index(name='response_count')
)


def format_text(row):
    return (
        f'---------- Response ({row["response_index"] + 1} / {row["response_count"]}) for this claim/evidence pair\n\n'
        + row['generated_text']
    )


html_df["text"] = html_df.apply(format_text, axis=1)


logging.info(f"writing {html_df.shape[0]} examples")

html_df = html_df.rename(columns={"html": "contextHtml"})
assert html_df[html_df.contextHtml.fillna("") == ""].shape[0] == 0


dump_jsonl(
    html_df[
        [
            "text",
            'contextHtml',
            "review_id",
            "gold_label_name",
            'civic.id',
            'response_index',
            'response_count',
            'claim.flat',
            'evidence.flat',
            'document.pmid',
        ]
    ].sample(frac=1),
    args.output[1],
)
