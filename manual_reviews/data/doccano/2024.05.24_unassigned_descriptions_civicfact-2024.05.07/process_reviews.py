#!/usr/bin/env python
# coding: utf-8

# Process the review results into a single review.jsonl file so that it can be merged with other results.
#
# First match it back up to its source file. To do this we need to do some minor fixes. There was a small number of examples in v2024.04.11 where the claim.flat was updated but the claim object wasn't updated to match so their significance fields don't match. Since claim.flat was used for all testing, that is what we put into the review html file and that's what we should be propogating downstream


import glob
import json
from statistics import mode
import logging
import pandas as pd

from article_utils.db import connect, get_or_create_textid
from article_utils.util import dump_jsonl
from nli.util import convert_col_to_words


def read_dataset(filename):
    # fix bad claim objects
    # claim.flat is correct but the claim object wasn't updated properly afte significance was flipped, do that now
    print(f"reading main datafile: {filename}")
    df = pd.read_json(filename, lines=True, dtype=False)
    df["document.pmid"] = (
        df["document.pmid"].apply(lambda x: "" if pd.isnull(x) else str(int(x))).astype(str)
    )
    df["_flat_sig"] = df["claim.flat"].str.extract(r"significance is ([^;]+)")
    df["_sig"] = df.claim.apply(
        lambda claims: [v for k, v in claims[0]["content"] if k == "significance"][0]
    )

    def replace_with_flat(row):
        unflat = [item.split(" is ") for item in row["claim.flat"].split(";")]
        assert len(unflat) == len(
            row.claim[0]["content"]
        ), f'the length of the flat claim ({len(unflat)}) should match the length of the built claim ({len(row.claim[0]["content"])})'

        for i in range(len(row.claim[0]["content"])):
            assert unflat[i][0].strip() == convert_col_to_words(
                row.claim[0]["content"][i][0]
            ), f'columns don\'t match from flattened ({[unflat[i][0].strip()]}) claim and built ({convert_col_to_words(row.claim[0]["content"][i][0])}) claim'
            row.claim[0]["content"][i][1] = unflat[i][1].strip()

    df[df._flat_sig != df._sig].apply(replace_with_flat, axis=1)
    total_examples = df.shape[0]
    print(f"read {total_examples} data examples")
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    return df


def remove_model_reviews(labels):
    """
    remove any labels that were input just for visually reviewing the model outputs
    """
    new_labels = [l for l in labels if not l.startswith('[')]
    return new_labels or ['']


df = read_dataset('../../../builds/civicfact-2024.05.07/data.jsonl.gz')
df.head()


# Now read in the review files. We want to create review objects which look something like this (example from previous review compilation)
#
# ```json
# {
#     "comment": "evidence is not predictive based on this variant",
#     "group_review_comments": [],
#     "group_review_label": "REFUTES",
#     "group_review_round": "IAv2_error_review",
#     "group_review_tags": [
#         "Bot Generated",
#         "Gold Label Incorrect"
#     ],
#     "group_review_validated": false,
#     "label": "NEI",
#     "original_label": "NEI",
#     "reason": "",
#     "round": "IAv2_civicfact2023.10.20",
#     "username": "camgrisdale"
# }
# ```

logging.basicConfig(level=logging.INFO)

reviews = []

for filename in glob.glob('*.jsonl'):
    if filename.startswith('reviews'):  # don't read output as input on rerun
        continue
    logging.info(f'reading: {filename}')
    reviews.append(pd.read_json(filename, dtype=False, lines=True))
review_df = pd.concat(reviews)
print(review_df.columns.tolist())
review_df['username'] = 'creisle'  # I was the only reviewer of the model errors file
review_df['round'] = 'model error review 2024.05.24 (unassigned partition)'
review_df['label'] = review_df.label.apply(remove_model_reviews)
review_df = review_df.rename(columns={'label': 'reason'})
review_df = review_df.explode('reason')
review_df['reason'] = review_df.reason.fillna('')

review_df['label'] = review_df.reason.str.extract(r'^(\w+)\.?')
review_df['reason'] = review_df.reason.str.replace(r'^(\w+)(\.\s+)?', '', regex=True).fillna('')
review_df['original_label'] = review_df.gold_label_name

review_df = review_df[review_df.reason != '']


# Now join back to the original dataset, drop previous reviews as they will be added from other review.jsonl files
review_df = (
    review_df.groupby(['claim.hash', 'evidence.hash', 'document.pmid'])
    .apply(
        lambda subset_df: subset_df[
            ['label', 'original_label', 'username', 'round', 'reason']
        ].to_dict('records')
    )
    .reset_index(name='reviews')
)
print(review_df.shape)
review_df = review_df.merge(
    df.drop(columns=['reviews']), on=['claim.hash', 'evidence.hash', 'document.pmid']
)
print(review_df.shape)

# finally fix the hashes to match the fixed claim objects
review_df['_claim'] = review_df['claim'].apply(
    lambda claims: json.dumps([c['content'] for c in claims])
)

conn = connect('../../articles_db.sqlite')
try:
    review_df['claim.hash'] = review_df._claim.apply(
        lambda claim_text: get_or_create_textid(conn, claim_text)
    )
finally:
    conn.close()


# now we need to update the gold_label_name for any were the majority review has changed the label
review_df['_gold_label_name'] = review_df.reviews.apply(
    lambda reviews: mode([r['label'] for r in reviews])
)


review_df['gold_label_name'] = review_df._gold_label_name
review_df = review_df.drop(columns=[c for c in review_df.columns if c.startswith('_')])
review_df.head()


dump_jsonl(review_df, 'reviews.jsonl')
