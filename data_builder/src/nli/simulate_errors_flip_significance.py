import argparse
import json
import logging
import os
import re
from copy import deepcopy

import pandas as pd

from article_utils.util import dump_jsonl

from .util import Text, concat_text

CLINICAL_SIG_OPPOSITES = {
    'PROGNOSTIC': [('poor outcome', 'better outcome')],
    'PREDICTIVE': [('sensitivity/response', 'resistance')],
    'FUNCTIONAL': [
        ('gain of function', 'unaltered function'),
        ('loss of function', 'unaltered function'),
        ('neomorphic', 'unaltered function'),
        ('dominant negative', 'unaltered function'),
    ],
    'DIAGNOSTIC': [('positive', 'negative')],
    'PREDISPOSING': [('predisposition', 'protectiveness')],
    'ONCOGENIC': [('oncogenicity', 'protectiveness')],
}
for ev_type in CLINICAL_SIG_OPPOSITES:
    CLINICAL_SIG_OPPOSITES[ev_type] = CLINICAL_SIG_OPPOSITES[ev_type] + [
        (t2, t1) for (t1, t2) in CLINICAL_SIG_OPPOSITES[ev_type]
    ]


def get_example_from_id(id):
    try:
        m = re.match(r'[^_]+_([^_]+)_.*', id)
        return m.group(1)
    except Exception as err:
        logging.warning(f'failed to extract group from ID: {id}')
        raise err


def modify_claim(claim: Text, target_col: str, replacement: str) -> Text:
    claim = deepcopy(claim)
    claim['content'][target_col] = replacement
    return claim


def check_claim_fields_consistency(df):
    df = df.copy()

    df['_claim.flat'] = df.claim.apply(concat_text)
    if df[df['_claim.flat'] == df['claim.flat']].shape[0] != df.shape[0]:
        print(df[df['_claim.flat'] != df['claim.flat']][['claim.flat', '_claim.flat', 'reviews']])
        raise AssertionError('some rows claim.flat are not consistent with their flattened claim')

    df['_sig.claim'] = df.claim.str['content'].str['significance']
    df['_sig.flat'] = df['claim.flat'].apply(
        lambda text: Text.parse(text)['content']['significance']
    )
    assert df[df['_sig.claim'] == df['_sig.flat']].shape[0] == df.shape[0]


def create_reasoning(row):
    original_significance = row['_original_sig']
    new_significance = row['_new_sig']
    assert original_significance != new_significance
    if row['claim.source'].count('+') > 2:
        # too hard to manage explanations after this point
        return []
    reason = f'The evidence discusses {original_significance} {row["_type"].lower()} significance but the claim refers to {new_significance}'

    if (
        isinstance(row['reviews'], list)
        and len(row['reviews']) == 1
        and row['gold_label_name'] == 'NEI'
        and row['reviews'][0]['username'] == 'bot'
        and row['reviews'][0]['reason'].startswith('T')
    ):
        return [
            {
                'reason': row['reviews'][0]['reason'] + 'Additionally, t' + reason[1:],
                'username': 'bot',
                'label': 'NEI',
                'script': os.path.basename(__file__),
            }
        ]
    elif row['gold_label_name'] != 'NEI':
        return [
            {
                'reason': reason,
                'username': 'bot',
                'label': 'NEI',
                'script': os.path.basename(__file__),
            }
        ]

    return []


def supplement_contradiction_examples(df):
    """
    create additional contradiction examples by flipping entailment examples
    """
    df = df.copy()
    df = df[
        df['civic.evidenceDirection'] == 'SUPPORTS'
    ]  # flips can sometimes be incorrect when the direction is dns b/c they often dns either

    df['original_example'] = df.id
    df['_significance'] = df.claim.str['content'].str['significance']
    df['_type'] = df.claim.str['content'].str['evidenceType']
    dfs = []

    if len(set(df['_type'].unique()) - set(CLINICAL_SIG_OPPOSITES)) != 0:
        print(df.columns.tolist())
        print(
            df[~df._type.isin(CLINICAL_SIG_OPPOSITES)][
                [
                    'claim.source',
                    'evidence.selection_method',
                    'civic.revision_state',
                    'civic.ground_truth',
                ]
            ]
        )
        raise AssertionError(
            f"unexpected evidence types: {set(df['_type'].unique()) - set(CLINICAL_SIG_OPPOSITES)}"
        )
    for ev_type in CLINICAL_SIG_OPPOSITES:
        for term1, term2 in CLINICAL_SIG_OPPOSITES[ev_type]:
            examples_df = df[(df['_significance'] == term1) & (df['_type'] == ev_type)].copy()
            examples_df['gold_label_name'] = examples_df.gold_label_name.apply(
                lambda l: (
                    'REFUTES' if l == 'SUPPORTS' else ('SUPPORTS' if l == 'REFUTES' else 'NEI')
                )
            )
            examples_df['civic.significance'] = term2

            dfs.append(examples_df)
    result_df = pd.concat(dfs)

    result_df['claim.source'] = result_df['claim.source'] + ' + flipped significance'
    result_df['claim.generated'] = True

    result_df['_original_sig'] = result_df.claim.str['content'].str['significance']
    result_df['claim'] = result_df.apply(
        lambda row: modify_claim(row.claim, 'significance', row['civic.significance']), axis=1
    )
    result_df['_new_sig'] = result_df.claim.str['content'].str['significance']
    result_df['claim.flat'] = result_df.claim.apply(concat_text)
    result_df['generated'] = True
    # replace reviews with empty list since they no longer apply post-flip
    result_df['reviews'] = result_df.apply(
        lambda _: [], axis=1
    )  # TODO: readd explanation generation

    result_df = result_df.drop(columns=[c for c in result_df.columns if c.startswith('_')])
    return result_df.drop(columns=['civic.significance'])


def swap_evidence_direction(df):
    df = df[df['civic.evidenceDirection'].isin(['SUPPORTS', 'DOES_NOT_SUPPORT'])].copy()
    df['claim'] = df.claim.apply(
        lambda claim: modify_claim(
            claim,
            'evidenceDirection',
            (
                'DOES_NOT_SUPPORT'
                if claim['content']['evidenceDirection'] == 'SUPPORTS'
                else 'SUPPORTS'
            ),
        )
    )
    df['claim.flat'] = df.claim.apply(concat_text)
    df['claim.generated'] = True
    df['generated'] = True
    df['claim.source'] = df['claim.source'] + ' + flipped evidenceDirection'
    return df


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        help='Path to the JSONL file with the hypothesis civic annotations pre-matched to their article xml content',
    )
    parser.add_argument('output', help='path to the output JSONL file to write the NLI data to')
    args = parser.parse_args()

    logging.info(f'reading: {args.input}')
    input_df = pd.read_json(args.input, lines=True, dtype=False)
    original_columns = [c for c in input_df.columns]
    check_claim_fields_consistency(input_df)

    logging.info(f'{dict(input_df.gold_label_name.value_counts())}')
    logging.info(f'read {input_df.shape[0]} examples')

    df = input_df.copy()
    # do not reflip any examples that have already been flipped
    df = df[~df['claim.source'].str.contains('flipped')]
    df = pd.concat([supplement_contradiction_examples(df), swap_evidence_direction(df)])
    check_claim_fields_consistency(df)
    logging.info(f'flipped {df.shape[0]} examples')

    check_claim_fields_consistency(input_df)

    # drop any generated entries that duplicate an existing entry
    df['civic.ground_truth'] = False
    df['_id'] = df['claim.flat'] + df['evidence.flat'] + df['document.pmid']
    input_df['_id'] = input_df['claim.flat'] + input_df['evidence.flat'] + input_df['document.pmid']
    input_df['_exists'] = True
    df = df.merge(input_df[['_id', '_exists']].drop_duplicates(), how='left')
    df = df[df._exists.isna()]
    df['id'] = df['id'] + '-fs'
    logging.info(f'created {df.shape[0]} new claims')

    df = pd.concat([input_df, df])
    logging.info(f'{df.shape[0]} combined examples')
    logging.info(f'{dict(df.gold_label_name.value_counts())}')
    assert df[df.claim.isna()].shape[0] == 0

    dump_jsonl(df[original_columns], args.output)


if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()
