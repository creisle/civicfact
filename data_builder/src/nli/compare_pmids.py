"""
Determine which documents we can safely swap claims between without accidentally creating a real claim
by making a list of documents that do not share claim elements or pubtator annotations
"""

import argparse
import itertools
import json
import logging

import pandas as pd
import tqdm

from article_utils.util import NpEncoder, read_jsonl

NON_SPECIFIC_CANCER_TERMS = {
    'benign neoplasm',
    'benign neoplasms',
    'ca',
    'cancer',
    'cancers',
    'doid:162',
    'malignancies',
    'malignancy',
    'malignant growth',
    'malignant neoplasm',
    'malignant neoplasms',
    'malignant neoplastic disease',
    'malignant tumor',
    'neoplasia',
    'neoplasias',
    'neoplasm, benign',
    'neoplasm, malignant',
    'neoplasm',
    'neoplasms, benign',
    'neoplasms, malignant',
    'neoplasms',
    'primary cancer',
    'tumor, malignant, nos',
    'tumor',
    'tumors',
    'unclassified tumor, malignant',
}


def get_claim_components(claim):
    components = set()

    skip_disease = claim['content'].get('disease', '').lower() in NON_SPECIFIC_CANCER_TERMS

    for k, v in claim['content'].items():
        if k == 'disease' and skip_disease:
            continue
        if k not in {'evidenceType', 'significance', 'evidenceDirection', 'variantOrigin'}:
            if isinstance(v, list):
                components.update(v)
            else:
                components.add(v)

    for syn in claim['meta'].get('synonyms', []):
        if syn['name'] == None:
            continue
        if syn['type'] == 'disease' and skip_disease:
            continue
        components.update(syn['synonyms'])
    return components


def get_disimilar_pmids(df, chunks_df) -> dict[str, set[str]]:
    components_by_pmid = {}
    df = df.drop_duplicates(['document.pmid', 'claim.flat'])
    df = df[df.gold_label_name != 'NEI']

    df['claim_components'] = df['claim'].apply(get_claim_components)
    df.apply(
        lambda row: components_by_pmid.setdefault(row['document.pmid'], set()).update(
            row['claim_components']
        ),
        axis=1,
    )
    for _, row in chunks_df.merge(df[['document.pmid']].drop_duplicates()).iterrows():
        components_by_pmid.setdefault(row['document.pmid'], set())
        if not isinstance(row['pubtator'], list):
            continue
        for match in row['pubtator']:
            for _, elements in match['pubtator.elements'].items():
                components_by_pmid[row['document.pmid']].update(set(elements))

    # now compute a paper similarity score
    for pmid in list(components_by_pmid):
        components_by_pmid[pmid] = {
            str(e).lower().strip() for e in components_by_pmid[pmid] if e and str(e).strip()
        }
        if not components_by_pmid[pmid]:
            # if there are no components at all, likely we are missing things
            del components_by_pmid[pmid]
        components_by_pmid[pmid] = components_by_pmid[pmid] - NON_SPECIFIC_CANCER_TERMS

    scores = []
    combos = list(itertools.combinations(components_by_pmid, r=2))
    logging.info(f'pairwise comparing {len(components_by_pmid)} documents')
    for pmid1, pmid2 in tqdm.tqdm(combos):
        scores.append(
            {
                'pmid1': pmid1,
                'pmid2': pmid2,
                'score': len(components_by_pmid[pmid1] & components_by_pmid[pmid2])
                / len(components_by_pmid[pmid1] | components_by_pmid[pmid2]),
                'shared': components_by_pmid[pmid1] & components_by_pmid[pmid2],
            }
        )

    scores_df = pd.DataFrame.from_records(scores)

    # only keep 0 scores among papers that have a non-zero score with at least one other paper (to avoid unannoted errors)
    mapping = {}
    for _, row in scores_df[scores_df.score == 0].iterrows():
        mapping.setdefault(row['pmid1'], set()).add(row['pmid2'])
        mapping.setdefault(row['pmid2'], set()).add(row['pmid1'])
    return mapping


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('document_chunks')
    parser.add_argument('output')
    args = parser.parse_args()

    input_df = read_jsonl(args.input)
    input_df = input_df[input_df['civic.ground_truth'] == True]
    logging.info(f'read {input_df.shape[0]} input pairs')
    original_columns = input_df.columns

    df = input_df.copy()
    all_chunks_df = read_jsonl(args.document_chunks)

    doc_scores = get_disimilar_pmids(df, all_chunks_df)
    logging.info(f'found {len(doc_scores)} PMIDs with no common elements to at least 1 other pmid')
    logging.info(f'found {sum([len(v) for v in doc_scores.values()])} PMID pairs')
    logging.info(f'writing: {args.output}')
    with open(args.output, 'w') as fh:
        json.dump(doc_scores, fh, cls=NpEncoder)
