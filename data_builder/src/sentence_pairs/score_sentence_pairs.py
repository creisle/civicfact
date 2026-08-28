import argparse
import logging
import re
from functools import lru_cache

import pandas as pd
import tqdm

from article_utils.acronyms import sub_acronym
from article_utils.util import (
    dump_jsonl,
    listify,
    normalize_and_expand_column,
    read_jsonl,
)

tqdm.tqdm.pandas()


def normalize_text(s: str) -> str:
    # collapse non-word runs to single spaces; lowercase
    return re.sub(r'[\W_]+', ' ', s).lower()


def split_list(input_list, chunk_size):
    # https://github.com/chroma-core/chroma/issues/1049#issuecomment-1699859480
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]


def text_match_variant(variant_name, text):
    if variant_name.lower().strip() in text.lower().strip():
        return True
    m = re.match(r'\D*(\d+).*', variant_name)
    if (
        m
        and len(m.group(1)) > 2
        and (
            re.search(r'\b' + m.group(1) + r'\b', text) or re.search(r'[a-z]\.' + m.group(1), text)
        )
    ):
        return True
    if ' > ' in text:
        return text_match_variant(variant_name, text.replace(' > ', '>'))
    return False


@lru_cache(maxsize=4096)
def _compile_gene_pattern(gene, flags):
    patt = r'\b' + gene
    return re.compile(patt, flags=flags)


def text_match_gene(gene, text):

    flags = re.MULTILINE
    if len(gene) > 2:
        flags |= re.IGNORECASE
    if _compile_gene_pattern(gene, flags=flags).search(text):
        return True
    # print('text_match_gene', [patt, bool(re.search(patt, text)), text])
    return False


@lru_cache(maxsize=4096)
def _compiled_disease_pattern(disease_name: str) -> re.Pattern:
    patt = re.escape(normalize_text(disease_name))
    patt = patt.replace('cancer', r'(?:cancer|tumou?r)')
    patt = re.sub(r's\b', r's?', patt)
    return re.compile(patt)


@lru_cache(maxsize=4096)
def text_match_disease(disease_name, text):
    return bool(_compiled_disease_pattern(disease_name).search(normalize_text(text)))


def expand_acronyms(row):
    text = row['chunk.text']
    for acronym in listify(row['acronyms']):
        text = sub_acronym(text, acronym['name'], acronym['defn'])
    return text


def similarity(row):
    sim = {'chunk.id': row['chunk.id']}
    chunk_text = row['chunk.text'].strip()

    gene_matches = []
    variant_matches = []
    # gene/variant matches
    for variant in row['civic.elements']['variants']:
        curr_gene_matches = 0
        for gene in variant['genes']:
            if set(gene) & row['pubtator.elements'].get('genes', set()):
                curr_gene_matches += 1
            elif any([text_match_gene(g, chunk_text) for g in gene]):
                curr_gene_matches += 1
        gene_matches.append(curr_gene_matches > 0)

        variant_match = False

        if set(variant.get('variant', set())) & row['pubtator.elements'].get('variants', set()):
            variant_match = True
        else:
            for variant_alias in variant.get('variant', []):
                if text_match_variant(variant_alias, chunk_text):
                    variant_match = True
                    break

        variant_matches.append(variant_match)

    # check for a disease match
    disease_matches = []
    for disease in row['civic.elements']['diseases']:
        match = bool(
            (set(disease) & row['pubtator.elements'].get('diseases', set()))
            or any([text_match_disease(d, chunk_text) for d in disease])
        )
        disease_matches.append(match)

    # check if all drugs match
    drug_matches = []

    for drugs in row['civic.elements']['drugs']:
        match = bool(
            (set(drugs) & row['pubtator.elements'].get('drugs', set()))
            or any([d.lower() in chunk_text.lower() for d in drugs])
        )
        drug_matches.append(match)

    for name, matches in zip(
        ['gene', 'variant', 'disease', 'therapy'],
        [gene_matches, variant_matches, disease_matches, drug_matches],
    ):
        if matches:
            score = sum(matches) / len(matches)
            sim[name] = max([sim.get(name, score), score])

    sim['score'] = sum([v for v in sim.values() if v and not isinstance(v, str)] + [0]) / sum(
        [1 for v in sim.values() if not pd.isnull(v)]
    )
    sim['document_match'] = row['document_match']
    return sim


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument(
        '--limit',
        required=False,
        type=int,
        help='limit to x number of total pairs per claim. only applicable when using inter_doc flag',
    )
    parser.add_argument(
        '--inter_doc',
        action='store_true',
        default=False,
        help='create pairs outside a single document',
    )
    parser.add_argument('--document_chunks', required=True)

    parser.add_argument('output')
    args = parser.parse_args()

    # generate all possible claims from civic entries
    civic_df = read_jsonl(args.input)
    civic_df = civic_df[civic_df['civic.ground_truth'] == True]

    logging.info(f'reading: {args.document_chunks}')
    chunks_df = pd.read_json(args.document_chunks, lines=True, dtype=False)
    print(chunks_df.columns.tolist())
    chunks_df['chunk.section'] = chunks_df.passages.fillna('').apply(
        lambda passages: ';'.join(set([p['passage.section'] for p in passages]))
    )
    print(chunks_df['chunk.section'].unique().tolist())
    chunks_df['text'] = chunks_df.apply(expand_acronyms, axis=1)

    def merge_pubtator_elements(row):
        elements = {}
        for annotation in listify(row['pubtator']):
            for cat, values in annotation['pubtator.elements'].items():
                elements.setdefault(cat, set()).update(values)
        return elements

    chunks_df['pubtator.elements'] = chunks_df.apply(merge_pubtator_elements, axis=1)

    if args.inter_doc:
        sim_df = civic_df.merge(chunks_df, how='cross')
    else:
        civic_df = civic_df.explode('civic.pmid').rename(columns={'civic.pmid': 'document.pmid'})
        sim_df = civic_df.merge(chunks_df, on=['document.pmid'])

    logging.info(f'{sim_df.shape[0]} possible combinations')
    assert sim_df[sim_df['claim.flat'].fillna('') == ''].shape[0] == 0, 'cross merge failed'
    assert sim_df[sim_df['chunk.id'].fillna('') == ''].shape[0] == 0, 'cross merge failed'

    # rearrange the dataframe the cumulative count the claim pairings to randomly select n pairings per claim
    sim_df = sim_df.sample(frac=1)

    def check_document_match(row):
        return bool(set(row['civic.pmid']) & set([row['document.pmid']]))

    logging.info(f'computing overlap with claim and annotation overlap on this fragment')
    if args.inter_doc:
        sim_df['document_match'] = sim_df.apply(check_document_match, axis=1)
    else:
        sim_df['document_match'] = True

    logging.info(f'computing {sim_df.shape[0]} scores')
    ### compute similarity between all claims from civic_df with all chunks from chunks_df
    sim_df['similarity'] = sim_df.progress_apply(similarity, axis=1)
    sim_df = sim_df[
        [
            'civic.id',
            'claim.flat',
            'document.pmid',
            'document.length',
            'document.content_source',
            'chunk.id',
            'chunk.text',
            'similarity',
            'chunk.section',
            'text',
        ]
    ].reset_index(drop=True)
    sim_df = normalize_and_expand_column(sim_df, 'similarity')
    assert sim_df[sim_df['similarity.score'].isnull()].shape[0] == 0, 'normalize and expand failed'

    dump_jsonl(
        sim_df[
            [
                'claim.flat',
                'document.pmid',
                'document.content_source',
                'document.length',
                'chunk.id',
                'chunk.section',
            ]
            + [c for c in sim_df.columns if c.startswith('similarity')]
        ],
        args.output,
    )


if __name__ == '__main__':
    main()
