import argparse
import logging
import re

import pandas as pd
import tqdm

from article_utils.util import dump_jsonl
from civic.util import load_dump
from nli.util import build_civic_claim, concat_text

tqdm.tqdm.pandas()


def normalize_strings(items):
    return (set(items) | {t.strip().lower() for t in items if t is not None}) - {
        'None',
        'none',
        '',
        None,
    }


def expand_variant_aliases(gene_name, variant_name):
    aliases = {variant_name}
    expression_terms = [
        'expression',
        f'{gene_name} expression',
        f'{gene_name} mRNA expression',
        'mRNA expression',
    ]
    if re.match(r'^(under|low|decreased)-?expression$', variant_name, re.IGNORECASE):
        for term in ['low', 'decreased']:
            aliases.update([f'{term} {t}' for t in expression_terms])
        aliases.update(['underexpression', 'under-expression'])
    if re.match(r'^(high|over|increased)-?expression$', variant_name, re.IGNORECASE):
        for term in ['high', 'increased']:
            aliases.update([f'{term} {t}' for t in expression_terms])
        aliases.update(['overexpression', 'over-expression'])
    m = re.match(r'^([^()\s]+)\s*\(\s*([^)]+)\s*\)\s*$', variant_name)
    if m:
        aliases.update(expand_variant_aliases(gene_name, m.group(1)))
        aliases.update(expand_variant_aliases(gene_name, m.group(2)))

    m = re.match(r'^(\d+)([a-z]{3})([A-Z]+)$', variant_name)
    if m:
        aliases.add(m.group(2) + m.group(1) + m.group(3))

    if re.match(r'[a-z]\.(.*)', variant_name):
        aliases.update(expand_variant_aliases(gene_name, variant_name[2:]))
    if variant_name == 'fusion':
        aliases.update(['fusions', 'rearrangment', 'rearrangements'])
    return aliases


def preprocess_civic_elements(row):
    all_variants = []
    # gene/variant matches
    for variant in row['civic.molecularProfile.variants']:
        if variant['feature']['featureInstance']['__typename'] == 'Gene':
            genes = [
                variant['feature']['name'],
                str(variant['feature']['featureInstance']['entrezId']),
                *variant['feature']['featureAliases'],
            ]
            variants = normalize_strings({variant['name'], *variant['variantAliases']})
            for name in list(variants):
                variants.update(
                    normalize_strings(expand_variant_aliases(variant['feature']['name'], name))
                )
            all_variants.append(
                {
                    'variant': variants,
                    'genes': [set(genes) | {g.lower().capitalize() for g in genes}],
                }
            )

        elif variant['feature']['featureInstance']['__typename'] == 'Fusion':
            genes = []
            for gene_pos in ['fivePrime', 'threePrime']:
                # TODO: sometimes these are "genes" that don't have entrezId b/c not real gene
                if variant['feature']['featureInstance'][f'{gene_pos}PartnerStatus'] != 'KNOWN':
                    continue
                current_genes = {
                    str(variant['feature']['featureInstance'][f'{gene_pos}Gene']['entrezId']),
                    *variant['feature']['featureInstance'][f'{gene_pos}Gene']['featureAliases'],
                    variant['feature']['featureInstance'][f'{gene_pos}Gene']['name'],
                }
                current_genes |= {g.lower().capitalize() for g in current_genes}
                genes.append(current_genes)

            variants = normalize_strings({variant['name'], *variant['variantAliases']})
            for name in list(variants):
                variants.update(
                    normalize_strings(expand_variant_aliases(variant['feature']['name'], name))
                )

            all_variants.append({'genes': genes, 'variant': variants})

        else:
            # TODO: match factor-based variants
            pass

    assert (
        len(all_variants) > 0
    ), f'must be at least one variant. found {len(all_variants)} variants'
    assert all(
        [len(v['genes']) for v in all_variants]
    ), f'genes={[len(v["genes"]) for v in all_variants]}'

    diseases = []
    if not pd.isnull(row['civic.disease.name']):
        diseases.append(
            normalize_strings(
                [
                    row['civic.disease.name'],
                    *row['civic.disease.diseaseAliases'],
                    row['civic.disease.doid'],
                    'DOID:' + row['civic.disease.doid'] if row['civic.disease.doid'] else '',
                ]
            )
        )

    if row['civic.evidenceType'].upper() not in ['ONCOGENIC', 'FUNCTIONAL']:
        if not len(diseases):
            logging.warning(
                f'evidence of type {row["civic.evidenceType"]} must have a defined disease'
            )
            return None

    drugs = []

    if isinstance(row['civic.therapies'], list) and len(row['civic.therapies']):

        for therapy in row['civic.therapies']:
            drugs.append(
                normalize_strings([therapy['name'], therapy['ncitId'], *therapy['therapyAliases']])
            )

    if row['civic.evidenceType'].upper() == 'PREDICTIVE':
        if not len(drugs):

            logging.warning('predictive evidence type entries must have a drug defined')
            return None

    return {'drugs': drugs, 'variants': all_variants, 'diseases': diseases}


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('input')

    parser.add_argument('output')
    args = parser.parse_args()

    # generate all possible claims from civic entries
    civic_df = load_dump(args.input, prefix='civic')
    civic_df['civic.elements'] = civic_df.apply(preprocess_civic_elements, axis=1)
    civic_df = civic_df[~civic_df['civic.elements'].isna()]

    grouping = {c: 'first' for c in civic_df.columns}
    civic_df['claim.flat'] = civic_df.apply(lambda row: concat_text(build_civic_claim(row)), axis=1)
    grouping['civic.pmid'] = 'unique'
    grouping['civic.id'] = 'unique'

    civic_df = civic_df.groupby('claim.flat').agg(grouping).reset_index()

    dump_jsonl(
        civic_df[
            [
                'claim.flat',
                'civic.elements',
                'civic.pmid',
                'civic.id',
                'civic.status',
                'civic.revisions.nodes',
                'civic.flagged',
                'civic.ground_truth',
                'civic.revision_state',
            ]
        ],
        args.output,
    )


if __name__ == '__main__':
    main()
