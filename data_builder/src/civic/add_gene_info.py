import argparse
import logging
import re

import pandas as pd

from article_utils.util import dump_jsonl


def rename_fields(input_dict: dict, **fields) -> dict:
    result = {}
    for k, v in input_dict.items():
        result[fields.get(k, k)] = v
    return result


def map_feature_to_gene(variants) -> None:
    for variant in variants:
        try:
            if variant['feature']['featureInstance']['__typename'] == 'Gene':
                gene = {}
                gene.update(variant['feature'])
                gene.update(variant['feature']['featureInstance'])
                variant['gene'] = rename_fields(gene, featureAliases='geneAliases')
            if variant['feature']['featureInstance']['__typename'] == 'Fusion':
                gene = {}
                if variant['feature']['featureInstance']['fivePrimeGene'] is None:
                    gene.update(variant['feature']['featureInstance']['threePrimeGene'])
                else:
                    gene.update(variant['feature']['featureInstance']['fivePrimeGene'])
                variant['gene'] = rename_fields(gene, featureAliases='geneAliases')
        except Exception as err:
            logging.error(f"failed to extract gene from variant: {variant}")
            raise err


def preprocess_variants(variants):
    result = []

    for variant in variants:
        name = variant['name']
        m = re.match(r'^(\w+::\w+) (\w+)(( and)? (\w+))?$', name)
        if m:
            variant.pop('name')
            result.append(dict(**variant, name=m.group(2)))
            result.append(dict(**variant, name=m.group(1)))
            if m.group(5):
                result.append(dict(**variant, name=m.group(5)))
        else:
            result.append(variant)
    return result


def pick_gene(symbol, lookup) -> dict:
    if len(lookup[symbol]) == 1:
        return lookup[symbol][0]

    for gene in lookup[symbol]:
        logging.error(gene)
    raise ValueError(f'ambiguous gene symbol: {symbol}')


def add_second_gene(variant, lookup) -> tuple[dict, bool]:
    """
    Add details for the second gene involved in the fusion
    """
    m = re.match(r'^(\S+)::(\S+)(\s+e\d+-e\d+)?$', variant['name'].strip())
    if not m:
        raise ValueError(f'fusion does not match expected pattern')
    gene1 = m.group(1)
    gene2 = m.group(2)
    missing_gene = gene2 if gene2 not in variant['gene']['geneAliases'] else gene1
    missing_gene1 = missing_gene == gene1
    assert missing_gene not in variant['gene']['geneAliases']
    missing_gene = pick_gene(missing_gene, lookup)
    missing_gene = {
        'name': missing_gene['Symbol_from_nomenclature_authority'],
        'geneAliases': missing_gene['geneAliases'],
        'entrezId': missing_gene['GeneID'],
    }
    if missing_gene1:
        variant['gene2'] = variant['gene']
        variant['gene'] = missing_gene
    else:
        variant['gene2'] = missing_gene
    return variant


def non_gene_feature(molecularProfile):
    for variant in molecularProfile['variants']:
        if variant['feature']['featureInstance']['__typename'] == 'Factor':
            return True
    return False


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('civic')
parser.add_argument('genes')
parser.add_argument('output')
args = parser.parse_args()

logging.info(f'reading: {args.civic}')
civic_df = pd.read_json(args.civic, lines=True, dtype=False)
civic_df['_non_gene'] = civic_df.molecularProfile.apply(non_gene_feature)
logging.warning(
    f'dropping {civic_df[civic_df._non_gene].shape[0]} entries without a gene type feature'
)
logging.warning(f'ids of non-gene feature entries: {civic_df[civic_df._non_gene]["id"].tolist()}')
civic_df = civic_df[civic_df._non_gene == False]

logging.info(f'reading: {args.genes}')
genes_df = pd.read_csv(args.genes, sep='\t')

genes_lookup = {}
symbol_lookup = {}

for _, row in genes_df.iterrows():
    names = set(
        [
            row.Symbol,
            row.Symbol_from_nomenclature_authority,
            row.Full_name_from_nomenclature_authority,
        ]
        + row.Synonyms.split('|')
    )
    genes_lookup[row.GeneID] = names
    row = {**dict(row), 'geneAliases': names}
    for symbol in row['geneAliases']:
        symbol_lookup.setdefault(symbol, []).append(row)


# resolve conflicts
for symbol in symbol_lookup:
    genes = symbol_lookup[symbol]
    matched = [gene for gene in genes if gene['Symbol'] == symbol]
    if len(matched) == 1:
        symbol_lookup[symbol] = matched


def augment_genes(molecularProfile):
    map_feature_to_gene(molecularProfile['variants'])
    molecularProfile['variants'] = preprocess_variants(molecularProfile['variants'])

    for variant in molecularProfile['variants']:
        try:

            names = genes_lookup.get(variant['gene']['entrezId'], set())
            variant['gene']['geneAliases'] = sorted(set(variant['gene']['geneAliases']) | names)
        except KeyError as err:
            logging.warning(f'failed to extract gene from variant: {variant}')
            logging.warning(err)
            return None

        if '::' in variant['name']:
            try:
                add_second_gene(variant, symbol_lookup)
            except Exception as err:
                logging.warning(f'failed to process variant: {variant["name"]}: {err}')

    return molecularProfile


civic_df['molecularProfile'] = civic_df['molecularProfile'].apply(augment_genes)
civic_df = civic_df[~civic_df.molecularProfile.isna()]

dump_jsonl(civic_df.drop(columns=[c for c in civic_df.columns if c.startswith('_')]), args.output)
