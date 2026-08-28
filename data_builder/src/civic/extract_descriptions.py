import argparse
import itertools
import logging
import re

import pandas as pd
import tqdm
from rapidfuzz import fuzz

from article_utils.acronyms import sub_acronym
from article_utils.util import dump_jsonl
from civic.util import load_dump
from nli.util import Text, build_civic_claim, concat_text

tqdm.tqdm.pandas()

GENE_MATCH = 0.5
HAS_ANNOTATIONS = 0.05
IS_NA = 4
TEXT_NAME_MATCH = 0.5
VARIANT_MATCH = 1

# since this is pretty loose matching we are forgiving on similarities
SWAPS = [
    {'loss', '-/-', 'deletion', 'null', 'deficien'},
    {
        'overexpression',
        'status',
        'express',
        'level',
        'staining',
        'overactive',
        'upregulat' 'expression',
    },
    {
        'underexpression',
        'express',
        'level',
        'staining',
        'deficien',
        'silenced',
        'ihc',
        'immunohistochemistry',
        'expression',
        'negative',
        'status',
    },
    {
        'mutation',
        'mutant',
        'variant',
        'mutated',
        'alteration',
        '-positive',
        'ihc',
        'immunohistochemistry',
        'expression',
        'positive',
    },
    {'fusion', 'rearrangement', 'rearranged', 'positive'},
    {'loss of function', 'silencing', 'deleterious'},
    {'nuclear expression', 'nuclear'},
    {
        'amplification',
        'copy-number',
        'copy number',
        'copy gain',
        'ihc',
        'immunohistochemistry',
        'amplified',
    },
    {'itd', 'tandem repeat', 'internal tandem duplication'},
]


def preprocess_variants(variants):
    result = []

    for variant in variants:
        name = variant['name']
        m = re.match(r'^(\w+::\w+) (\w+)$', name)
        if m:
            variant.pop('name')
            result.append(dict(**variant, name=m.group(2)))
            result.append(dict(**variant, name=m.group(1)))
        else:
            result.append(variant)
    return result


def normalize(input_set):
    if not isinstance(input_set, set) and not isinstance(input_set, list) and pd.isnull(input_set):
        return set()
    return {item.strip().lower() for item in input_set if not pd.isnull(item)}


def pull_disease_names(row):
    possible_names = set() if not isinstance(row['do.synonyms'], list) else set(row['do.synonyms'])
    possible_names.update([row['civic.disease.name'], row['do.name']])
    possible_names = normalize(possible_names)

    for name in list(possible_names):
        possible_names.add(
            name.replace('malignant ', '')
        )  # generally the malignant part is not particularly informative
        if name == 'solid tumor':
            possible_names.add('solid tumors')
    return possible_names


def pull_disease_ev_names(row):
    possible_names = set() if not isinstance(row['do.synonyms'], list) else set(row['do.synonyms'])
    possible_names.update([row['civic.disease.name'], row['do.name']])
    possible_names = normalize(possible_names)

    for name in list(possible_names):
        possible_names.add(
            name.replace('malignant ', '')
        )  # generally the malignant part is not particularly informative
    return possible_names


def pull_disease_evidence(row, mesh_names_lookup, clo_lookup):
    annotation_ids = set()
    annotation_names = set()
    if isinstance(row['civic.pubtator.annotations'], list):
        for annotation in row['civic.pubtator.annotations']:
            mention = annotation['text'].lower().strip()
            mention_type = annotation['type']
            if mention_type not in {'disease', 'cellline'}:
                continue
            for ann_id in annotation['id']:
                if ann_id and ann_id != 'CUI-less':
                    annotation_names.update(
                        mesh_names_lookup.get(ann_id, set())
                        if mention_type == 'disease'
                        else clo_lookup.get(ann_id, set())
                    )

            if annotation['type'] == 'cellline':
                annotation_names.update(clo_lookup.get(mention, set()))
            annotation_names.add(mention)

    return normalize(annotation_ids), normalize(annotation_names)


def check_text_match(claim_names: set[str], evidence_text: str, threshold=70):
    for name in claim_names:
        if name in evidence_text:
            return TEXT_NAME_MATCH
    for name in claim_names:
        if len(name) < len(evidence_text):
            return 0
        alignment = fuzz.partial_ratio_alignment(evidence_text, name, score_cutoff=threshold)
        if alignment:
            match_text = evidence_text[alignment.src_start : alignment.src_end]
            return TEXT_NAME_MATCH
    return 0


def check_disease(row):
    if pd.isnull(row['civic.disease.name']) or row['civic.disease.name'].lower() == 'cancer':
        return IS_NA
    id_match = 2 if row['disease_ids'] & row['ev.disease_ids'] else 0
    name_match = 1 if row['disease_names'] & row['ev.disease_names'] else 0
    has_annotations = HAS_ANNOTATIONS if row['ev.disease_names'] | row['ev.disease_ids'] else 0
    desc = row['text'].strip().lower()
    names = row['disease_names']
    for name in list(names):
        names.add(name.replace('tumour', 'tumor'))
        names.add(name.replace('tumor', 'tumour'))

    acronyms = set()

    do_synonyms = row['do.synonyms'] if isinstance(row['do.synonyms'], list) else []
    for name in do_synonyms:
        acronym = name.split(' - ')[0].strip().upper()
        if name.startswith(acronym) and len(acronym) > 1:
            acronyms.add(acronym)

    text_match = TEXT_NAME_MATCH if check_text_match(row['disease_names'], desc) else 0

    if check_text_match(acronyms, row['civic.description']):
        # should be case sensitive
        text_match = TEXT_NAME_MATCH

    return name_match + id_match + has_annotations + text_match


def check_drugs(row):
    drug_annotations = (
        []
        if not isinstance(row['civic.pubtator.annotations'], list)
        else row['civic.pubtator.annotations']
    )
    drug_annotations = [a for a in drug_annotations if a['type'].lower() == 'chemical']
    ev_drug_names = normalize([d['text'] for d in drug_annotations])
    status = []
    desc = row['text'].strip().lower()
    text_match = []
    missing_drugs = []
    for drug in row['civic.therapies'] or []:
        names = normalize([drug['name']] + drug.get('therapyAliases', []))
        for name in list(names):
            names.add(re.sub(r'^(.*) inhibitors?$', r'anti-\1', name))
            names.add(re.sub(r'^(.*) inhibitors?$', r'\1 antibody', name))

        tmatch = TEXT_NAME_MATCH if check_text_match(names, desc) else 0
        match = bool(names & ev_drug_names)
        if not (match or tmatch):
            missing_drugs.append(drug['name'])
        text_match.append(tmatch)
        status.append(match)

    total = 0
    if not status:
        return (IS_NA, missing_drugs)
    if all(status):
        total += 1
    if all(text_match):
        total += TEXT_NAME_MATCH
    if drug_annotations:
        total += HAS_ANNOTATIONS
    return total, missing_drugs


def expand_variant_names(variant: dict) -> set[str]:
    """
    Create alternative/equivalent variant names
    """
    variant_names = normalize([variant['name']] + variant['variantAliases'])

    if 'gene2' in variant:
        # we added gene2 info to the civic variant entry
        for gene1, gene2 in itertools.product(
            variant['gene']['geneAliases'], variant['gene2']['geneAliases']
        ):
            variant_names.add(f'{gene1}::{gene2}')

    for name in list(variant_names):
        m = re.search(r'(\(c\.[^)]+\))', name)
        if m:
            variant_names.add(name.replace(m.group(1), '').strip())
            variant_names.add(m.group(1)[1:-1].strip())
        variant_names.add(name.replace('::', '-'))
        variant_names.add(name.replace('::', '/'))
        variant_names.add(name.replace('::', ':'))
        variant_names.add(name.replace('-', ' '))
        variant_names.add(re.sub(r'(variant|mutation|fusion)\b', r'\1s', name))
        variant_names.add(re.sub(r'(exon \d+) mutation', r'\1', name))
        variant_names.add(re.sub(r'^\s*(\S+\s+domain)\s+mutations?\s*$', r'\1', name))
        variant_names.add(re.sub(r' mutations?', '', name))

    for swap in SWAPS:
        for name in list(variant_names):
            if name in swap:
                variant_names.update(swap)

    return normalize(variant_names) - {''}


def check_mp(row):
    status = []
    missing_variants = []
    desc = row['text'].lower()
    for variant in row['civic.molecularProfile.variants']:
        # don't normalize gene names, they are generally cased ok and can be regular words when lowercase
        gene_names = normalize([variant['gene']['name']] + variant['gene']['geneAliases'])
        variant_names = expand_variant_names(variant)

        gene_match = GENE_MATCH if check_text_match(gene_names, desc) else 0
        variant_match = VARIANT_MATCH if check_text_match(variant_names, desc) else 0

        if variant["gene"]["name"] in variant['name']:
            display_name = variant['name']
        else:
            display_name = f'{variant["gene"]["name"]} {variant["name"]}'

        if not variant_match:
            missing_variants.append(display_name)
        elif not gene_match:
            missing_variants.append(f"gene: {variant['gene']['name']}")

        status.append(gene_match + variant_match)
    return min(status), missing_variants


def check_phenotypes(row):
    if not row['civic.phenotypes']:
        return IS_NA, []
    desc = row['text'].lower()
    missing_phenotypes = []
    for phenotype in row['civic.phenotypes']:
        if phenotype['name'].lower() == 'adult onset':
            continue
        names = normalize([phenotype['name']] + phenotype['phenotypeAliases'])
        for name in list(names):
            if name.endswith(' onset'):
                names.add(name.replace(' onset', ''))
        check = check_text_match(names, desc)
        if not check:
            if phenotype['name'].lower().endswith(' onset'):
                if any(
                    [
                        re.search(r'\bage[sd]?\b', desc),
                        re.search(r'y(ea)?r-old', desc),
                        re.search(r'\d+(\.\d+)? [ym]\.?o\.?\b', desc),
                    ]
                ):
                    check = True  # forgiving since more likely to be correct than not in desc

        if not check:
            # assume adult onset unless ages described otherwise
            missing_phenotypes.append(phenotype['name'])
    return TEXT_NAME_MATCH if not missing_phenotypes else 0, missing_phenotypes


def build_cellline_lookup(filename):
    logging.info(f'reading: {filename}')
    clo_df = pd.read_json(filename, lines=True, dtype=False)

    lookup: dict[str, set[str]] = {}

    for _, row in tqdm.tqdm(
        clo_df.iterrows(), desc='converting cells dataframe to mapping', total=clo_df.shape[0]
    ):
        names = normalize(
            [row['name'], row["cellosaurus.id"], row["clo.id"]] + row.synonyms + row.derives_from
        ) - {''}
        for name in list(names):
            if 'cells' in name:
                names.add(name.replace('cells', 'cell'))
            elif 'cell' in name:
                names.add(name.replace('cell', 'cells'))

        for name in names:
            lookup.setdefault(name, set()).update(names)
    return lookup


def reasoning(row):
    missing = []
    if row['check_disease'] < TEXT_NAME_MATCH:
        missing.append(('disease', row['civic.disease.name']))

    for col, name in [
        ('missing_variants', 'variants'),
        ('missing_drugs', 'therapies'),
        ('missing_phenotypes', 'phenotypes'),
    ]:
        if row[col]:
            items = row[col]
            if len(items) == 1:
                if items[0].startswith('gene: '):
                    items = [items[0].replace('gene: ', '')]
                    name = 'genes'

                name = re.sub(r's$', '', name)
                name = re.sub(r'ie$', 'y', name)

            missing.append((name, ', '.join(items)))

    if not missing:
        return ''

    if len(missing) > 1:
        missing = [f'{item_type} ({values})' for item_type, values in missing]

        text = '; '.join(missing[:-1]) + (';' if len(missing) > 2 else '') + ' and ' + missing[-1]
        reason = (
            f'Several elements of the claim are not mentioned in the evidence including: {text}'
        )
    else:
        missing = missing[0]
        verb = 'were' if missing[0].endswith('s') else 'was'
        missing = '{} ({})'.format(*missing)
        reason = f'The {missing} {verb} not mentioned in the evidence'
    return reason


def is_unique_index(df, columns):
    return df.shape[0] == df.drop_duplicates(columns).shape[0]


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument(
        '--do',
        required=True,
        help='path to the disease ontology file with the MESH terms (csv)',
    )
    parser.add_argument(
        '--clo', required=True, help='path to the simplified Cell line ontology terms'
    )
    parser.add_argument('--acronyms', required=True)
    parser.add_argument('output')
    args = parser.parse_args()

    civic_df = load_dump(args.input, prefix='civic')
    civic_df = civic_df[civic_df['civic.evidenceDirection'].isin({'SUPPORTS', "DOES_NOT_SUPPORT"})]
    civic_df = civic_df[
        (civic_df['civic.ground_truth'] == True) & (civic_df['civic.status'] == 'ACCEPTED')
    ]

    # if there are mutliple entries using the same description, pick the first one for consistency
    civic_df = civic_df.sort_values(['civic.id'])
    civic_df = civic_df.drop_duplicates(['civic.description'])
    assert is_unique_index(civic_df, ['civic.id'])

    civic_df['flagged'] = (
        civic_df['civic.significance'].str.lower().isin({'na', 'n/a'}) | civic_df['civic.flagged']
    )
    # logging.info(f'dropping n/a significance entries: {civic_df["_flagged"].sum()}')
    # civic_df = civic_df[civic_df['_flagged'] == False]

    civic_df['flagged'] = (
        civic_df['civic.significance'].str.lower() == 'uncertain significance'
    ) | civic_df['flagged']
    # logging.info(f'dropping uncertain significance entries: {civic_df["_flagged"].sum()}')
    # civic_df = civic_df[civic_df['_flagged'] == False]

    civic_df['flagged'] = (civic_df['civic.source.sourceType'] != 'PUBMED') | civic_df['flagged']
    # logging.info(f'dropping non-pubmed entries: {civic_df["_flagged"].sum()}')
    # civic_df = civic_df[civic_df['_flagged'] == False]

    logging.info(f'starting labels: {dict(civic_df["civic.evidenceDirection"].value_counts())}')

    # add the acronyms
    logging.info(f'reading: {args.acronyms}')
    acronyms_df = pd.read_json(args.acronyms, lines=True, dtype=False)

    acronyms_lookup = {}
    for _, row in acronyms_df.iterrows():
        acronyms_lookup.setdefault(row.pmid, {})[row.acronym] = row.defn

    def replace_acronyms(row):
        text = row['civic.description']
        for _, arow in acronyms_df[acronyms_df.pmid == row['civic.source.citationId']].iterrows():
            text = sub_acronym(text, arow.acronym, arow.defn)
        return text

    civic_df['text'] = civic_df.apply(replace_acronyms, axis=1)

    logging.info(f'reading: {args.do}')
    do_df = pd.read_json(args.do, lines=True, dtype=False)
    do_df['id'] = do_df.id.str.replace('DOID:', '')
    do_df = do_df.rename(columns={c: f'do.{c}' for c in do_df.columns})
    assert is_unique_index(do_df, ['do.id'])

    df = civic_df.merge(do_df, how='left', left_on='civic.disease.doid', right_on='do.id')
    assert df.shape[0] == civic_df.shape[0]
    is_unique_index(df, ['civic.id'])

    mesh_names_lookup: dict[str, set[str]] = {}
    logging.info('generating mesh names lookup and disease terms fields')
    for _, row in do_df[~do_df['do.mesh'].isna()].iterrows():
        for mesh_id in row['do.mesh']:
            mesh_names_lookup[mesh_id] = normalize(row['do.synonyms'])

    df['disease_names'] = df.apply(pull_disease_names, axis=1)
    df['disease_ids'] = df['do.mesh'].apply(normalize)
    clo_lookup = build_cellline_lookup(args.clo)
    logging.info(f'normalizing and matching disease names')

    df['ev.disease_ids'], df['ev.disease_names'] = zip(
        *df.progress_apply(
            lambda row: pull_disease_evidence(row, mesh_names_lookup, clo_lookup), axis=1
        )
    )
    logging.info('checking disease terms')
    df['check_disease'] = df.apply(check_disease, axis=1)
    logging.info('checking drug terms')
    df['check_drugs'], df['missing_drugs'] = zip(*df.apply(check_drugs, axis=1))
    logging.info('preprocessing variants to split old-style combination variants')

    df['civic.molecularProfile.variants'] = df['civic.molecularProfile.variants'].apply(
        preprocess_variants
    )
    logging.info('checking variant terms')
    df['check_mp'], df['missing_variants'] = zip(*df.apply(check_mp, axis=1))

    logging.info('checking phenotype terms')
    df['check_phenotypes'], df['missing_phenotypes'] = zip(
        *df.apply(lambda row: check_phenotypes(row), axis=1)
    )

    logging.info('assigning a term-match status')
    # set the overall status based on the above checks
    df['status'] = False
    df.loc[
        (df.check_mp >= 1.5)
        & (df.check_drugs >= TEXT_NAME_MATCH)
        & (df.check_disease >= TEXT_NAME_MATCH)
        & (df.check_phenotypes >= TEXT_NAME_MATCH),
        'status',
    ] = True
    print(
        df[df['civic.id'] == '1410'][
            ['check_mp', 'check_drugs', 'check_disease', 'check_phenotypes', 'status']
        ].drop_duplicates()
    )
    logging.info('generating reasoning')
    df['reason'] = ''
    df.loc[df.status == False, 'reason'] = df[df.status == False].apply(reasoning, axis=1)

    ### now build into claims and evidence
    df = df.copy()
    df['document.pmid'] = df['civic.source.citationId']
    df['document.pmcid'] = df['civic.source.pmcId']

    df['gold_label_name'] = df['civic.evidenceDirection'].apply(
        lambda label: 'SUPPORTS' if label != 'DOES_NOT_SUPPORT' else 'REFUTES'
    )
    df.loc[df.status == False, 'gold_label_name'] = "NEI"

    df['description.status'] = df['status']
    df['claim.generated'] = df['evidence.generated'] = df['generated'] = df['in_table'] = False
    df['evidence.source'] = df['claim.source'] = 'civic'
    df['evidence.selection_method'] = 'none'
    df['hypothesis.tag'] = ''
    df['hypothesis.username'] = 'none'
    df['example'] = 'description'
    df['conversation'] = df.apply(lambda _: [], axis=1)
    df['claim.id'] = df['evidence.id'] = df['civic.id']
    df['claim'] = df.apply(build_civic_claim, axis=1)
    df['evidence'] = df.apply(
        lambda row: [
            Text(
                row['civic.description'],
                meta=dict(start=0, end=len(row['civic.description']), id=row['evidence.id']),
            )
        ],
        axis=1,
    )
    df['evidence.flat'] = df.evidence.apply(concat_text)
    df['claim.flat'] = df.claim.apply(concat_text)
    df['id'] = (
        'EID' + df['civic.id'].astype(str) + '_' + df['civic.revision_state'].fillna('') + '_desc'
    )

    check_cols = [c for c in df.columns if c.startswith('check_')]

    assert is_unique_index(df, ['id'])

    df = (
        df.groupby(['id'], dropna=False)
        .apply(lambda s: s[check_cols].to_dict('records')[0])
        .reset_index(name='description.scores')
        .merge(df.drop(columns=check_cols), on='id')
    )
    df['reviews'] = df.apply(
        lambda row: (
            [{'reason': row.reason, 'username': 'bot', 'label': row.gold_label_name}]
            if row.reason
            else []
        ),
        axis=1,
    )
    keep_columns = [
        'id',
        'reviews',
        'gold_label_name',
        'example',
        'conversation',
        'flagged',
        'in_table',
        'generated',
        'description.status',
        'description.scores',
        'civic.id',
        'civic.status',
        'civic.flagged',
        'civic.evidenceDirection',
        'civic.description',
        'civic.events',
        'civic.revision_state',
        'civic.evidenceRating',
        'civic.evidenceLevel',
        'civic.evidenceType',
        'civic.corrections',
        'civic.ground_truth',
        'civic.lastUpdatedAt',
        'civic.submittedAt',
    ]
    for col in df.columns:
        if any(
            [col.startswith(prefix) for prefix in ['evidence', 'claim', 'hypothesis', 'document']]
        ):
            keep_columns.append(col)
    logging.info(str(dict(df.gold_label_name.value_counts())))
    dump_jsonl(df[keep_columns], args.output)


if __name__ == '__main__':
    main()
