import argparse
import logging
import re

import pandas as pd
import requests_cache
import xmltodict

from article_utils.util import dump_jsonl

requests_cache.install_cache('.requests_cache')

CELLOSAURUS_URL = 'https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo'


def _extract_text(text):
    if isinstance(text, dict):
        return _extract_text(text['#text'])
    if isinstance(text, list):
        return [_extract_text(s) for s in text]
    return text if not pd.isnull(text) else ''


def extract_text(text):
    text = _extract_text(text)
    if isinstance(text, list):
        text = ';'.join(sorted(set(text) - {''}))
    return text


def pull_derives(row):
    derives = []
    for item in row['rdfs:subClassOf']:
        if 'owl:Restriction' not in item:
            continue
        try:
            if item["owl:Restriction"]["owl:onProperty"]["@rdf:resource"] in {
                "derives from patient having disease",
                'is disease model for',
            }:
                derives.append(item["owl:Restriction"]["owl:someValuesFrom"]["@rdf:resource"])
        except (TypeError, KeyError) as err:
            logging.error(str(item))
            logging.error(str(err))
            pass

    if row['rdfs:comment'].startswith('disease: '):
        comment = row['rdfs:comment'].replace('disease: ', '')
        derives.extend([re.sub(r'\s+', ' ', disease.strip()) for disease in comment.split(';')])

    derives = sorted(set([d.split('/')[-1].replace('DOID_', 'DOID:') for d in derives]))
    return ';'.join(derives)


def replace_ids_with_labels(record, mapping):
    if isinstance(record, dict):
        return {
            replace_ids_with_labels(k, mapping): (
                replace_ids_with_labels(v, mapping) if k != '@rdf:about' else v
            )
            for k, v in record.items()
        }
    if isinstance(record, list):
        return [replace_ids_with_labels(v, mapping) for v in record]
    if isinstance(record, str):
        name = record
        if name.startswith('http'):
            name = name.split('/')[-1]
        if ':' in name:
            name = name.split(':')[1]
        return labelsById.get(name, record)
    return record


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--clo', help='path to the CLO OWL file', required=True)
parser.add_argument('--cellosaurus', help='path to the cellosaurus jsonl file', required=True)
parser.add_argument('--ncit', help='path to the simplified NCIT file', required=True)
parser.add_argument('output')
args = parser.parse_args()

labelsById = {}
recordsById = {}

logging.info(f'reading: {args.clo}')
with open(args.clo, 'r') as fh:
    data = xmltodict.parse(fh.read())['rdf:RDF']

for group in data:
    if not isinstance(data[group], list):
        continue
    for record in data[group]:
        if not isinstance(record, dict):
            continue
        try:
            rid = record['@rdf:about']
            assert rid not in recordsById
            recordsById[rid] = record
            labelsById[rid.split('/')[-1]] = record['rdfs:label']['#text']
        except (TypeError, KeyError):
            pass
        except Exception as err:
            logging.error(str(record))
            raise err

labelsById = {id: label for id, label in labelsById.items() if label is not None}
prefix = 'http://purl.obolibrary.org/obo'
records = [replace_ids_with_labels(rec, labelsById) for rec in recordsById.values()]
df = pd.DataFrame.from_records(records).fillna('')
df['rdfs:comment'] = df['rdfs:comment'].apply(extract_text)

df['derives_from'] = df.apply(pull_derives, axis=1)
df['name'] = (
    df['rdfs:label']
    .str['#text']
    .apply(lambda text: text if not isinstance(text, dict) else text['#text'])
).fillna('')
df['id'] = df['@rdf:about'].str.split('/').str[-1]
df = df[df['derives_from'] != '']

# now create xrefs to cellosaurus which bern2 uses for annotationimport obonet

logging.info(f'loading: {args.cellosaurus}')
full_cellosaurus_df = pd.read_json(args.cellosaurus, lines=True, dtype=False)

cellosaurus_df = full_cellosaurus_df.explode('xref')
# only keep references to the CLO
cellosaurus_df = cellosaurus_df[cellosaurus_df.xref.str.startswith('CLO:')]
cellosaurus_df['xref'] = cellosaurus_df.xref.str.replace('CLO:', '')
cellosaurus_df = cellosaurus_df.rename(
    columns={c: f'cellosaurus.{c}' for c in cellosaurus_df.columns}
)

df = df.merge(cellosaurus_df, left_on='id', right_on='cellosaurus.xref', how='outer').fillna('')

# ones where we can't map them back to CLO?
logging.info(f'reading: {args.ncit}')
ncit_df = pd.read_json(args.ncit, lines=True, dtype=False)
ncit_df = ncit_df[(~ncit_df.name.isna()) & (ncit_df.name != 0)]


def merge_names(row):
    names = set([str(row.name), *row.synonyms])
    return sorted(names)


ncit_df['derives_from'] = ncit_df.apply(merge_names, axis=1)
ncit_df = ncit_df.rename(columns={'id': 'xref'})
cellosaurus_df = full_cellosaurus_df.explode('xref')
# only keep references to NCIT
cellosaurus_df = cellosaurus_df[cellosaurus_df.xref.str.startswith('NCIt:')]


def clean_cellosaurus_synonyms(synonyms):
    if isinstance(synonyms, str):
        synonyms = [synonyms]
    if not isinstance(synonyms, list):
        return []
    result = []
    for synonym in synonyms:
        synonym = re.sub(r'\s+\S+\s+\[\s*\]$', '', synonym)
        result.append(synonym.strip('"'))
    return result


cellosaurus_df['synonyms'] = cellosaurus_df['synonym'].apply(clean_cellosaurus_synonyms)
cellosaurus_df['xref'] = cellosaurus_df.xref.str.replace('NCIt:', '')
cellosaurus_df = cellosaurus_df.merge(ncit_df[['xref', 'derives_from']])
cellosaurus_df = cellosaurus_df.rename(columns={c: f'cellosaurus.{c}' for c in ['id', 'xref']})
cellosaurus_df = cellosaurus_df.rename(columns={'cellosaurus.xref': 'ncit.id'})


def simplify_cell_names(names):
    names = set(names.split(';')) - {''}
    for name in list(names):
        names.add(re.sub(r'\scells?$', '', name))
    return sorted(names)


df['synonyms'] = df['obo:IAO_0000118'].apply(extract_text) + ';' + df['cellosaurus.name']
df['synonyms'] = df.synonyms.apply(simplify_cell_names)
df['derives_from'] = df.derives_from.apply(simplify_cell_names)

df = pd.concat(
    [
        df,
        cellosaurus_df[['cellosaurus.id', 'name', 'derives_from', 'synonyms', 'ncit.id']],
    ]
)


dump_jsonl(
    df[['derives_from', 'id', 'name', 'synonyms', 'cellosaurus.id', 'ncit.id']].rename(
        columns={'id': 'clo.id'}
    ),
    args.output,
)
