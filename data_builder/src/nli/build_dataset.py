import argparse
import itertools
import json
import logging
import uuid
from json import JSONEncoder

import pandas as pd
from numpy import random

from article_utils.util import INVALID_PUBMED_IDS, dump_jsonl, read_jsonl
from civic.util import load_dump

from .util import (
    CIVIC_CLAIM_COL_ORDER,
    LABEL_COL,
    NEUTRAL_LABEL,
    NEUTRAL_WORKFLOW_NAME,
    Text,
    build_civic_claim,
    concat_text,
)

random.seed(2)

SEPERATOR = ' </s> '

logging.basicConfig(level=logging.INFO)


# mokey patch json encoder to be less dumb
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default


def string_join(series, delimiter=';'):
    return delimiter.join([str(t) for t in series.tolist() if t])


def array_join(series):
    return series.tolist()


def include_table_header(evidence_selections):
    """
    For any example where it includes a table selection, include the table header as an additional selection
    """
    new_selections = []
    headers_seen = set()

    for selection in evidence_selections:
        if selection['meta']['table_header']:
            header = json.loads(selection['meta']['table_header'])
            key = (header['start'], header['end'])
            if key in headers_seen:
                continue
            headers_seen.add(key)
            new_selection = Text(
                content=header['text'],
                meta=dict(
                    start=header['start'],
                    end=header['end'],
                    is_table_header=True,
                    section=selection['meta']['section'],
                ),
            )
            new_selections.append(new_selection)

    if not new_selections:
        return evidence_selections
    return evidence_selections + new_selections


def merge_overlapping_selections(evidence_selections: list[Text]):
    merged = []

    for selection in sorted(evidence_selections, key=lambda s: s['meta']['start']):
        if not merged:
            merged.append(selection)
            continue

        last_selection = merged[-1]

        if selection['meta']['start'] > last_selection['meta']['end']:
            # do not overlap
            merged.append(selection)
            continue
        elif selection['meta']['start'] == last_selection['meta']['end']:
            # back-to-back
            text = last_selection['content'] + selection['content']
            last_selection['meta']['end'] = selection['meta']['end']
            merged_selection = Text(text, meta=last_selection['meta'], type=last_selection.type)
            merged[-1] = merged_selection
            assert len(text) == merged_selection['meta']['end'] - merged_selection['meta']['start']
        else:
            # can skip if completely encased in other selection
            if selection['meta']['end'] > last_selection['meta']['end']:
                relative_start = int(selection['meta']['end'] - last_selection['meta']['start'])
                text = last_selection['content'] + selection['content'][relative_start:]
                merged_selection = Text(text, meta=last_selection['meta'], type=last_selection.type)
                merged[-1] = merged_selection
                assert (
                    len(text) == merged_selection['meta']['end'] - merged_selection['meta']['start']
                )

        if 'id' in last_selection['meta']:
            last_selection['meta']['id'] += '_' + selection['meta']['id']
        # grab any metadata that isn't already in a selection
        for meta_field, meta_value in selection['meta'].items():
            if meta_field not in last_selection['meta']:
                last_selection['meta'][meta_field] = meta_value

    return merged


def load_claim_from_hypothesis_conversation(conv_element):
    claim = json.loads(conv_element['comment'])['claim'][0]
    claim['type'] = 'claim'
    claim_content = {}
    for col, val in claim['content']:
        if col == 'diseases':
            col = 'disease'
        assert col in CIVIC_CLAIM_COL_ORDER
        claim_content[col] = val
    del claim['content']

    claim_content.setdefault('variantOrigin', 'SOMATIC')

    return Text(content=claim_content, **claim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
        help='Path to the JSONL file with the hypothesis civic annotations pre-matched to their article xml content',
    )
    parser.add_argument('output', help='path to the output JSONL file to write the NLI data to')
    parser.add_argument('--civic_dump', required=True)
    parser.add_argument(
        '--neutral',
        default=False,
        action='store_true',
        help='Input is the neutral examples and therefore will not match CIViC EIDs and should not be supplemented with generated examples',
    )
    args = parser.parse_args()

    hyp_cols = [
        'id',
        'username',
        'comment',
        'updated',
        'uri',
        'tag',
        'not_in_main_text',
        'requires_supplement',
        'requires_image',
    ]
    df = read_jsonl(args.input, required_columns=hyp_cols + ['text']).rename(
        columns={f: f'hypothesis.{f}' for f in hyp_cols}
    )
    print('EID4879 rows', df[df['civic.id'] == '4879'].shape[0])

    if df.shape[0] == 0:
        # if the input file is empty, re-output an empty file
        dump_jsonl(df, args.output)
        return
    df['claim.id'] = df['claim.id'].astype(str).fillna('')
    df['pmid'] = df.pmid.apply(lambda id: INVALID_PUBMED_IDS.get(id, id))

    # merge with civic to check EIDs from tags are real EIDs
    civic_df = load_dump(args.civic_dump)
    civic_df['id'] = civic_df.id.astype(str)
    civic_df['matched'] = True

    df = df.merge(
        civic_df.rename(columns={c: f'civic.{c}' for c in civic_df.columns}).rename(
            columns={'civic.id': 'claim.id'}
        ),
        how='left',
    )
    assert 'events' in civic_df.columns

    errors_df = None

    if not args.neutral:
        errors_df = df[df['claim.id'] == ''].copy()
        logging.warning(
            f'There are {errors_df.shape[0]} annotations that do not contain a civic evidence id and will be dropped'
        )
        df = df[df['claim.id'] != '']

        df['civic.matched'] = df['civic.matched'].fillna(False)
        civic_match_failure = df[df['civic.matched'] == False]  # noqa
        errors_df = pd.concat([errors_df, civic_match_failure])
        print(errors_df)
        if civic_match_failure['claim.id'].nunique() > 50:
            print(civic_match_failure['claim.id'].unique())
            print(errors_df)
            raise AssertionError(
                f'something weird is going on if there are this many failures: {civic_match_failure.shape[0]} of {df.shape[0]}'
            )
        logging.warning(
            f'There are {civic_match_failure.shape[0]} annotations that will be ignored as their corresponding civic entry was not found: {civic_match_failure["claim.id"].unique()}'
        )
        df = df[df['civic.matched'] != False]  # noqa
        print('EID4879 rows', df[df['civic.id'] == '4879'].shape[0])

    # filter out "statement" examples
    logging.info(
        f'filtering out {df[df.example == "statement"].shape[0]} rows with example type of statement (only used in supporting civic curation for editors)'
    )
    df = df[df.example != 'statement']
    print('EID4879 rows', df[df['civic.id'] == '4879'].shape[0])

    df['example_id'] = (
        'EID'
        + df['claim.id']
        + df['civic.revision_state'].fillna('')
        + '_ex:'
        + df.example.astype(str)
        + '_'
        + df['hypothesis.username']
        + '_'
        + df['hypothesis.label']
    )

    def count_non_empty(series):
        return sum([1 if s not in {'', None} else 0 for s in series])

    # LOG ERRORS to a separate file

    match_errors_df = (
        df.groupby(['example_id'])
        .agg({'text.match_error': count_non_empty, 'text.query': 'count'})
        .reset_index()
    )
    match_errors_df['match_rate'] = (
        match_errors_df['text.query'] - match_errors_df['text.match_error']
    )
    match_errors_df['match_rate'] = match_errors_df['match_rate'] / match_errors_df['text.query']
    match_errors_df['total_expected_matches'] = match_errors_df['text.query']
    df = df.merge(
        match_errors_df[['example_id', 'match_rate', 'total_expected_matches']], how='left'
    )
    partials = df[(df.match_rate > 0) & (df.match_rate < 1)]
    fully_missed = df[df.match_rate == 0]

    errors_df = (
        pd.concat([partials, fully_missed])
        if errors_df is None
        else pd.concat([errors_df, partials, fully_missed])
    )
    logging.warning(
        f'There are {partials.example_id.nunique()} annotation examples on {partials["claim.id"].nunique()} civic evidence items which were only partially matched to source text'
    )
    logging.warning(
        f'There are {fully_missed.example_id.nunique()} annotation examples on {fully_missed["claim.id"].nunique()} civic evidence items which could not be matched to source text at all'
    )

    errors_df = errors_df.rename(
        columns={
            'example_id': 'id',
            'effective_text': 'evidence',
            'effective_label': LABEL_COL,
            'doi': 'document.doi',
            'pmcid': 'document.pmcid',
            'pmid': 'document.pmid',
            'license': 'document.license',
            'uri': 'document.uri',
            'comment': 'hypothesis.comment',
            'conversation': 'hypothesis.conversation',
        }
    )
    dump_jsonl(errors_df, args.output.replace('.jsonl', '.errors.jsonl'))

    df = df[df.match_rate == 1]
    logging.info(
        f'There are {df.example_id.nunique()} annotation examples for {df["claim.id"].nunique()} civic entries on {df.pmid.nunique()} publications which were successfully matched to source text'
    )

    df[LABEL_COL] = df['hypothesis.label'].fillna('')
    df.loc[
        (df['hypothesis.label'] == 'default')
        & (df['civic.evidenceDirection'] == 'DOES_NOT_SUPPORT'),
        LABEL_COL,
    ] = 'REFUTES'
    df[LABEL_COL] = df[LABEL_COL].str.replace('default', 'SUPPORTS', regex=False)
    assert df[df['text.extracted'] == ''].shape[0] == 0
    df['effective_text'] = df['text.extracted'].apply(lambda t: Text(t))

    def add_metadata(row):
        score = None
        comment = ''
        if row['hypothesis.comment'].startswith('{'):
            json_comment = json.loads(row['hypothesis.comment'])
            score = json_comment['meta']['score']
        else:
            comment = row['hypothesis.comment']

        if pd.isnull(row['text.start']) or pd.isnull(row['text.end']):
            logging.error(str(dict(row)))
            raise ValueError(
                'Cannot build example with section that failed to match reference text'
            )
        row['effective_text'].meta.update(
            {
                'start': row['text.start'],
                'end': row['text.end'],
                'hypothesis.id': row['hypothesis.id'],
                'hypothesis.comment': comment,
                'section': row['text.section'],
                'score': score,
                'table_header': row['text.table_header'],
            }
        )

    df.apply(add_metadata, axis=1)

    if not args.neutral:
        df['claim.source'] = 'civic'
        df['claim.generated'] = False
        df['evidence.selection_method'] = 'hypothesis'
    else:
        df['claim.source'] = NEUTRAL_WORKFLOW_NAME
        df['claim.generated'] = True
        df['evidence.selection_method'] = NEUTRAL_WORKFLOW_NAME

    df = df.sort_values(['pmid', 'claim.id', 'text.start'])

    df['hypothesis.tag_id'] = df['hypothesis.tag_id'].apply(
        lambda tag_id: tuple(tag_id)
    )  # json turns this into a non-hashable list on dump/load

    df = df.drop_duplicates(
        [
            'pmid',
            'claim.id',
            'example_id',
            LABEL_COL,
            'effective_text',
            'hypothesis.comment',
            'hypothesis.tag_id',
            'document.title',
            'pmcid',
            'doi',
            'license',
            'evidence.source',
            'hypothesis.uri',
            'in_table',
            'civic.revision_state',
            'civic.id',
        ]
    )

    agg_states = {
        **{
            col: 'first'
            for col in df.columns
            if col.startswith('hypothesis.')
            if col not in ['hypothesis.tag_id']
        },
        **{
            c: 'first'
            for c in df.columns
            if c.startswith('civic.') and c not in {'civic.revision_state', 'civic.id'}
        },
    }
    agg_states.update(
        {
            'claim.generated': 'any',
            'conversation': lambda convs: list(itertools.chain(*convs)),
            'document.title': 'first',
            'doi': 'first',
            'evidence.selection_method': 'first',
            'evidence.source': 'first',
            'in_table': 'any',
            'license': 'first',
            'pmcid': 'first',
            'effective_text': list,
        }
    )

    df = (
        df.groupby(
            [
                'pmid',
                'claim.id',
                'claim.source',
                'hypothesis.tag_id',
                'civic.revision_state',
                'civic.id',
                LABEL_COL,
            ],
            dropna=False,
        )
        .agg(agg_states)
        .reset_index()
        .rename(
            columns={
                'example_id': 'id',
                'effective_text': 'evidence',
                'doi': 'document.doi',
                'pmcid': 'document.pmcid',
                'pmid': 'document.pmid',
                'license': 'document.license',
                'uri': 'document.uri',
            }
        )
    )

    if not args.neutral:
        df['claim'] = df.apply(lambda row: build_civic_claim(row), axis=1)
    else:
        df['claim'] = df.conversation.str[0].apply(load_claim_from_hypothesis_conversation)
        # now remove the first element from the conversation as well as any duplicates
        df['conversation'] = df.conversation.apply(
            lambda conv: [c for c in conv if c['comment'] != conv[0]['comment']]
        )

    df = df[
        [
            'claim.generated',
            'claim.id',
            'claim.source',
            'claim',
            'document.doi',
            'document.pmcid',
            'document.pmid',
            'document.title',
            'document.license',
            'evidence.source',
            'evidence.selection_method',
            'evidence',
            LABEL_COL,
            'hypothesis.tag',
            'hypothesis.username',
            'hypothesis.label',
            'hypothesis.tag_id',
            'hypothesis.not_in_main_text',
            'hypothesis.requires_supplement',
            'hypothesis.requires_image',
            'in_table',
            'conversation',
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
    ]
    df['evidence.generated'] = args.neutral
    df['evidence.id'] = df['evidence'].apply(
        lambda ev_list: '_'.join([e.meta['hypothesis.id'] for e in ev_list])
    )
    df['generated'] = df['claim.generated'] | df['evidence.generated']

    def has_flag(conversation):
        for comment in conversation:
            if any(['fix' in tag for tag in comment['tags']]):
                return True
        return False

    df['flagged'] = df.conversation.apply(has_flag)
    df[LABEL_COL] = df[LABEL_COL].fillna('')
    df['claim.flat'] = df.claim.apply(concat_text)

    df['evidence'] = df['evidence'].apply(include_table_header)
    df['evidence'] = df['evidence'].apply(merge_overlapping_selections)

    def concat_ev(row):
        try:
            return concat_text(row.evidence)
        except AttributeError as err:
            logging.error(row)
            logging.error(row.evidence)
            raise err

    df['evidence.flat'] = df.apply(concat_ev, axis=1)
    df['id'] = df.apply(lambda _: uuid.uuid4())

    # shuffle all the rows
    df = df.sample(frac=1).reset_index(drop=True).rename({'civic.corrections': 'corrections'})
    dump_jsonl(df, args.output)


if __name__ == '__main__':
    main()
