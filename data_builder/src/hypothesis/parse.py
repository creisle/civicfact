"""
Verify the annotation in our biocXML version of the article. Check the positions are consistent
"""

import argparse
import itertools
import json
import logging
import re

import pandas as pd
from bioconverters.utils import cleanup_text

from article_utils.align import strip_citations
from article_utils.db import connect, load_metadata_from_api
from article_utils.util import INVALID_PUBMED_IDS, dump_jsonl
from civic.util import load_dump
from nli.util import NEUTRAL_LABEL


def get_pmcid_from_url(url) -> str:
    for part in url.split('/'):
        if part and re.match(r'^PMC\d+$', part):
            return part
    return ''


def get_pubmed_from_url(url) -> str:
    m = re.match(r'https?://pubmed.ncbi.nlm.nih.gov/(\d+)/?$', url)
    if m:
        return m.group(1)
    return ''


def parse_example(tag_name):
    match = re.match(r'.*(\b|-)ex:?(\d+)(\b|-).*', tag_name, re.IGNORECASE)
    if match:
        return match[2]
    if 'statement' in tag_name.lower() or 'statment' in tag_name.lower():
        return 'statement'
    return '1'


def parse_evidence_id(tag_name):
    match = re.match(r'.*(\b|-)eid:?(\d+)(\b|-).*', tag_name, re.IGNORECASE)
    if match:
        return match[2]
    return None


def parse_status_label(tag_name, default_label='default'):
    m = re.search(r'(\b|-)(status:)?nei', tag_name, re.IGNORECASE)
    if m:
        return NEUTRAL_LABEL
    return default_label


def parse_selected_text(target):
    try:
        pos = None
        text = None

        for selector in target.get('selector', []):
            if selector['type'] == 'TextQuoteSelector':
                text = selector['exact']
            elif selector['type'] == 'TextPositionSelector':
                pos = selector['start']
        if text:
            text = clean_selection_artifacts(text.strip())
            text = strip_citations(text)
            text = text.replace('±', '+-')  # TODO: keep in xml
            text = cleanup_text(text)
        return pos, text or None
    except Exception as err:
        logging.error(f'failed to parse select text from target [{len(target)}]: {target}')
        raise err


def parse_container(target):
    for selector in target.get('selector', []):
        if selector['type'] == 'RangeSelector':
            return selector['startContainer'], selector['endContainer']
    return '', ''


def parse_in_table(target):
    in_table = False
    for selector in target.get('selector', []):
        if selector['type'] == 'RangeSelector':
            html_start = re.sub(r'\[\d+\]', '', selector['startContainer']).split('/')
            html_end = re.sub(r'\[\d+\]', '', selector['endContainer']).split('/')
            in_table = in_table or bool('td' in html_start and 'td' in html_end)
    return in_table


def parse_comments_by_tags(text) -> list[list[str]]:
    """
    If the start of a line in the comment describes a specific tag or set of tags it should be applied to then
    this is parsed to split these comments by the tags otherwise the comment is added to all tags
    """
    comment_text = text or ''
    split_comments: dict[str, list[str]] = {NEUTRAL_LABEL: []}

    for line in comment_text.split('\n'):
        if not line.strip():
            continue
        match = re.match(
            r'^(eid:?(?P<eid>\d+))?-?(status:)?NEI(:| - )(?P<comment>.*)',
            line.strip(),
            re.IGNORECASE,
        )
        if not match:
            continue

        if match.group('eid'):
            split_comments.setdefault(match.group('eid'), []).append(match.group('comment'))
        else:
            # generic
            split_comments[NEUTRAL_LABEL].append(match.group('comment'))

    result = []

    for tag, comments in split_comments.items():
        for comment in comments:
            result.append([tag, comment.strip()])

    if not result:
        result.append(['any', comment_text.strip()])

    # drop any empty comments
    result = [(t, c) for t, c in result if c]

    return result


def extract_tag_specific_comments(df):
    df = df[['comment', 'id']].copy()
    df['tag_comments'] = df.comment.apply(parse_comments_by_tags)
    df = df.explode('tag_comments')
    df['tag_comment_filter'] = df['tag_comments'].str[0]
    df['tag_comment'] = df['tag_comments'].str[1]
    df = df[['id', 'tag_comment', 'tag_comment_filter']].drop_duplicates()
    df = df.fillna('')
    df = df[(df.tag_comment != '') & (df.tag_comment_filter != '')]
    return df


def clean_selection_artifacts(text):
    # remove json inserts on links in PMC
    text = re.sub(r'{\s*"type":\s*"[^"]+",\s*"attrs":\s*{[^}]+}\s*}', '', text)
    # remove duplicate table/fig references
    text = re.sub(r'(Table|Figure)\s\u200b\(?\1(\d+[A-Z]*);\2;', r'\1 \2;', text)
    # remove any links since we parse these out in the source text
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    return text


def parse_supplemental_requirements_tags(conversation):
    supplement_req = None
    image_req = None
    not_in_main_text = None
    for comment in conversation:
        for tag in comment['tags']:
            if 'requires' in tag.lower():
                if tag == 'requires-supplement':
                    supplement_req = True
                elif tag == 'requires-image':
                    image_req = True
                else:
                    raise NotImplementedError('unrecognized requirements tag')
            elif tag.lower().strip() in {'not_in_main_text', 'not_in_manuscript'}:
                not_in_main_text = True

    return supplement_req, image_req, not_in_main_text


def supplement_metadata(df, db_file: str):
    df = df.copy()
    db_conn = connect(db_file)

    try:
        meta_df = pd.read_sql_query(
            'SELECT * FROM article_metadata',
            db_conn,
            dtype={'pmid': str, 'pmcid': str, 'doi': str},
        ).rename(columns={'id': 'db_id'})
        pubmed_df = (
            df[df.pmid != '']
            .drop(columns=['pmcid'])
            .merge(meta_df[meta_df.pmid != ''], on=['pmid'], how='left')
        )
        pmc_df = (
            df[df.pmid == '']
            .drop(columns=['pmid'])
            .merge(meta_df[meta_df.pmcid != ''], on=['pmcid'], how='left')
        )
        # for any PMC/pubmed ids that are not found/matched
        missing_pubmed_ids = set(
            pubmed_df[pubmed_df.db_id.isnull()].pmid.tolist()
            + pmc_df[pmc_df.db_id.isnull()].pmcid.tolist()
        ) - {''}
        if missing_pubmed_ids:
            load_metadata_from_api(db_conn, missing_pubmed_ids)
            meta_df = pd.read_sql_query(
                'SELECT * FROM article_metadata',
                db_conn,
                dtype={'pmid': str, 'pmcid': str, 'doi': str},
            ).rename(columns={'id': 'db_id'})

        return pd.concat(
            [
                df[df.pmid != '']
                .drop(columns=['pmcid'])
                .merge(meta_df[meta_df.pmid != ''], on=['pmid'], how='left'),
                df[df.pmid == '']
                .drop(columns=['pmid'])
                .merge(meta_df[meta_df.pmcid != ''], on=['pmcid'], how='left'),
            ]
        )
    finally:
        db_conn.close()


def find_fix_flagged_annotations(raw_annotations) -> dict[str, str]:
    annotations_by_id = {}
    fix_flagged = []
    for annotation in raw_annotations:
        annotations_by_id[annotation['id']] = annotation
        tags = {t.lower().strip() for t in annotation['tags']}
        if "fix" in tags:
            fix_flagged.append((annotation['id'], annotation['text']))

    result = {}
    while fix_flagged:
        current_id, comment = fix_flagged.pop(0)
        if current_id in result or current_id not in annotations_by_id:
            continue
        current = annotations_by_id.get(current_id)
        result[current_id] = comment
        for ref in current.get('references', []):
            fix_flagged.append((ref, comment))

    return result


def nest_annotation_replies(df):
    df = df.copy()
    df['parent_id'] = df.references.apply(lambda refs: '' if not refs else refs[0])

    ref_df = df.copy()
    ref_df = (
        ref_df[['id', 'parent_id', 'created', 'user', 'comment', 'tag']]
        .groupby(['id', 'parent_id', 'created', 'user', 'comment'])
        .agg({'tag': list})
        .reset_index()
        .rename(columns={'tag': 'tags'})
    )

    def cleanup_tags(tags):
        return [t.strip() for t in tags if t.strip()]

    ref_df['tags'] = ref_df.tags.apply(cleanup_tags)

    conversations = {}

    for _, row in ref_df.iterrows():
        parent_id = row.parent_id
        row = dict(row)
        del row['parent_id']
        if parent_id:
            conversations.setdefault(parent_id, []).append(row)
        else:
            conversations.setdefault(row['id'], []).append(row)

    df['conversation'] = df['id'].apply(lambda id: conversations.get(id, []))
    df['conversation'] = df['conversation'].apply(
        lambda conv: sorted(conv, key=lambda comment: comment['created'])
    )

    # now remove the nested replies
    df = df[df['parent_id'] == ''].copy().drop(columns={'parent_id'})

    return df


def collect_conversation(conversations):
    conv_by_id = {}
    for conversation in conversations:
        for comment in conversation:
            conv_by_id[comment['id']] = comment

    return sorted(conv_by_id.values(), key=lambda x: x['created'])


def fix_nli_labels(label):
    if label in {
        'NEI',
        'NOT ENOUGH INFO',
        'neutral',
        NEUTRAL_LABEL,
    }:
        return NEUTRAL_LABEL
    elif label == 'entailment':
        return 'SUPPORTS'
    elif label in {'REFUTES', 'contradiction'}:
        return 'REFUTES'
    return label


def apply_overrides(row) -> str:
    status = row['label']
    for comment in row.conversation:
        for tag in comment['tags']:
            m = re.search(r'overrides?[ -]status:(.*)', tag)
            if m:
                override_status_string = m.group(1)
                override_status = fix_nli_labels(override_status_string)
                logging.debug(
                    f'applying override on status from {status} to {override_status} using {override_status_string} for example {row.id}'
                )
                status = override_status
    return status


def copy_comments_to_other_tag_segments(df):
    """
    When a conversation reply is made a selection in hypothesis, it should be copied to appear as a reply
    on any segments with the same tag. This way things like "overrides" or "fix" flags apply to
    the entire example at once and not just the segment that was flagged
    """
    df['hypothesis.tag_id'] = df.apply(
        lambda row: (row['username'], row['claim.id'], row['example'], row['label']), axis=1
    )

    conv_df = (
        df.groupby(['hypothesis.tag_id']).agg({'conversation': collect_conversation}).reset_index()
    )
    print(conv_df)
    df = df.drop(columns=['conversation']).merge(conv_df, how='left')
    return df


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'input',
        help='path to the JSON file downloaded from hypothesis',
    )
    parser.add_argument(
        '--article_db_file',
        help='path to the JSON file downloaded from hypothesis',
        default='data/articles_db.sqlite',
    )
    parser.add_argument(
        '--civic_dump', required=True, help='path to the civic JSONL dump of evidence IDs'
    )
    parser.add_argument(
        'output',
        help='path to the output jsonl hypothesis file split by tags and examples',
    )
    parser.add_argument('--min_user_selections', type=int)
    parser.add_argument('--neutral', action='store_true', default=False)
    args = parser.parse_args()

    logging.info(f'reading: {args.input}')
    df = pd.read_json(args.input, lines=True, dtype=False)
    if df.shape[0] == 0:
        # just re-output empty files
        dump_jsonl(df, args.output)
        return

    data = []
    with open(args.input, 'r') as fh:
        for line in fh.readlines():
            row = json.loads(line)
            data.append(row)
            row.setdefault('references', [])
            # stop normalize from dropping anything w/o a tag
            if not row['tags']:
                row['tags'].append('')

    meta_columns = [
        'id',
        'group',
        'created',
        'updated',
        'user',
        'uri',
        'text',
        'target',
        ['document', 'title'],
        ['links', 'html'],
        ['links', 'incontext'],
    ]
    # for panda bug https://github.com/pandas-dev/pandas/issues/37782#issuecomment-1146338699
    no_refs = all([not row['references'] for row in data])
    if not no_refs:
        meta_columns.append('references')

    df = pd.json_normalize(
        data,
        record_path='tags',
        meta=meta_columns,
        errors='raise',
    ).rename(columns={'text': 'comment', 0: 'tag'})
    df['document.title'] = df['document.title'].str[0]  # gets parsed nested in a list

    if no_refs:
        df['references'] = df.tag.apply(lambda _: [])

    df['username'] = df.user.str.replace('acct:', '', regex=False).str.replace(
        '@hypothes.is', '', regex=False
    )
    df = nest_annotation_replies(df)

    if args.neutral:
        df = df[df.tag.str.contains('_ex:')]
        df['claim.id'] = df.tag.str.split('_ex:').str[0]
        df['example'] = df.tag.str.split('_ex:').str[1]
        df['label'] = df.tag.apply(lambda tag: parse_status_label(tag, NEUTRAL_LABEL))
    else:
        df['example'] = df.tag.apply(parse_example)
        df['claim.id'] = df.tag.apply(parse_evidence_id)
        df['label'] = df.tag.apply(parse_status_label)

    assert (df.target.str.len() == 1).all()
    df['target'] = df.target.str[0]
    df['text_start'], df['text'] = zip(*df.target.apply(parse_selected_text))
    df['in_table'] = df.target.apply(parse_in_table)
    df['html_start'], df['html_end'] = zip(*df.target.apply(parse_container))
    df['pmcid'] = df.uri.apply(get_pmcid_from_url)
    df['pmid'] = df.uri.apply(get_pubmed_from_url).apply(
        lambda pmid: INVALID_PUBMED_IDS.get(pmid, pmid)
    )

    # now try to fill-in any PMID/PMCIDs missing based on the civic ID (if highlighted not in PMC/PubMed)
    if not args.neutral:
        civic_df = load_dump(args.civic_dump)
        civic_df = civic_df[civic_df.ground_truth != False]
        civic_df = civic_df[civic_df['pmid'] != '']
        civic_df = civic_df[['pmid', 'id']].rename(columns={'id': 'claim.id', 'pmid': 'civic.pmid'})
        civic_df['claim.id'] = civic_df['claim.id'].astype(str)

        df = df.merge(civic_df[['claim.id', 'civic.pmid']], how='left')
        df.loc[(df.pmid == '') & (df['civic.pmid'] != ''), 'pmid'] = df['civic.pmid']
        df['pmid'] = df.pmid.fillna('')
        df['civic.id'] = df['claim.id']
    else:
        df['civic.id'] = ''

    df = supplement_metadata(df, args.article_db_file).fillna('')

    tag_comments = extract_tag_specific_comments(df)
    df = df.merge(tag_comments, how='left', on=['id'])
    df = df.fillna('')
    # filter impossible tag filter combinations
    df['to_remove'] = False
    df.loc[(df.label != NEUTRAL_LABEL) & (df.tag_comment_filter == NEUTRAL_LABEL), 'to_remove'] = (
        True
    )
    df.loc[
        (~df.tag_comment_filter.isin({NEUTRAL_LABEL, 'any', ''}))
        & (df.tag_comment_filter != df['claim.id']),
        'to_remove',
    ] = True

    df = df[~df.to_remove].rename(columns={'comment': 'original_comment', 'tag_comment': 'comment'})
    df = df[df['claim.id'] != '']

    if args.min_user_selections:
        # filter out user selections below a minimum threshold
        user_counts = df.groupby(['username']).agg({'id': 'nunique'}).reset_index()
        dropped_users = (
            user_counts[user_counts.id < args.min_user_selections].username.unique().tolist()
        )
        logging.warning(
            f'dropping selections by users: {dropped_users} as they did not meet the minimum number of selections thresold ({args.min_user_selections})'
        )
        user_counts = user_counts[user_counts.id >= args.min_user_selections]
        df = df.merge(user_counts[['username']])

    df = copy_comments_to_other_tag_segments(df)
    df['hypothesis.label'] = df.apply(apply_overrides, axis=1)
    df['_supp'] = df.conversation.apply(parse_supplemental_requirements_tags)
    df['requires_supplement'] = df._supp.str[0]
    df['requires_image'] = df._supp.str[1]
    df['not_in_main_text'] = df._supp.str[2]
    df['not_in_main_text'] = (
        df['not_in_main_text'].fillna(0)
        + df.requires_supplement.fillna(0)
        + df.requires_image.fillna(0)
    ) > 0

    df = df.drop(columns=['_supp'])
    dump_jsonl(
        df[
            [
                'id',
                'uri',
                'document.title',
                'username',
                'created',
                'updated',
                'comment',
                'text',
                'text_start',
                'tag',
                'example',
                'claim.id',
                'pmid',
                'pmcid',
                'doi',
                'license',
                'hypothesis.label',
                'hypothesis.tag_id',
                'in_table',
                'html_start',
                'html_end',
                'conversation',
                'requires_supplement',
                'requires_image',
                'not_in_main_text',
                'civic.id',
            ]
        ],
        args.output,
    )


if __name__ == '__main__':
    main()
