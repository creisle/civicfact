"""
Given pending revisions, if done by CIViC editor, assume correct. If not, ignore for now. This will
change the expected status of the claim/evidence pair

Also, use the revision history to generate error claim/evidence pairs
"""

import argparse
import itertools
import logging
import re

import pandas as pd
import requests_cache

from article_utils.util import dump_jsonl, normalize_and_expand_column, read_jsonl

from .util import get_item_by_id

all_rev_fields = set()

CANCER_DISEASE_ID = 216

CLAIM_REVISION_FIELDS = {
    'significance',
    'molecular_profile_id',
    'evidence_type',
    'evidence_direction',
    'therapy_ids',
    'therapy_interaction_type',
    'phenotype_ids',
    'disease_id',
    'variant_origin',
}

REV_FIELD_MAPPING = {
    'disease_id': 'disease',
    'phenotype_ids': 'phenotypes',
    'therapy_ids': 'therapies',
    'molecular_profile_id': 'molecularProfile',
    'evidence_direction': 'evidenceDirection',
    'therapy_interaction_type': 'therapyInteractionType',
    'evidence_type': 'evidenceType',
    'significance': 'significance',
    'variant_origin': 'variantOrigin',
}

assert all([c in REV_FIELD_MAPPING for c in CLAIM_REVISION_FIELDS])


def normalize_evidence_direction(evidence_direction):
    if pd.isnull(evidence_direction):
        return 'SUPPORTS'
    if re.match(r'^does(_|\s)not(_|\s)support$', evidence_direction, re.IGNORECASE):
        return 'DOES_NOT_SUPPORT'
    elif re.match(r'^supports$', evidence_direction, re.IGNORECASE):
        return 'SUPPORTS'
    elif re.match(r'^n/?a$', evidence_direction, re.IGNORECASE):
        return 'N/A'
    raise NotImplementedError(
        f'no normalization procedure for unexpected evidence direction: {evidence_direction}'
    )


def normalize_significance(sig):
    if pd.isnull(sig):
        return sig
    if re.match(r'^n/?a$', sig.strip(), re.IGNORECASE):
        return 'N/A'
    if sig == 'SENSITIVITYRESPONSE':
        sig = 'SENSITIVITY/RESPONSE'

    sig = sig.replace('_', ' ').lower()
    return sig


def normalize_variant_origin(term):
    if pd.isnull(term):
        return None
    if re.match(r'^n/?a$', term, re.IGNORECASE):
        return 'N/A'
    if term.lower() in {'germline mutation'}:
        return None
    if term.lower() == 'somatic mutation':
        term = 'SOMATIC'
    term = term.replace(' ', '_').upper()

    return term


def nullable_string(value):
    assert isinstance(value, str) or pd.isnull(value)
    return value


def filter_relevant_revisions(row):
    revisions = row['revisions.nodes']
    # only keep the last change to any given field
    filtered_revs = []
    field_seen = set()
    for rev in sorted(revisions, key=lambda r: r['creationActivity']['createdAt'], reverse=True):
        if rev['fieldName'] in field_seen:
            continue
        filtered_revs.append(rev)
        field_seen.add(rev['fieldName'])

    revisions = filtered_revs
    filtered_revs = []
    for rev in revisions:
        if rev['fieldName'] not in CLAIM_REVISION_FIELDS:
            continue

        # Due to ongoing discussions regarding best-practices for cell line evidence disease field represetation we need to temporarily exclude
        # these revisions until the guidelines are formalized
        # if rev['fieldName'] == 'disease_id' and row['evidenceLevel'] == 'D' and rev['suggestedValue'] != CANCER_DISEASE_ID:
        #     logging.warning(f'skipping controversial revision on preclinical disease field')
        #     continue

        if rev['linkoutData']['diffValue']['__typename'] == 'ObjectFieldDiff':
            valid_objects = True
            # check all objects have a valid name
            for obj in (
                rev['linkoutData']['diffValue']['suggestedObjects']
                + rev['linkoutData']['diffValue']['removedObjects']
            ):
                if pd.isnull(obj['displayName']):
                    valid_objects = False
                elif obj['displayName'] == 'Childhood B-cell Acute Lymphoblastic Leukemia':
                    valid_objects = False
            if not valid_objects:
                # usually this means something super super old that no longer exists in the DB
                continue
        elif rev['status'] != 'NEW':
            if pd.isnull(rev['currentValue']) or pd.isnull(rev['suggestedValue']):
                continue
        elif pd.isnull(rev['suggestedValue']):
            continue

        if rev['fieldName'] == 'significance' and rev['status'] != 'NEW':
            if {'NA', 'N/A', 'Uncertain Significance'} & {
                rev['currentValue'],
                rev['suggestedValue'],
            } or pd.isnull(rev['currentValue']):
                continue
        if rev['fieldName'] in {'evidence_type', 'evidenceDirection'}:
            rev['currentValue'] = rev['currentValue'].upper().replace(' ', '_')
            rev['suggestedValue'] = rev['suggestedValue'].upper().replace(' ', '_')

        # ignore CIViC_bot rejected suggestions (could be automated nonsense suggestions that don't reflect user errors properly)
        if rev['creationActivity']['user']['displayName'] == 'CIViC_Bot':
            if rev['status'] == 'REJECTED':
                continue
            if not all(
                [
                    rev['status'] == 'NEW',
                    pd.isnull(rev['currentValue']) or rev['currentValue'] in {'N/A', 'NA'},
                    rev['fieldName'] in {'evidence_direction', 'significance'},
                ]
            ):
                continue
        elif not rev['creationActivity']['note']:
            continue

        filtered_revs.append(rev)

    return sorted(filtered_revs, key=lambda r: r['creationActivity']['createdAt'])


def resolve_revision_links(rev_df):

    objects_by_type_by_id = {}

    # only need to worry about objects
    rev_df = rev_df[rev_df['rev.linkoutData.diffValue.__typename'] == 'ObjectFieldDiff']

    for _, row in rev_df.drop_duplicates('id').iterrows():
        for field in ['disease', 'molecularProfile']:
            if not pd.isnull(row[field]):
                objects_by_type_by_id.setdefault(field, {})[row[field]['id']] = row[field]

        for field in ['therapies', 'phenotypes']:
            if isinstance(row[field], list):
                for therapy in row[field]:
                    try:
                        objects_by_type_by_id.setdefault(field, {})[therapy['id']] = therapy
                    except KeyError:
                        logging.warning(
                            f'for some reson this object ({field}) with name ({therapy.get("name", "")}) is missing an ID for EID:{row["id"]}'
                        )

    # now that we have the objects that were automatcically retrieved grab any that still need retrieval

    field_to_querytype = {
        'disease_id': 'disease',
        'phenotype_ids': 'phenotype',
        'therapy_ids': 'therapy',
        'molecular_profile_id': 'molecularProfile',
    }
    for _, row in rev_df.iterrows():
        field_type = REV_FIELD_MAPPING[row['rev.fieldName']]

        for field in [
            'rev.linkoutData.diffValue.removedObjects',
            'rev.linkoutData.diffValue.suggestedObjects',
        ]:
            for i, obj in enumerate(row[field]):
                if obj['id'] in objects_by_type_by_id[field_type]:
                    row[field][i] = objects_by_type_by_id[field_type][obj['id']]
                else:
                    item = get_item_by_id(obj['id'], field_to_querytype[row['rev.fieldName']])
                    objects_by_type_by_id[field_type][obj['id']] = item
                    row[field][i] = item


def splittable_mp(row, ground_truth):
    """
    For revisions where the difference is in a list of variants with OR use the revision comment
    to determine if these changes are necessiary or just convenient to avoid making/having a separate EID
    """
    ground_vars = {v.strip() for v in ground_truth['molecularProfile']['name'].split(' OR ')}
    curr_vars = {v.strip() for v in row['molecularProfile']['name'].split(' OR ')}

    if len(ground_vars - curr_vars) == 0 or len(curr_vars - ground_vars) == 0:
        for phrase in [
            r'not report(ed)?.* separately',
            'did not conduct any analysis into the combined subgroup',
            'not separated in paper',
            r'(did|do) not report results .*(separately|for.*excluding)',
            'do not distinguish individual',
            'outcomes were tested as KRAS mut vs wt not KRAS G12 vs wt',
            'analysis was done on patients with either mutation or amplification',
            r'was not analyzed.*independently',
            'no results were reported for individual',
            'The only evidence for drug response was experiments conducted in KM12 cells which have the TPM3-NTRK1 fusion',
        ]:
            if not pd.isnull(row['revision_note']) and re.search(phrase, row['revision_note']):
                return False
    return True


def generate_corrections(row, ground_truth):
    diff = {}

    def normalize(value):
        return value.lower().replace('_', ' ').replace('/', '')

    for field in ['molecularProfile', 'disease']:
        if isinstance(row[field], dict) and isinstance(ground_truth[field], dict):
            if row[field]['name'] != ground_truth[field]['name']:
                if field == 'molecularProfile':
                    if set(row[field]['name'].split(' ')) ^ set(
                        ground_truth[field]['name'].split(' ')
                    ) == {'Fusion', 'AND'}:
                        # skip any revisions that are just changing the MP from old to new notation since these are now largely correct
                        continue
                    if splittable_mp(row, ground_truth):
                        # skip any revisions that are valid both as combined or individual EIDs
                        continue

                diff[field] = [row[field]['name'], ground_truth[field]['name']]
        elif isinstance(row[field], dict):
            diff[field] = [row[field]['name'], ground_truth[field]]
        elif isinstance(ground_truth[field], dict):
            diff[field] = [row[field], ground_truth[field]['name']]

    for field in ['phenotypes', 'therapies']:
        current = {item['name'] for item in row[field]}
        truth = {item['name'] for item in ground_truth[field]}

        if len(current - truth) > 0 and len(truth - current) == 0:
            # removed something
            if (
                field == 'therapies'
                and len(
                    {ground_truth['therapyInteractionType'], row['therapyInteractionType']}
                    & {'COMBINATION', 'SEQUENTIAL'}
                )
                == 0
            ):
                continue

        if len(truth - current) > 0 and len(current - truth) == 0:
            # missing from
            if (
                field == 'phenotypes'
                or len(
                    {ground_truth['therapyInteractionType'], row['therapyInteractionType']}
                    & {'COMBINATION', 'SEQUENTIAL'}
                )
                == 0
            ):
                continue

        if current != truth:
            diff[field] = [current, truth]

    for field in REV_FIELD_MAPPING.values():
        if field in ['phenotypes', 'therapies', 'disease', 'molecularProfile']:
            continue
        if pd.isnull(row[field]) or pd.isnull(ground_truth[field]):
            if pd.isnull(row[field]) != pd.isnull(ground_truth[field]):
                diff[field] = [row[field], ground_truth[field]]

        elif normalize(row[field]) != normalize(ground_truth[field]):
            diff[field] = [row[field], ground_truth[field]]

    if len(diff) == 1:
        # something removed exclusively to match new stds for diff evidence types
        _, truth_value = list(diff.values())[0]
        if pd.isnull(truth_value):
            diff = {}
    if 'therapies' in diff:
        diff['therapyInteractionType'] = [
            row['therapyInteractionType'],
            ground_truth['therapyInteractionType'],
        ]

    return diff


def apply_revision(row, rev):
    row_field = REV_FIELD_MAPPING[rev['fieldName']]
    if rev['linkoutData.diffValue.__typename'] == 'ObjectFieldDiff':
        if not isinstance(row[row_field], list):
            row[row_field] = (
                rev['linkoutData.diffValue.suggestedObjects'][0]
                if len(rev['linkoutData.diffValue.suggestedObjects'])
                else None
            )
        else:
            revised_objects = row[row_field]
            removed_ids = {o['id'] for o in rev['linkoutData.diffValue.removedObjects']}
            revised_objects = [r for r in revised_objects if r['id'] not in removed_ids]
            revised_objects.extend(rev['linkoutData.diffValue.suggestedObjects'])
            row[row_field] = revised_objects
    else:
        if (
            rev['fieldName'] == 'therapyInteractionType'
            and rev['suggestedValue'] is None
            or pd.isnull(rev['suggestedValue'])
        ):
            row[row_field] = 'SUBSTITUTES'
        else:
            if row_field == 'evidenceDirection':
                row[row_field] = normalize_evidence_direction(rev['suggestedValue'])
            elif row_field == 'significance':
                row[row_field] = normalize_significance(rev['suggestedValue'])
            elif row_field == 'variantOrigin':
                row[row_field] = normalize_variant_origin(rev['suggestedValue'])
            elif row_field == 'therapyInteractionType':
                row[row_field] = rev['suggestedValue'].upper()
            else:
                row[row_field] = rev['suggestedValue']


def revert_revision(row, rev):
    row_field = REV_FIELD_MAPPING[rev['fieldName']]

    if rev['linkoutData.diffValue.__typename'] == 'ObjectFieldDiff':
        if not isinstance(row[row_field], list):
            row[row_field] = (
                rev['linkoutData.diffValue.removedObjects'][0]
                if len(rev['linkoutData.diffValue.removedObjects'])
                else None
            )
        else:
            revised_objects = row[row_field]
            removed_ids = set()
            for obj in rev['linkoutData.diffValue.suggestedObjects']:
                removed_ids.add(obj['id'])
            revised_objects = [r for r in revised_objects if r['id'] not in removed_ids]
            revised_objects.extend(rev['linkoutData.diffValue.removedObjects'])
            row[row_field] = revised_objects
    else:
        if (
            rev['fieldName'] == 'therapyInteractionType'
            and rev['currentValue'] is None
            or pd.isnull(rev['currentValue'])
        ):
            row[row_field] = 'SUBSTITUTES'
        else:
            if row_field == 'evidenceDirection':
                row[row_field] = normalize_evidence_direction(rev['currentValue'])
            elif row_field == 'significance':
                row[row_field] = normalize_significance(rev['currentValue'])
            elif row_field == 'variantOrigin':
                row[row_field] = normalize_variant_origin(rev['currentValue'])
            elif row_field == 'therapyInteractionType':
                row[row_field] = rev['currentValue'].upper()
            else:
                row[row_field] = rev['currentValue']


def auto_accept_pending_editor_revisions(row):
    revision_state = []
    for rev in row['revisions.nodes']:
        if rev['role'] == 'EDITOR' and rev['status'] == 'NEW':
            # apply revision to current state
            rev['status'] = 'ACCEPTED'
            row['auto_accepted'] = True
            apply_revision(row, rev)
            revision_state.append(f"+RID{rev['id']}")

    row['revision_state'] += ''.join(revision_state)
    return dict(row)


def build_new_revsion_state(row):
    # for a given row, accept any editor pending changes and make this the "correct" state
    # create previous states from revisions 1-n elements and include suggested changes
    # give it a new EID as EID-RID and include a field for "revision_state"
    row['revision_state'] = ''
    row = auto_accept_pending_editor_revisions(row)
    row['ground_truth'] = True
    row['corrections'] = {}
    ground_truth = row

    # now choose some revisions to do rollbacks with to create civic entries that need revision
    prev_accepted_revisions = sorted(
        [r for r in ground_truth['revisions.nodes'] if r['status'] == 'ACCEPTED'],
        key=lambda r: r['revisionSetId'],
    )
    prev_accepted_revision_sets = []
    for rev in prev_accepted_revisions:
        if not prev_accepted_revision_sets:
            prev_accepted_revision_sets.append([rev])
        elif rev['revisionSetId'] == prev_accepted_revision_sets[-1][-1]['revisionSetId']:
            prev_accepted_revision_sets[-1].append(rev)
        else:
            prev_accepted_revision_sets.append([rev])

    result = [ground_truth]
    for rev_set in prev_accepted_revision_sets:
        modified_row = dict(ground_truth)
        rev_state = ''.join(
            sorted({f"-RID{r['id']}" for r in sorted(rev_set, key=lambda x: x['id'])})
        )
        notes = []
        for rev in rev_set:
            if rev['creationActivity.note'] not in notes and isinstance(
                rev['creationActivity.note'], str
            ):
                notes.append(rev['creationActivity.note'])
        modified_row['revision_note'] = '; '.join(notes)
        modified_row['revision_state'] += rev_state
        modified_row['ground_truth'] = False
        modified_row['revisions.nodes'] = [{**r, 'status': 'NEW'} for r in rev_set]

        for rev in rev_set:
            revert_revision(modified_row, rev)

        modified_row['corrections'] = generate_corrections(modified_row, ground_truth)
        result.append(modified_row)
    assert len(result) >= 1 and result[0]['ground_truth']
    return result


def get_admin_flagged(events):
    flag_by = None
    events = events['nodes']
    for event in sorted(events, key=lambda x: x['createdAt']):
        if event['action'] == 'FLAGGED':
            flag_by = event['originatingUser']['displayName']
            if flag_by == 'CIViC_Bot' or event['originatingUser']['role'] not in {
                'EDITOR',
                'ADMIN',
            }:
                flag_by = None
        if event['action'] == 'FLAG_RESOLVED':
            flag_by = None
    return flag_by is not None


def filter_bad_eids(df):
    """
    drop anything with a non CIViC-bot flag or any of the newly invalidated combinations
    does not support -> N/A
    """

    df['_drop'] = False
    df.loc[
        (df.significance.isin({'NA', 'N/A'})) & (df['evidenceDirection'] == 'DOES_NOT_SUPPORT'),
        '_drop',
    ] = True
    logging.info(f'dropping {df._drop.sum()} EIDs with DOES_NOT_SUPPORT NA')
    df = df[df._drop == False]
    # df['_flagged'] = df.events.apply(get_admin_flagged)
    # df.loc[(df['_flagged'] == True), '_drop'] = True
    # logging.info(f'dropping {df._drop.sum()} EIDs flagged by non-CIViC_Bot ADMIN/EDITOR users')
    # df = df[df._drop == False]
    df.loc[
        (df['evidenceDirection'].isin({'NA', 'N/A'})) | (df['evidenceDirection'].isna()), '_drop'
    ] = True
    dropped_eids = set(df[df._drop == True]['id'].unique()) - set(
        df[df._drop == False]['id'].unique()
    )
    logging.info(f'dropping {len(dropped_eids)} EIDs with N/A for evidence direction')
    df = df[df._drop == False]
    return df.drop(columns=[c for c in df.columns if c.startswith('_')])


def get_last_update(events):
    """
    Any non-bot interaction/update
    """
    ev_df = pd.json_normalize(events['nodes'])
    ev_df = ev_df[ev_df['originatingUser.displayName'].str.lower() != 'civic_bot']
    ev_df = ev_df[
        (ev_df['originatingUser.role'].isin({'ADMIN', 'EDITOR'})) | (ev_df['action'] == 'SUBMITTED')
    ]
    ev_df['createdAt'] = pd.to_datetime(ev_df.createdAt)
    return ev_df.createdAt.max()


def main():
    logging.basicConfig(level=logging.INFO)
    requests_cache.install_cache(
        '.requests_cache',
        urls_expire_after={
            '*': 24 * 60 * 60 * 10,  # 10 days
        },
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    df = read_jsonl(args.input)
    df['lastUpdatedAt'] = df.events.apply(get_last_update)
    print(df.events.tolist()[0])
    df['submittedAt'] = df.events.apply(
        lambda events: min([e['createdAt'] for e in events['nodes']])
    )
    df = filter_bad_eids(df)
    df['evidenceDirection'] = df.evidenceDirection.apply(normalize_evidence_direction)
    df['significance'] = df.significance.apply(normalize_significance)
    df['variantOrigin'] = df.variantOrigin.apply(normalize_variant_origin)

    df['therapyInteractionType'] = df['therapyInteractionType'].fillna('SUBSTITUTES').str.upper()
    initial_rows = df.shape[0]
    logging.info(f'{initial_rows} initial rows')
    df['revisions.nodes'] = df['revisions'].str['nodes']
    df = df.drop(columns=['revisions'])

    rev_df = df.copy()
    rev_df['revisions.nodes'] = rev_df.apply(filter_relevant_revisions, axis=1)
    unchanged_df = rev_df[rev_df['revisions.nodes'].fillna('').str.len() == 0].copy()
    unchanged_df['ground_truth'] = True

    rev_df = rev_df[rev_df['revisions.nodes'].fillna('').str.len() > 0].copy()
    rev_eids = set(rev_df['id'].unique())
    rev_df = (
        rev_df.explode('revisions.nodes')
        .reset_index(drop=True)
        .rename(columns={'revisions.nodes': 'rev'})
    )
    assert rev_eids == set(rev_df['id'].unique())
    rev_df = normalize_and_expand_column(rev_df, 'rev')
    assert rev_eids == set(rev_df['id'].unique())
    rev_df['resolved'] = rev_df['rev.status'] != 'NEW'
    rev_df['rev.role'] = rev_df['rev.creationActivity.user.role']
    rev_df.loc[rev_df['rev.role'] == 'ADMIN', 'rev.role'] = 'EDITOR'

    resolve_revision_links(rev_df)
    assert rev_eids == set(rev_df['id'].unique())

    grouped_df = (
        rev_df.groupby(['id'], dropna=False)
        .apply(
            lambda s: s[[c for c in rev_df.columns if c.startswith('rev.')]]
            .rename(columns={c: c[len('rev.') :] for c in rev_df.columns if c.startswith('rev.')})
            .to_dict('records')
        )
        .reset_index(name='revisions.nodes')
        .merge(df.drop(columns=['revisions.nodes']), on='id')
    )
    assert rev_eids == set(grouped_df['id'].unique())

    new_rows = grouped_df.apply(build_new_revsion_state, axis=1).tolist()
    df = pd.DataFrame.from_records(itertools.chain.from_iterable(new_rows))
    assert rev_eids == set(df['id'].unique())
    df = df[(df.corrections.str.len() > 0) | (df.ground_truth == True)]
    df = pd.concat([df, unchanged_df])

    df = df[~df.variantOrigin.isna()]

    logging.info(f'{df.shape[0]} final rows')
    assert df.shape[0] >= initial_rows
    dump_jsonl(df, args.output)


if __name__ == '__main__':
    main()
    pass
