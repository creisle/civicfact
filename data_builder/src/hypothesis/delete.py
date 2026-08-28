import argparse
import json
import logging
import os

from .api import delete_annotation, fetch_annotations_by_group

URL = 'https://api.hypothes.is/api/annotations'
LIMIT = 200

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api_key',
        default=os.environ.get('BOT_HYPOTHESIS_API_KEY'),
        help='API key to access hypothesis content. Defaults to the ENV variable BOT_HYPOTHESIS_API_KEY',
    )
    parser.add_argument(
        '--group',
        help='The annotation group you wish to download annotations from',
        default='ADmgdiEi',
    )
    args = parser.parse_args()

    annotations = fetch_annotations_by_group(args.group, args.api_key)

    annotations_by_id = {}
    for annotation in annotations:
        annotations_by_id[annotation['id']] = annotation

    protected = set()
    flagged_for_deletion = set()

    protected_claims = set()
    # check for comments on this annotation, do not remove those with comments unless the comment is tagged override-delete
    for annotation in annotations:
        if not annotation.get('references'):
            continue
        if 'override-delete' in annotation['tags']:
            flagged_for_deletion.update([annotation['id'], *annotation['references']])
        else:
            protected.update([annotation['id'], *annotation['references']])
            for referee in annotation['references']:
                referred_comment = annotations_by_id[referee]['text']
                if referred_comment.strip().startswith('{'):
                    claim = json.loads(referred_comment)['claim']
                    protected_claims.add(json.dumps(claim, sort_keys=True))

    # keep all claim-pairs where at least one segment has a comment
    for annotation in annotations:
        if annotation['text'].strip().startswith('{'):
            claim = json.dumps(json.loads(annotation['text'])['claim'], sort_keys=True)
            if claim in protected_claims:
                protected.add(annotation['id'])

    protected -= flagged_for_deletion

    for annotation_id in set(annotations_by_id) - protected:
        delete_annotation(args.group, args.api_key, annotation_id)

    logging.info(f'{len(protected)} annotations were protected from deleted by comments/replies')
    logging.info(f'{len(set(annotations_by_id) - protected)} annotations were deleted')
