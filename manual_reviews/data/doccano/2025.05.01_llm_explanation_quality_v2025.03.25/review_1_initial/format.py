"""
Since using API dumps instead of downloads now, format this like an API dump where possible

{
            "created_at": "2025-07-30T16:31:29.937534Z",
            "end_offset": 118,
            "example": 118735,
            "id": 11977,
            "label": 1195,
            "labelname": "",
            "prob": 0.0,
            "start_offset": 10,
            "updated_at": "2025-07-30T16:31:29.937542Z",
            "user": 1,
            "username": "admin"
        },
"""

import glob
import logging
import os

from article_utils.util import dump_jsonl, read_jsonl

logging.basicConfig(level=logging.INFO)

CURR_FOLDER = os.path.dirname(__file__)
project = {
    'name': "CIViC-Fact full pipeline review - LLM responses",
    'collaborative_annotation': False,
    "project_type": "SequenceLabeling",
}

df = read_jsonl(*glob.glob(os.path.join(CURR_FOLDER, '*.jsonl')), add_filename=True)
df['username'] = df.filename.str.extract(r'.*/(\w+)\.jsonl')
df['project'] = df.apply(lambda _: project, axis=1)
df['meta'] = df.apply(
    lambda row: {'review_id': row['review_id'], 'contextHtml': row['contextHtml']}, axis=1
)
df['comments'] = df.apply(
    lambda row: [{'username': row['username'], 'text': c} for c in row['Comments']], axis=1
)
df['annotations'] = df.apply(
    lambda row: [
        {'start_offset': c[0], 'end_offset': c[1], 'labelname': c[2], 'username': row['username']}
        for c in row['label']
    ],
    axis=1,
)
df['is_confirmed'] = True
df = df.drop(columns=['Comments', 'review_id', 'label', 'contextHtml'])
df['assignments'] = df.username.apply(lambda name: [{'username': name}])
dump_jsonl(df, os.path.join(CURR_FOLDER, 'examples.jsonl.gz'))
