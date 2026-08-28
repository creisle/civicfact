import pandas as pd
from article_utils.util import read_jsonl

from doccano_client import DoccanoClient
import logging
import json
import argparse

from doccano.util import get_project_by_name, list_examples, list_annotations, set_example_label, list_users


def list_categories(client, project_id):
    return client._base_repository.get(f'projects/{project_id}/category-types').json()


rand_sel_df = read_jsonl('../data_builder/builds/civicreview-2025.03.25v3/data.jsonl.gz')
RANDON_REVIEW_EIDS = set(rand_sel_df['civic.id'].unique())


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', default='Application to CIViC Manual Review v2')
parser.add_argument('--api_url', default='https://doccano.bcgsc.ca')
parser.add_argument(
    '--passwords',
    default='.doccano_users.json',
    help='json file with user password pairs b/c doccano won\'t let you dump annotations properly otherwise but also excludes loads of details from the export',
)
args = parser.parse_args()

with open(args.passwords, 'r') as fh:
    passwords = json.load(fh)

# instantiate a client and log in to a Doccano instance
client = DoccanoClient(args.api_url)
client.login(username='admin', password=passwords['admin'])

users = list_users(client)
admin_user = [u for u in users if u['username'] == 'admin'][0]

proj = get_project_by_name(client, args.project_name)


examples = list_examples(client, proj['id'])
print(examples[0]['meta'].keys())

tag_types = list_categories(client, proj['id'])
print(tag_types)
TAG_ID = [t['id'] for t in tag_types if t['text'] == 'r100'][0]

colab_mode = proj['collaborative_annotation']

for ex in examples:
    if ex['meta']['civic.id'] not in RANDON_REVIEW_EIDS:
        continue
    annotations = list_annotations(client, proj['project_type'], proj['id'], ex['id'])
    labels = {l['labelname'] for l in annotations}
    # print(labels)
    if 'r100' not in labels:
        set_example_label(client, proj['id'], ex['id'], admin_user['id'], TAG_ID)
