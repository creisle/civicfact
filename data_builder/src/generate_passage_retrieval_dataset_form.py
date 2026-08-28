import argparse
import itertools

import pandas as pd

from article_utils.util import dump_jsonl, read_jsonl


def filter_conversation(conversation):
    if not conversation:
        return []
    conv_df = pd.DataFrame.from_records(conversation)
    conv_df = conv_df[conv_df.comment.fillna('') != '']
    conv_df = (
        conv_df.groupby(['user', 'comment', 'id'])
        .agg({'tags': lambda x: set(itertools.chain.from_iterable(x)), 'created': 'last'})
        .reset_index()
    )

    return conv_df.to_dict('records')


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()


df = read_jsonl(args.input)
df = df[df['civic.ground_truth'] == True]
print(sorted(df.columns))
df['conversation'] = df['conversation'].apply(filter_conversation)
df.loc[df.gold_label_name == 'REFUTES', 'gold_label_name'] = (
    'SUPPORTS'  # we add this part to the claim itself now
)

dump_jsonl(
    df[df.flagged != True][
        [
            'civic.id',
            'claim',
            'evidence',
            'claim.flat',
            'evidence.flat',
            'conversation',
            'hypothesis.username',
            'gold_label_name',
            'document.pmid',
        ]
    ],
    args.output,
)

pass
