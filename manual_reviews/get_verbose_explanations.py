import os

from article_utils.util import dump_jsonl, read_jsonl

df = read_jsonl(
    'data/doccano/2025.05.01_llm_explanation_quality_v2025.03.25/inputs/examples.jsonl.gz'
)
print(df)
print(df.columns.tolist())
df = df.drop_duplicates(['claim.flat', 'evidence.flat', 'gold_label_name', 'review_id']).drop(
    columns=[
        'template',
        'predicted_label_name',
        'model',
        'generated_text',
        'filename',
        'prompt',
        'document.content_source_x',
    ]
)
df['id'] = df['review_id']

# add group consensus labels
gold_df = read_jsonl(
    'data/doccano/2025.05.01_llm_explanation_quality_v2025.03.25/processed/gold_labels.jsonl.gz'
)
gold_df = gold_df[~gold_df['is_deleted'].fillna(False)]
print(gold_df.columns)
print(gold_df['comment'].tolist()[0])
gold_df = (
    gold_df.groupby(
        [
            'claim.flat',
            'evidence.flat',
            'document.pmid',
            'consensus.gold_label_name',
        ]
    )
    .agg({'consensus.counts': 'first', 'comment': list})
    .reset_index()
)

df = df.merge(
    gold_df.rename(columns={'comment': 'reviews'}),
    how='left',
    validate='one_to_one',
    on=['claim.flat', 'evidence.flat', 'document.pmid'],
)
df.loc[~df['consensus.gold_label_name'].isna(), 'gold_label_name'] = df['consensus.gold_label_name']
assert df.gold_label_name.isin({'NEI', "REFUTES", 'SUPPORTS'}).sum() == df.shape[0]
dump_jsonl(
    df,
    os.path.join(
        '/projects/creisle_prj/git/civicfact/data_builder/builds/explreview-2025.03.25',
        'data.jsonl.gz',
    ),
)
