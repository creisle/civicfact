from article_utils.util import dump_jsonl, read_jsonl

# create spans of claim-relevant text from the answers we copied to the comments


def create_spans(row):
    matches = []

    for comment in row.Comments:
        comment = comment.strip()
        pos = row.generated_text.find(comment.strip())
        if pos >= 0:
            matches.append([pos, pos + len(comment), "claim-relevant"])
    return matches


df = read_jsonl('admin.jsonl.gz')


df['cat'] = df.label
df['label'] = df.apply(create_spans, axis=1)
df['contextHtml'] = df.text
df['text'] = df.generated_text
df = df.drop(columns=['generated_text'])
dump_jsonl(df, 'admin.seqlabel.jsonl.gz')
