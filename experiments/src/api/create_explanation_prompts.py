import argparse

from utils import dump_jsonl, read_jsonl

INSTRUCTIONS = """\
You will receive:
- a claim inside <claim>...</claim> tags
- evidence inside <evidence>...</evidence> tags
- a pre-assigned label inside <label>...</label> tags
- optionally, a revised version of the claim inside <revised_claim>...</revised_claim> tags

Your task is to generate a brief explanation (3-6 sentences) for why the given label is appropriate.

Label meanings:
- SUPPORTS: The evidence fully supports the claim as written. No revision is needed.
- REVISIONS: The evidence partially supports the claim, and the revised claim is fully supported by the evidence.
- NEI: The evidence does not support the claim, is irrelevant to the claim, or provides insufficient information.

Explanation requirements:
- Do not change or reassign the label.
- Explain why the provided label fits the relationship between the evidence and the claim.
- Focus on the stance of the evidence toward the claim.
- If the label is REVISIONS, explain how the revised claim better aligns with the evidence.
- Keep the output under 500 tokens.\
"""


def create_prompt(row):
    prompt = [
        INSTRUCTIONS,
        "<claim>",
        row["claim.flat"],
        "</claim>",
        "<evidence>",
        row["evidence.flat"],
        "</evidence>",
        "<label>",
        row["gold_label_name"],
        "</label>",
    ]
    if row["gold_label_name"] == "REVISIONS":
        prompt.extend(["<revised_claim>", row["revised_claim.flat"], "</revised_claim>"])

    return [{"role": "user", "content": "\n".join(prompt)}]


parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--random_seed", required=True, type=int)
parser.add_argument("--subsample", required=True, type=float)
parser.add_argument("output")
args = parser.parse_args()

assert args.subsample > 0 and args.subsample <= 1

df = read_jsonl(
    args.input, required_columns=["claim.flat", "gold_label_name", "evidence.flat", "partition", "revised_claim.flat"]
)
df = df[df.partition.isin({"train", "dev"})]
df = df.sample(frac=args.subsample, random_state=args.random_seed)
df["chat"] = df.apply(create_prompt, axis=1)

dump_jsonl(df, args.output)
