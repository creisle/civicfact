import re
from dataclasses import dataclass

import pandas as pd

EXAMPLES_COLNAME = "prompt_examples"
DEFAULT_RESPONSE_TEMPLATE = "[/INST]"  # Llama2 style, marks end of user instruction
RESPONSE_TEMPLATE_FIELD = "prompt.response_template"


# JINJA templates used for apply_chat_template function
# https://huggingface.co/docs/transformers/main/en/chat_templating#how-do-i-create-a-chat-template
@dataclass
class ChatTemplate:
    template: str
    generation_prompt: str


class Template:
    passage_ordering: bool = False
    preface: str = ""

    @classmethod
    def build_label(self, example: dict) -> str:
        return f"label: {example['gold_label_name']}"

    @classmethod
    def build_example(self, example) -> str:
        return f"""\
claim: {example["claim.flat"]}
evidence: {example["evidence.flat"]}
{self.build_label(example)}"""

    @classmethod
    def build_chat(cls, example):
        prompt = [cls.preface]
        for prompt_example in example.get(EXAMPLES_COLNAME, []):
            prompt.append(cls.build_example(prompt_example))

        current_ex = dict()
        current_ex.update(example)
        current_ex["gold_label_name"] = ""
        current_ex["reason"] = ""
        current_ex["summary"] = ""
        prompt.append(cls.build_example(current_ex))

        prompt = "\n\n".join([p for p in prompt if p])

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": cls.build_label(example)},
        ]


class ObjectiveTemplate(Template):
    preface = "Given a claim and some evidence, determine the stance of the evidence toward the claim."


class BasicTemplate(Template):
    preface = "Given a claim and some evidence, determine the stance of the evidence toward the claim. Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information)."


class OnlyLabelsTemplate(Template):
    preface = "Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information). Respond with the label first, followed by a short explanation of your reasoning."


class LabelsTemplate(Template):
    preface = (
        "Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information)."
    )


class BasicFewTemplate(BasicTemplate):
    preface = (
        BasicTemplate.preface
        + " You will be given a few examples, determine the stance of the last claim/evidence pair."
    )


class ReasonTemplate(Template):
    preface = BasicTemplate.preface + " Give your reasoning."

    @classmethod
    def build_label(self, example: dict) -> str:
        if example["reason"]:
            return f"label: {example['gold_label_name']}. {example['reason']}"
        return Template.build_label(example)


class ReasonLimit100(Template):
    preface = ReasonTemplate.preface + " Your response should not exceed 100 tokens."


class ReasonTemplate2(Template):
    preface = BasicTemplate.preface + " Give your reasoning."

    @classmethod
    def build_label(self, example: dict) -> str:
        if not example["gold_label_name"]:
            return ""
        if example["reason"]:
            return (
                f"label: {example['gold_label_name']}\nreasoning: {example['reason']}"
            )
        return Template.build_label(example)


class LabelFirstTemplate(BasicTemplate):
    preface = (
        BasicTemplate.preface
        + " Respond with the label first, followed by a short explanation of your reasoning."
    )

    @classmethod
    def build_label(self, example: dict) -> str:
        if not example["gold_label_name"]:
            return ""
        if example["reason"]:
            return (
                f"label: {example['gold_label_name']}\nreasoning: {example['reason']}"
            )
        return Template.build_label(example)


class VerboseTemplate(LabelFirstTemplate):
    preface = """\
Given a claim and some evidence, determine the stance of the evidence toward the claim. \

Use one of the following 3 labels:
- SUPPORTS: Given only the evidence (and users' expertise) but no other parts of the paper, the evidence is sufficient and specific enough to support the claim.
- REFUTES: There is evidence against the claim. For example, the evidence suggests. sensitivity instead of resistance. The evidence is sufficient and specific enough to refute the claim.
- NEI (not enough information): The evidence is irrelevant to the claim or some elements of the claim are missing or are not specific enough; the conclusion is ambiguous; or there is conflicting information.

Respond with the label first, followed by a short explanation of your reasoning."""


class VerboseLimit100Template(VerboseTemplate):
    preface = VerboseTemplate.preface + " Your response should not exceed 100 tokens."


class VerboseOpenAIMdTemplate(LabelFirstTemplate):
    preface = """\
# Role and Objective

Given a claim and some evidence, determine the stance of the evidence toward the claim.

# Instructions

Use one of the following 3 labels:
- SUPPORTS: Given only the evidence (and users' expertise) but no other parts of the paper, the evidence is sufficient and specific enough to support the claim.
- REFUTES: There is evidence against the claim. For example, the evidence suggests. sensitivity instead of resistance. The evidence is sufficient and specific enough to refute the claim.
- NEI (not enough information): The evidence is irrelevant to the claim or some elements of the claim are missing or are not specific enough; the conclusion is ambiguous; or there is conflicting information.

# Output Format

Respond with the label first, followed by a short explanation of your reasoning.

# Context
"""

    @classmethod
    def build_label(self, example: dict) -> str:
        if example["reason"]:
            return f"## Label and Explanation\n{example['gold_label_name']}. {example['reason']}"
        return f"## Label and Explanation\n{example['gold_label_name']}"

    @classmethod
    def build_example(self, example) -> str:
        return f"""\
## Claim
{example["claim.flat"]}

## Evidence
{example["evidence.flat"]}

{self.build_label(example)}"""


class OpenAIMdTemplateLimit(VerboseOpenAIMdTemplate):
    preface = """\
# Role and Objective

Given a claim and some evidence, determine the stance of the evidence toward the claim.

# Instructions

Use one of the following 3 labels:
- SUPPORTS: Given only the evidence (and users' expertise) but no other parts of the paper, the evidence is sufficient and specific enough to support the claim.
- REFUTES: There is evidence against the claim. For example, the evidence suggests. sensitivity instead of resistance. The evidence is sufficient and specific enough to refute the claim.
- NEI (not enough information): The evidence is irrelevant to the claim or some elements of the claim are missing or are not specific enough; the conclusion is ambiguous; or there is conflicting information.

# Output Format

Respond with the label first, followed by a short explanation of your reasoning. Your response should not exceed 100 tokens.

# Context
"""


class ClaimOnlyTemplate(ReasonTemplate):
    preface = "Given a claim, determine if the claim is true. Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information). Give your reasoning"

    @classmethod
    def build_example(self, example) -> str:
        return f"""\
claim: {example["claim.flat"]}
{self.build_label(example)}"""


class ClaimOnlyLabelFirstTemplate(ReasonTemplate):
    preface = "Given a claim, determine if the claim is true. Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information). Respond with the label first, followed by a short explanation of your reasoning."

    @classmethod
    def build_example(self, example) -> str:
        return f"""\
claim: {example["claim.flat"]}
{self.build_label(example)}"""


class FullTextSummaryTemplate(ReasonTemplate):
    preface = """Given a claim and some evidence, summarize the relevant parts of the evidence and \
then determine the stance of the evidence toward the claim. \
Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information). \
Give your reasoning."""

    @classmethod
    def build_label(self, example: dict) -> str:
        if not example["gold_label_name"]:
            return example["gold_label_name"]
        label = f"label: {example['gold_label_name']}"
        if example["reason"]:
            label += f". {example['reason']}"
        if example["summary"]:
            label = f"summary: {example['summary']}\n" + label
        return label

    @classmethod
    def build_example(self, example) -> str:
        content = f"""\
claim: {example["claim.flat"]}
evidence: {example["evidence.flat"]}"""
        label = self.build_label(example)
        if label:
            content += "\n" + label

        return content


class JsonTrain(Template):
    preface = ""

    @classmethod
    def build_label(self, example: dict) -> str:
        if not example["gold_label_name"]:
            return example["gold_label_name"]
        label = f"{example['gold_label_name']}"
        if example["gold_label_name"] == "REVISIONS":
            if (
                not pd.isnull(example["revised_claim.flat"])
                and example["revised_claim.flat"] != ""
            ):
                label += "\n\n" + example["revised_claim.flat"]
            elif example["gold_label_name"] == "REVISIONS":
                raise ValueError(
                    "prompt examples with the revison label must have the revised_claim.flat field as well"
                )
        if example.get("reason"):
            label += f"\n\n{example['reason']}"
        return label

    @classmethod
    def build_chat(cls, example):
        assert cls.preface == ""
        prompt = f"""\
<claim id="target">
{example["claim.flat"]}
</claim>
<evidence id="target">
{example["evidence.flat"]}
</evidence>"""

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": cls.build_label(example)},
        ]


class JsonLimit200(JsonTrain):
    preface = """\
# Instructions

You will be a claim/evidence pair where the input is denoted with the <claim> and \
<evidence> tags.

classify the pair with one of the following labels:

- SUPPORTS: Supported by the evidence, no changes are required
- REVISIONS: Partially supported by the evidence but would be fully supported if the claim were revised
- NEI: The claim is not relevant to or not supported by the evidence

Respond with the label first, followed by the newly suggested claim in JSON format if the REVISIONS label is applicable and finally a short written explanation.
Limit your output to 200 tokens.
"""

    @classmethod
    def build_chat(cls, example):
        prompt = [cls.preface, ""]

        prompt.append(
            f"""\
<claim id="target">
{example["claim.flat"]}
</claim>
<evidence id="target">
{example["evidence.flat"]}
</evidence>"""
        )

        prompt = "\n\n".join([p for p in prompt if p])

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": cls.build_label(example)},
        ]


class JsonLimit200Examples(JsonTrain):
    preface = """\
# Instructions

You will be given a series of claim/evidence pairs where the input is denoted with the <claim> and \
<evidence> tags and the expected reponse is given by the <assistant_response> tag. You will also be \
given a final pair with only the <claim> and <evidence> and asked to generate the corresponding response \
following the formatting of previous examples.

classify the pair with one of the following labels:

- SUPPORTS: Supported by the evidence, no changes are required
- REVISIONS: Partially supported by the evidence but would be fully supported if the claim were revised
- NEI: The claim is not relevant to or not supported by the evidence

Respond with the label first, followed by the newly suggested claim if the REVISIONS label is applicable and finally a short written explanation.
Limit your output to 200 tokens.
"""

    @classmethod
    def build_chat(cls, example):
        if len(example.get(EXAMPLES_COLNAME, [])) < 1:
            raise AssertionError(
                "This prompt expects a non-zero number of input examples"
            )
        prompt = [cls.preface, "", "## Examples", ""]
        for ex_i, prompt_example in enumerate(example.get(EXAMPLES_COLNAME, [])):
            prompt.append(
                f"""\
<claim id="example-{ex_i}">
{prompt_example["claim.flat"]}
</claim>
<evidence id="example-{ex_i}">
{prompt_example["evidence.flat"]}
</evidence>
<assistant_response id="example-{ex_i}">
{cls.build_label(prompt_example)}
</assistant_response>"""
            )

        prompt.append(
            f"""\
<claim id="target">
{example["claim.flat"]}
</claim>
<evidence id="target">
{example["evidence.flat"]}
</evidence>"""
        )

        prompt = "\n\n".join([p for p in prompt if p])

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": cls.build_label(example)},
        ]


class JsonLimit200ExamplesVerbose(JsonLimit200Examples):
    preface = (
        JsonLimit200Examples.preface
        + """\

# Context

## disease

The disease field generally requires an value that is known to the Disease Ontology database. \
Within the CIViC database, the disease should be the cancer or cancer subtype that is associated to \
the described Molecular Profile (variant or ensemble of variants) by the evidence being curated. \
The disease selected should be as specific as possible and should reflect the disease type \
described in the source that supports the evidence statement. In CIViC, Disease is largely \
synonymous with cancer type or sub-type.

## evidenceDirection

Evidence Direction indicates if the Evidence Statement supports or refutes the significance of an event.
The available options include: “SUPPORTS” or “DOES_NOT_SUPPORT”.

## evidenceType and significance

Significance describes how the variant is related to a specific, clinically-relevant property as described in the Evidence Statement. Significance terms vary depending on the Evidence Type (e.g. “Resistance” for Predictive Evidence, “Better Outcome” for Prognostic Evidence, etc.).

### PREDICTIVE

- Sensitivity/Response: Associated with a clinical or preclinical response to treatment
- Reduced Sensitivity: Response to treatment is lower than seen in other treatment contexts
- Resistance: Associated with clinical or preclinical resistance to treatment
- Adverse Response: Associated with an adverse response to drug treatment
- N/A: Variant does not inform clinical interpretation

### DIAGNOSTIC

- Positive: Associated with diagnosis of disease or subtype
- Negative: Associated with lack of disease or subtype

### PROGNOSTIC

- Better Outcome: Demonstrates better than expected clinical outcome
- Poor Outcome: Demonstrates worse than expected clinical outcome
- N/A: Variant does not inform clinical action

### PREDISPOSITION

- Predisposition: Germline variant has predisposing potential for cancer, and may meet select ACMG/AMP criteria supporting pathogenic or benign classification.
- Protectiveness: Germline variant has properties that protect individuals from acquiring cancer.

### ONCOGENIC

- Oncogenicity: Somatic variant has oncogenic potential for driving cancer, and may meet select ClinGen/CGC/VICC criteria supporting pathogenic or benign classification.
- Protectiveness: Somatic variant has a protective role against cancer.

### FUNCTIONAL

- Gain of Function: Sequence variant confers an increase in normal gene function
- Loss of Function: Sequence variant confers a diminished or abolished function
- Unaltered Function: Gene product of sequence variant is unchanged
- Neomorphic: Sequence variant creates a novel function
- Dominant Negative: Sequence variant abrogates function of wildtype allele gene product
- Unknown: Sequence variant that cannot be precisely defined by the other listed categories

## therapies

Multiple Therapies can be added to a single Evidence Item, however, if two or more Therapies are added to an Evidence Item the curator is required to provide a Therapy Interaction Type. Therapy Interaction Types can be either combination, sequential, or substitutes.

Therapy terms should be derived from NCIt whereever possible using the preferred name.

The NCIT supports drug classes (e.g., EGFR Tyrosine Kinase Inhibitor, C2167), but the most precise therapy or combination should be used whenever possible (e.g., Afatinib, Erlotinib - Substitutes).

## variantOrigin

- Somatic: Variant is a mutation, found only in tumor cells, having arisen in a specific tissue (non-germ cell), and is not expected to be inherited or passed to offspring.
- Rare Germline: Variant is found in every cell (not restricted to tumor/diseased cells) and is thought to exist in less than 1%% of the population relevant to this evidence item.
- Common Germline: Variant is found in every cell (not restricted to tumor/diseased cells) and is thought to exist in at least 1%% of the population relevant to this evidence item.
- Combined: Variants in the corresponding Complex Molecular Profile have heterogeneous origins.
- Mixed: The population of patient variants described in the Evidence is a mixture of Somatic and Germline.
- Unknown: The variant origin is uncertain based on the available evidence.
- N/A: The variant type (e.g., expression) is not compatible (or easily classified) with the CIViC concept of variant origin.
"""
    )


def condense_md_table(md_table):
    """
    remove unnecessary whitespace in md table
    """
    lines = []
    for line in md_table.split("\n"):
        line = line.strip()
        line = re.sub(r"\s+\|\s+", " | ", line)
        line = re.sub(r"\s+\|\s*$", " |", line)
        for _ in range(line.count("|") - 1):  # will never need more repeats than this
            # overlapping matches
            new_line = re.sub(r"\|-----+\|", "|----|", line)
            if new_line == line:
                line = new_line
                break
            line = new_line
        lines.append(line)
    return "\n".join(lines)


def build_synonyms_table(row):
    assert "claim" in row, f"missing claim field in row: {row.keys()}"
    assert "meta" in row["claim"], f"missing meta field in claim: {row['claim']}"
    if len(row["claim"]["meta"]["synonyms"]) == 0:
        return ""
    df = pd.DataFrame.from_records(row["claim"]["meta"]["synonyms"])
    df = df[~df.synonyms.isna()]
    if df.shape[0] == 0:
        return ""
    assert df.synonyms.apply(lambda syn: isinstance(syn, list)).all(), (
        f"synonyms field must be a list of strings: {df.synonyms.apply(lambda s: type(s)).unique()}"
    )
    df["synonyms"] = df.synonyms.apply(lambda s: "; ".join(sorted(s)))
    return condense_md_table(df.to_markdown(index=False, tablefmt="github"))


class JsonLimit200ExamplesSynonyms(JsonTrain):
    preface = """\
# Instructions

You will be given a series of claim/evidence pairs where the input is denoted with the <claim> and \
<evidence> tags and the expected reponse is given by the <assistant_response> tag. Additional information is
provided on the elements of the claim as a markdown table of synonyms <synonyms> where known. You will also be \
given a final pair with only the <claim> and <evidence> and asked to generate the corresponding response \
following the formatting of previous examples.

classify the pair with one of the following labels:

- SUPPORTS: Supported by the evidence, no changes are required
- REVISIONS: Partially supported by the evidence but would be fully supported if the claim were revised
- NEI: The claim is not relevant to or not supported by the evidence

Respond with the label first, followed by the newly suggested claim if the REVISIONS label is applicable and finally a short written explanation.
Limit your output to 200 tokens.
"""

    @classmethod
    def build_chat(cls, example):
        if len(example.get(EXAMPLES_COLNAME, [])) < 1:
            raise AssertionError(
                "This prompt expects a non-zero number of input examples"
            )
        prompt = [cls.preface, "", "## Examples", ""]
        for ex_i, prompt_example in enumerate(example.get(EXAMPLES_COLNAME, [])):
            prompt.append(
                f"""\
<claim id="example-{ex_i}">
{prompt_example["claim.flat"]}
</claim>
<synonyms id="example-{ex_i}">
{build_synonyms_table(prompt_example)}
</synonyms>
<evidence id="example-{ex_i}">
{prompt_example["evidence.flat"]}
</evidence>
<assistant_response id="example-{ex_i}">
{cls.build_label(prompt_example)}
</assistant_response>"""
            )

        prompt.append(
            f"""\
<claim id="target">
{example["claim.flat"]}
</claim>
<synonyms id="target">
{build_synonyms_table(example)}
</synonyms>
<evidence id="target">
{example["evidence.flat"]}
</evidence>"""
        )

        prompt = "\n\n".join([p for p in prompt if p])

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": cls.build_label(example)},
        ]


class FullTextRetrieveAndVerify(JsonTrain):
    passage_ordering = True
    preface = """\
# Task

You are given a claim/evidence pair.

- The claim appears inside <claim> tags.
- The evidence appears inside <evidence> tags.
- The claim is a CIViC Evidence Item.
- The evidence is a PubMed article or abstract divided into passages of up to 1000 characters.
- Each passage has a numeric id.

Classify whether the evidence supports the claim.

Use only the provided evidence. Do not use outside biomedical knowledge except to interpret standard terminology, gene/protein names, variants, diseases, therapies, and study types.

# Input format

<claim>
...
</claim>

<evidence>
<passage id="0">
...
</passage>
<passage id="1">
...
</passage>
...
</evidence>

# Labels

Choose exactly one label:

## SUPPORTS

Use this when the evidence directly supports the claim as written.

Requirements:

- The key scientific assertions in the claim are supported.
- The evidence direction, significance, molecular profile, disease, therapy, and evidence type are supported when present.
- No material revision is needed.

## REVISIONS

Use this when the evidence is relevant and partially supports the claim, but the claim needs changes to be fully supported.

Use this label when:

- Some fields are supported but others are too broad, too specific, missing nuance, or incorrect.
- The claim could be made accurate with a focused revision.
- The revised claim should stay as close as possible to the original while removing or correcting unsupported parts.

## NEI

Use this when the claim is not supported by the evidence.

Use this label when:

- The article is unrelated to the claim.
- The evidence does not address the key claim.
- The evidence is only background or too indirect.
- The claim would require a substantially different claim rather than a revision.

# Decision rules

1. Check whether the evidence supports the claim’s core relationship, not just whether it mentions the same gene, variant, disease, or therapy.
2. Use SUPPORTS only when all material parts of the claim are supported.
3. Use REVISIONS when the article supports a narrower, corrected, or more precise version of the claim.
4. Use NEI when the evidence does not provide enough information to support or revise the claim.
5. Passage IDs should be ordered from most relevant to least relevant.
6. If any passages are relevant, all passage IDs should be ordered
7. If no passages are relevant, write: passages: none
8. Keep the explanation concise: 2–4 sentences.
9. Do not include reasoning steps, uncertainty scores, or extra sections.
10. Do not include any text before or after the required output fields.

# Output format

Return exactly the following fields, in this order:

label: SUPPORTS, REVISIONS, or NEI

revised claim: none, unless the label is REVISIONS. If the label is REVISIONS, provide a revised CIViC Evidence Item in JSON-like format.

passages: comma-separated passage IDs ordered by relevance

explanation: 2–4 sentence explanation

# Output examples

## For SUPPORTS:

label: SUPPORTS

revised claim: none

passages: 0, 3, 1, 2, 4

explanation: The evidence directly supports the claim as written. The cited passages describe the relevant molecular profile, disease context, and scientific relationship without requiring material changes to the claim.

## For REVISIONS:

label: REVISIONS

revised claim:
{
  "evidenceType": "FUNCTIONAL",
  "significance": "gain-of-function",
  "molecularProfile": "KRAS G12D",
  "evidenceDirection": "SUPPORTS"
}

passages: 0, 6, 2, 3, 1, 5, 4

explanation: The evidence supports a functional interpretation for the molecular profile, but the original claim includes details that are too broad or not directly supported. The revised claim narrows the assertion to the relationship supported by the cited passages.

## For NEI:

label: NEI

revised claim: none

passages: none

explanation: The evidence does not support the core claim. Although the article may mention related terminology, it does not provide evidence for the specific CIViC evidence relationship described in the claim.
"""

    @classmethod
    def build_label(self, example: dict) -> str:
        if not example["gold_label_name"]:
            return example["gold_label_name"]
        label = [f"label: {example['gold_label_name']}"]
        if example["gold_label_name"] == "REVISIONS":
            if (
                not pd.isnull(example["revised_claim.flat"])
                and example["revised_claim.flat"] != ""
            ):
                label.append("revised claim:\n" + example["revised_claim.flat"])
            elif example["gold_label_name"] == "REVISIONS":
                raise ValueError(
                    "prompt examples with the revison label must have the revised_claim.flat field as well"
                )
            else:
                label.append("revised claim: none")
        if example.get("reason"):
            label.append(f"explanation: {example['reason']}")

        return "\n\n".join(label)

    @classmethod
    def build_chat(cls, example):
        document_chunks = []
        for p_i, passage in enumerate(
            sorted(example["document.chunks"], key=lambda p: p["chunk.start"])
        ):
            document_chunks.append(
                f'<passage id="{p_i}">\n{passage["chunk.text"]}\n</passage>'
            )

        evidence = "\n".join(document_chunks)

        prompt = f"""\
{cls.preface}

<claim id="target">
{example["claim.flat"]}
</claim>

<evidence id="target">
{evidence}
</evidence>"""

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": cls.build_label(example)},
        ]


class VerifyPassages(JsonTrain):
    passage_ordering = False
    preface = """\
# Task

You are given a claim/evidence pair.

- The claim appears inside <claim> tags.
- The evidence appears inside <evidence> tags.
- The claim is a CIViC Evidence Item.
- The evidence is a series of passages from a PubMed article or abstract.
- Each passage has a start position representing the character index start of this passage in the original document.

Classify whether the evidence supports the claim.

Use only the provided evidence. Do not use outside biomedical knowledge except to interpret standard terminology, gene/protein names, variants, diseases, therapies, and study types.

# Input format

<claim>
...
</claim>

<evidence>
<passage start="123" end="670">
...
</passage>
<passage start="1073" end="2022">
...
</passage>
...
</evidence>

# Labels

Choose exactly one label:

## SUPPORTS

Use this when the evidence directly supports the claim as written.

Requirements:

- The key scientific assertions in the claim are supported.
- The evidence direction, significance, molecular profile, disease, therapy, and evidence type are supported when present.
- No material correction, narrowing, or qualification is needed.

## REVISIONS

Use this when the evidence is relevant and partially supports the claim, but the claim needs changes to be fully supported.

Use this label when:

- Some fields are supported but others are too broad, too specific, missing nuance, or incorrect.
- The claim could be made accurate with a focused revision.
- The revised claim should stay as close as possible to the original while removing or correcting unsupported parts.

## NEI

Use this when the claim is not supported by the evidence.

Use this label when:

- The article is unrelated to the claim.
- The evidence does not address the key claim.
- The evidence is only background or too indirect.
- The claim would require a substantially different claim rather than a revision.

# Decision rules

1. Check whether the evidence supports the claim’s core relationship, not just whether it mentions the same gene, variant, disease, or therapy.
2. Use SUPPORTS only when all material parts of the claim are supported.
3. Use REVISIONS when the article supports a narrower, corrected, or more precise version of the claim.
4. Use NEI when the evidence does not provide enough information to support or revise the claim.
5. Keep the explanation concise: 2–5 sentences.
6. Do not include reasoning steps, uncertainty scores, or extra sections.
7. Do not include any text before or after the required output fields.

# Output format

Return exactly the following fields, in this order:

label: SUPPORTS, REVISIONS, or NEI

revised claim: none, unless the label is REVISIONS. If the label is REVISIONS, provide a revised CIViC Evidence Item in JSON-like format.

explanation: 2–5 sentence explanation

# Output examples

## For SUPPORTS:

label: SUPPORTS

revised claim: none

explanation: The evidence directly supports the claim as written. The cited passages describe the relevant molecular profile, disease context, and scientific relationship without requiring material changes to the claim.

## For REVISIONS:

label: REVISIONS

revised claim:
{
  "evidenceType": "FUNCTIONAL",
  "significance": "gain-of-function",
  "molecularProfile": "KRAS G12D",
  "evidenceDirection": "SUPPORTS"
}

explanation: The evidence supports a functional interpretation for the molecular profile, but the original claim includes details that are too broad or not directly supported. The revised claim narrows the assertion to the relationship supported by the cited passages.

## For NEI:

label: NEI

revised claim: none

explanation: The evidence does not support the core claim. Although the article may mention related terminology, it does not provide evidence for the specific CIViC evidence relationship described in the claim.
"""

    @classmethod
    def build_label(self, example: dict) -> str:
        if not example["gold_label_name"]:
            return example["gold_label_name"]
        label = [f"label: {example['gold_label_name']}"]
        if example["gold_label_name"] == "REVISIONS":
            if (
                not pd.isnull(example["revised_claim.flat"])
                and example["revised_claim.flat"] != ""
            ):
                label.append("revised claim:\n" + example["revised_claim.flat"])
            else:
                raise ValueError(
                    "prompt examples with the revison label must have the revised_claim.flat field as well"
                )
        else:
            label.append("revised claim: none")
        if example.get("reason"):
            label.append(f"explanation: {example['reason']}")

        return "\n\n".join(label)

    @classmethod
    def convert_evidence_to_passages(cls, evidence):
        document_chunks = []
        for passage in sorted(evidence, key=lambda p: p["meta"]["start"]):
            start = int(passage["meta"]["start"])
            end = int(passage["meta"]["end"])
            document_chunks.append(
                f'<passage start="{start}" end="{end}">\n{passage["content"]}\n</passage>'
            )

        evidence = "\n".join(document_chunks)
        return evidence

    @classmethod
    def build_chat(cls, example):
        prompt = [cls.preface, ""]
        if len(example.get(EXAMPLES_COLNAME, [])) > 0:
            prompt.extend(["## Examples", ""])
            for ex_i, prompt_example in enumerate(example.get(EXAMPLES_COLNAME, [])):
                prompt.append(
                    f"""\
<claim id="example-{ex_i}">
{prompt_example["claim.flat"]}
</claim>

<evidence id="example-{ex_i}">
{cls.convert_evidence_to_passages(prompt_example["evidence"])}
</evidence>

<assistant_response id="example-{ex_i}">
{cls.build_label(prompt_example)}
</assistant_response>"""
                )

        prompt.append(
            f"""\
<claim id="target">
{example["claim.flat"]}
</claim>

<evidence id="target">
{cls.convert_evidence_to_passages(example["evidence"])}
</evidence>"""
        )

        prompt = "\n\n".join([p for p in prompt if p])

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": cls.build_label(example)},
        ]


class VerifyPassagesTrain(VerifyPassages):
    preface = ""


class VerifyEvidence(VerifyPassages):
    preface = """\
# Task

You are given a claim/evidence pair.

- The claim appears inside <claim> tags.
- The evidence appears inside <evidence> tags.
- The claim is a CIViC Evidence Item.
- The evidence is a series of one or more passages from a PubMed article or abstract.

Classify whether the evidence supports the claim.

Use only the provided evidence. Do not use outside biomedical knowledge except to interpret standard terminology, gene/protein names, variants, diseases, therapies, and study types.

# Input format

<claim>
...
</claim>

<evidence>
...
</evidence>

# Labels

Choose exactly one label:

## SUPPORTS

Use this when the evidence directly supports the claim as written.

Requirements:

- The key scientific assertions in the claim are supported.
- The evidence direction, significance, molecular profile, disease, therapy, and evidence type are supported when present.
- No material correction, narrowing, or qualification is needed.

## REVISIONS

Use this when the evidence is relevant and partially supports the claim, but the claim needs changes to be fully supported.

Use this label when:

- Some fields are supported but others are too broad, too specific, missing nuance, or incorrect.
- The claim could be made accurate with a focused revision.
- The revised claim should stay as close as possible to the original while removing or correcting unsupported parts.

## NEI

Use this when the claim is not supported by the evidence.

Use this label when:

- The article is unrelated to the claim.
- The evidence does not address the key claim.
- The evidence is only background or too indirect.
- The claim would require a substantially different claim rather than a revision.

# Decision rules

1. Check whether the evidence supports the claim’s core relationship, not just whether it mentions the same gene, variant, disease, or therapy.
2. Use SUPPORTS only when all material parts of the claim are supported.
3. Use REVISIONS when the article supports a narrower, corrected, or more precise version of the claim.
4. Use NEI when the evidence does not provide enough information to support or revise the claim.
5. Keep the explanation concise: 2–5 sentences.
6. Do not include reasoning steps, uncertainty scores, or extra sections.
7. Do not include any text before or after the required output fields.

# Output format

Return exactly the following fields, in this order:

label: SUPPORTS, REVISIONS, or NEI

revised claim: none, unless the label is REVISIONS. If the label is REVISIONS, provide a revised CIViC Evidence Item in JSON-like format.

explanation: 2–5 sentence explanation

# Output examples

## For SUPPORTS:

label: SUPPORTS

revised claim: none

explanation: The evidence directly supports the claim as written. The cited passages describe the relevant molecular profile, disease context, and scientific relationship without requiring material changes to the claim.

## For REVISIONS:

label: REVISIONS

revised claim:
{
  "evidenceType": "FUNCTIONAL",
  "significance": "gain-of-function",
  "molecularProfile": "KRAS G12D",
  "evidenceDirection": "SUPPORTS"
}

explanation: The evidence supports a functional interpretation for the molecular profile, but the original claim includes details that are too broad or not directly supported. The revised claim narrows the assertion to the relationship supported by the cited passages.

## For NEI:

label: NEI

revised claim: none

explanation: The evidence does not support the core claim. Although the article may mention related terminology, it does not provide evidence for the specific CIViC evidence relationship described in the claim.
"""

    @classmethod
    def build_chat(cls, example):
        prompt = [cls.preface, ""]
        if len(example.get(EXAMPLES_COLNAME, [])) > 0:
            prompt.extend(["## Examples", ""])
            for ex_i, prompt_example in enumerate(example.get(EXAMPLES_COLNAME, [])):
                prompt.append(
                    f"""\
<claim id="example-{ex_i}">
{prompt_example["claim.flat"]}
</claim>

<evidence id="example-{ex_i}">
{prompt_example["evidence.flat"]}
</evidence>

<assistant_response id="example-{ex_i}">
{cls.build_label(prompt_example)}
</assistant_response>"""
                )

        prompt.append(
            f"""\
<claim id="target">
{example["claim.flat"]}
</claim>

<evidence id="target">
{example["evidence.flat"]}
</evidence>"""
        )

        prompt = "\n\n".join([p for p in prompt if p])

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": cls.build_label(example)},
        ]


class VerifyEvidenceTrain(VerifyEvidence):
    preface = ""


class FullTextRetrieveAndVerifyLlama4(FullTextRetrieveAndVerify):
    # prompt to generate prompt in gpt5.5: below is a user prompt formatted for entry to gpt5.2. Rewrite the prompt to optimize performance for the llama 4 models instead
    preface = """\
You are a biomedical evidence-classification assistant.

Your task is to classify whether the provided PubMed evidence supports the provided CIViC Evidence Item claim.

Use only the evidence inside the <evidence> tags. Do not use outside biomedical knowledge, except to understand standard biomedical terminology, gene/protein names, variants, diseases, therapies, and study types.

Do not reveal your reasoning process. Think through the comparison internally, then return only the required output fields.

---

INPUT

The claim appears inside <claim> tags.

The evidence appears inside <evidence> tags.

The evidence is divided into passages. Each passage has a numeric id.

Example input structure:

<claim>
...
</claim>

<evidence>
<passage id="0">
...
</passage>
<passage id="1">
...
</passage>
...
</evidence>

---

TASK

Classify the claim using exactly one label:

SUPPORTS
REVISIONS
NEI

Evaluate whether the evidence supports the claim’s core scientific relationship, not merely whether it mentions the same gene, variant, disease, therapy, or study type.

Check all material claim components when present, including:

- molecular profile
- variant
- gene/protein
- disease
- therapy
- evidence type
- evidence direction
- significance
- clinical, functional, diagnostic, prognostic, predictive, or predisposing relationship

---

LABEL DEFINITIONS

SUPPORTS

Use SUPPORTS only when the evidence directly supports the claim as written.

Choose SUPPORTS when:

- The key scientific assertion is supported.
- The evidence direction is supported.
- The molecular profile is supported.
- The disease context is supported, if present.
- The therapy context is supported, if present.
- The evidence type and significance are supported.
- No material correction, narrowing, or qualification is needed.

Do not choose SUPPORTS if any important part of the claim is unsupported, overstated, too broad, too narrow, or incorrect.

---

REVISIONS

Use REVISIONS when the evidence is relevant and supports a corrected, narrowed, or more precise version of the claim.

Choose REVISIONS when:

- The article is relevant to the claim.
- The evidence supports the same general scientific relationship.
- Some material parts of the original claim are unsupported, overstated, too broad, too specific, missing nuance, or incorrect.
- The claim can be made accurate with a focused revision.
- The revised claim stays as close as possible to the original claim while removing or correcting unsupported parts.

The revised claim should include only fields supported by the evidence. Do not add unsupported details.

---

NEI

Use NEI when the evidence does not support the claim.

Choose NEI when:

- The article is unrelated to the claim.
- The evidence does not address the core scientific relationship.
- The evidence only provides background information.
- The evidence is too indirect.
- The article mentions related genes, variants, diseases, or therapies but does not support the specific claim.
- A substantially different claim would be needed.

Do not use REVISIONS when the evidence only has weak topical overlap. Use NEI instead.

---

PASSAGE SELECTION RULES

- Include passage IDs only for passages that are relevant to the classification.
- Order passage IDs from most relevant to least relevant.
- If multiple passages are relevant, include all relevant passage IDs.
- Do not include irrelevant passage IDs.
- If no passages are relevant, write exactly:

passages: none

---

OUTPUT RULES

Return exactly four fields, in this order:

label:
revised claim:
passages:
explanation:

Do not include any text before or after these fields.

The label field must be exactly one of:

label: SUPPORTS
label: REVISIONS
label: NEI

For SUPPORTS and NEI, write exactly:

revised claim: none

For REVISIONS, provide a revised CIViC Evidence Item in JSON-like format. The revised claim must stay close to the original and include only evidence-supported fields.

The explanation must be concise: 2–4 sentences.

Do not include:

- reasoning steps
- uncertainty scores
- confidence scores
- extra sections
- citations outside the passage IDs
- markdown headings
- commentary

---

OUTPUT FORMAT

For SUPPORTS:

label: SUPPORTS

revised claim: none

passages: 0, 3, 1

explanation: The evidence directly supports the claim as written. The cited passages describe the relevant molecular profile, disease context, and scientific relationship without requiring material changes.

For REVISIONS:

label: REVISIONS

revised claim:
{
  "evidenceType": "FUNCTIONAL",
  "significance": "gain-of-function",
  "molecularProfile": "KRAS G12D",
  "evidenceDirection": "SUPPORTS"
}

passages: 0, 6, 2

explanation: The evidence supports a functional interpretation for the molecular profile, but the original claim includes details that are too broad or not directly supported. The revised claim narrows the assertion to the relationship supported by the cited passages.

For NEI:

label: NEI

revised claim: none

passages: none

explanation: The evidence does not support the core claim. Although the article may mention related terminology, it does not provide evidence for the specific CIViC evidence relationship described in the claim.
"""


TEMPLATES: "dict[str, Template]" = {
    "none": Template(),
    "basic": BasicTemplate(),
    "basicfew": BasicFewTemplate(),
    "reason": ReasonTemplate(),
    "reason2": ReasonTemplate2(),
    "summary": FullTextSummaryTemplate(),
    "claim": ClaimOnlyTemplate(),
    "labelfirst": LabelFirstTemplate(),
    "verbose": VerboseTemplate(),
    "openaimd": VerboseOpenAIMdTemplate(),
    "openaimdlimit100": OpenAIMdTemplateLimit(),
    "objective": ObjectiveTemplate(),
    "onlylabels": OnlyLabelsTemplate(),
    "claimonlylabelfirst": ClaimOnlyLabelFirstTemplate(),
    "verboselimit100": VerboseLimit100Template(),
    "labels": LabelsTemplate(),
    "reasonlimit100": ReasonLimit100(),
    "jsontrain": JsonTrain(),
    "jsonlimit200": JsonLimit200(),
    "jsonexampleslimit200": JsonLimit200Examples(),
    "jsonexampleslimit200verbose": JsonLimit200ExamplesVerbose(),
    "jsonexampleslimit200synonyms": JsonLimit200ExamplesSynonyms(),
    "fulltextretrieveandverify": FullTextRetrieveAndVerify(),
    "verifypassages": VerifyPassages(),
    "verifypassagestrain": VerifyPassagesTrain(),
    "verifyevidence": VerifyEvidence(),
    "verifyevidencetrain": VerifyEvidenceTrain(),
}

CHAT_TEMPLATES: "dict[str, ChatTemplate]" = {
    "mpt": ChatTemplate(
        template=re.sub(
            r"\n\s*",
            "",
            """\
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' %}
{% endif %}

{% for message in loop_messages %}
    {% if loop.index0 == 0 and system_message != false %}
        {{ system_message + '\\n' }}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '### Instruction:\\n' + message['content'].strip() }}
    {% elif message['role'] == 'assistant' %}
        {{ '\\n### Response:\\n'  + message['content'].strip() }}
    {% endif %}

    {% if message['role'] == 'user' and loop.last and add_generation_prompt %}
        {{ '\\n### Response:\\n' }}
    {% endif %}
{% endfor %}
""",
        ),
        generation_prompt="### Response:\n",
    ),
    "chat-doctor": ChatTemplate(
        template=re.sub(
            r"\n\s*",
            "",
            """\
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = false %}
{% endif %}

{% for message in loop_messages %}
    {% if loop.index0 == 0 and system_message != false %}
        {{ system_message + '\\n' }}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '###Question: ' + message['content'].strip() }}
    {% elif message['role'] == 'assistant' %}
        {{ '\\n###Answer: '  + message['content'].strip() }}
    {% endif %}

    {% if message['role'] == 'user' and loop.last and add_generation_prompt %}
        {{ '\\n###Answer: ' }}
    {% endif %}
{% endfor %}
""",
        ),
        generation_prompt="###Answer: ",
    ),
    "llama-3-chat": ChatTemplate(
        # from https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/llama-3-chat.jinja
        re.sub(
            r"\n\s*",
            "",
            """\
{{ bos_token }}
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = '<|start_header_id|>' + 'system' + '<|end_header_id|>\\n\\n' + messages[0]['content'].strip() + '<|eot_id|>' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}

{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/bot/user/bot/...') }}
    {% endif %}

    {% if loop.index0 == 0 %}
        {{ system_message }}
    {% endif %}

    {{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'].strip() + '<|eot_id|>' }}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\\n\\n' }}
    {% endif %}
{% endfor %}""",
        ),
        generation_prompt="<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "phi-3": ChatTemplate(
        re.sub(
            r"\n\s*",
            "",
            """\
{{ bos_token }}
{% for message in messages %}
    {{'<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n'}}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|assistant|>\\n' }}
{% endif %}""",
        ),
        generation_prompt="<|end|>\n<|assistant|>\n",
    ),  # https://github.com/unslothai/unsloth/issues/398#issuecomment-2123408734
}
