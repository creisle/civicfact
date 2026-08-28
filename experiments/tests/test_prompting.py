import pytest
from transformers import AutoTokenizer

import prompting


@pytest.fixture
def mock_examples():
    examples = {}

    for i in range(6):
        examples[i] = {
            "claim.flat": f"c{i}",
            "evidence.flat": f"e{i}",
            "gold_label_name": f"l{i}",
            "reason": f"r{i}",
            "summary": f"s{i}",
        }
    return examples


@pytest.mark.parametrize(
    "template,example_id,prompt_example_ids,text,label",
    [
        ("none", 0, [], "claim: c0\nevidence: e0\nlabel: ", "l0"),
        ("none", 0, [1], "claim: c1\nevidence: e1\nlabel: l1\n\nclaim: c0\nevidence: e0\nlabel: ", "l0"),
        (
            "basic",
            0,
            [],
            """\
Given a claim and some evidence, determine the stance of the evidence toward the claim. Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information).

claim: c0
evidence: e0
label: """,
            "l0",
        ),
        (
            "basic",
            0,
            [1, 2],
            """\
Given a claim and some evidence, determine the stance of the evidence toward the claim. Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information).

claim: c1
evidence: e1
label: l1

claim: c2
evidence: e2
label: l2

claim: c0
evidence: e0
label: """,
            "l0",
        ),
        (
            "reason",
            0,
            [],
            """\
Given a claim and some evidence, determine the stance of the evidence toward the claim. Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information). Give your reasoning.

claim: c0
evidence: e0
label: """,
            "l0. r0",
        ),
        (
            "reason",
            0,
            [1],
            """\
Given a claim and some evidence, determine the stance of the evidence toward the claim. Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information). Give your reasoning.

claim: c1
evidence: e1
label: l1. r1

claim: c0
evidence: e0
label: """,
            "l0. r0",
        ),
        (
            "summary",
            0,
            [1],
            """\
Given a claim and some evidence, summarize the relevant parts of the evidence and then determine the stance of the evidence toward the claim. Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information). Give your reasoning.

claim: c1
evidence: e1
summary: s1
label: l1. r1

claim: c0
evidence: e0""",
            "summary: s0\nlabel: l0. r0",
        ),
        (
            "reason2",
            0,
            [],
            """\
Given a claim and some evidence, determine the stance of the evidence toward the claim. Use the following labels: SUPPORTS, REFUTES, and NEI (not enough information). Give your reasoning.

claim: c0
evidence: e0
""",
            "label: l0\nreasoning: r0",
        ),
    ],
)
def test_build_prompt(template, example_id, prompt_example_ids, text, label, mock_examples):
    result = prompting.TEMPLATES[template].build_chat(
        {
            **mock_examples[example_id],
            "prompt_examples": [mock_examples[i] for i in prompt_example_ids],
        }
    )
    print(result)
    assert result[0]['content'] == text

    label = prompting.TEMPLATES[template].build_label(
        {
            **mock_examples[example_id],
        }
    )
    assert label == label


@pytest.mark.parametrize(
    "template,example_id,prompt_example_ids,text",
    [
        ("none", 0, [], "label: l0"),
        ("basic", 0, [], "label: l0"),
        ("reason", 0, [], "label: l0. r0"),
        ("reason2", 0, [], "label: l0\nreasoning: r0"),
        ("summary", 0, [], "summary: s0\nlabel: l0. r0"),
    ],
)
def test_build_label(template, example_id, prompt_example_ids, text, mock_examples):
    result = prompting.TEMPLATES[template].build_label(
        {
            **mock_examples[example_id],
            "prompt_examples": [mock_examples[i] for i in prompt_example_ids],
        }
    )
    print(result)
    assert result == text


@pytest.fixture
def chat():
    return [
        {"role": "system", "content": "STUFF TO SAY AT START"},
        {"role": "user", "content": "some question?"},
        {"role": "assistant", "content": "The answer."},
    ]


@pytest.mark.parametrize(
    "template_name,model,exp_result",
    [
        (
            "llama-3-chat",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

STUFF TO SAY AT START<|eot_id|><|start_header_id|>user<|end_header_id|>

some question?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The answer.<|eot_id|>""",
        ),
    ],
)
def test_apply_chat_template(template_name, model, exp_result, chat):
    tokenizer = AutoTokenizer.from_pretrained(model)
    if template_name in prompting.CHAT_TEMPLATES:
        template = prompting.CHAT_TEMPLATES[template_name].template
    else:
        template = tokenizer.default_chat_template
    actual_result = tokenizer.apply_chat_template(chat, chat_template=template, tokenize=False)
    print(actual_result)
    assert exp_result == actual_result
