import pytest

from utils import (
    compute_overlap,
    convert_with_template,
    get_predicted_label,
    parse_filepath_metadata,
)


@pytest.mark.parametrize(
    "text,label",
    [
        ("The evidence does not support the claim", "REFUTES"),
        ("label the evidence as REFUTES", "REFUTES"),
        (
            "The evidence suggests that KRAS G12D confers resistance to dabrafenib in patients with V600E-positive NSCLC, which supports the claim",
            "SUPPORTS",
        ),
        (
            "Upregulation of AKT3 was associated with resistance in a breast cancer model suggesting the evidence refutes claim",
            "REFUTES",
        ),
        ("The evidence is NEI because:", "NEI"),
        ("2. REFUTES", "REFUTES"),
        ("✗ REFUTES ✗", "REFUTES"),
        ("teacher: SUPPORTS", "SUPPORTS"),
        ("- Support: SUPPORTS", "SUPPORTS"),
        ("label: label: REFUTES;", "REFUTES"),
        ("reply: SUPPORTS", "SUPPORTS"),
        ("stance toward claim: REFUTES", "REFUTES"),
        ("- answer: SUPPORTS", "SUPPORTS"),
        ("Answer: SUPPORTS.", "SUPPORTS"),
        (
            "Based on the given evidence, the stance toward the claim is SUPPORTS",
            "SUPPORTS",
        ),
        (
            "Based on the given evidence, the stance toward the claim is REFUTES.",
            "REFUTES",
        ),
        ("The evidence does NOT REFUTE the claim", ""),
        (
            "The evidence label for the claim is NEI (not enough information) because the given claim does not meet the requirements for all the evidence fields.",
            "NEI",
        ),
        ("The evidence provided does not directly support or refute the claim", "NEI"),
        (
            "Based on the given claim and evidence, the label for the relationship between them is SUPPORTS.",
            "SUPPORTS",
        ),
        (
            "Based on the given claim, evidence, and labels, I would choose the label SUPPORTS.",
            "SUPPORTS",
        ),
        ("refute", "REFUTES"),
        ("support", "SUPPORTS"),
        ("RESPONSES: SUPPORTS", "SUPPORTS"),
        ("refuting: The claim suggests poor outcome", "REFUTES"),
        ("The evidence neither supports nor refutes the claim", "NEI"),
        (
            "The evidence does not provide enough information to support the claim",
            "NEI",
        ),
        (
            "There is not enough information to determine the stance of the evidence toward the claim",
            "NEI",
        ),
        ("RESPONDS: SUPPORTS", "SUPPORTS"),
        (
            "The evidence is NEI (not enough information) because it only describes the presence of the mutation, so it cannot be labeled as SUPPORTS or REFUTES.",
            "NEI",
        ),
    ],
)
def test_get_predicted_label(text, label):
    assert label == get_predicted_label(text)


@pytest.mark.parametrize(
    "filepath,expected_elements",
    [
        (
            "results/claim_verification/pretokenize/BioMistral__BioMistral-7B-DARE/civicfact-2025.01.14/chat-default/prompts.claim.0-shot.train.processed.jsonl.gz",
            {"model_dir": "BioMistral__BioMistral-7B-DARE"},
        ),
        (
            "../results/evidence_retrieval/cross/evaluations/BAAI__bge-reranker-large/civicfactpairs-2026.03.03/qt0/data.dev.retrieval_accuracy.jsonl.gz",
            {
                "query_template": "0",
                "model_dir": "BAAI__bge-reranker-large",
                "task": "evidence_retrieval",
            },
        ),
        (
            "results/evidence_retrieval/cross/fine-tuning/anchors/config-7/civicfact-2026.03.03/qt0/ncbi__MedCPT-Cross-Encoder/evaluations/civicfactpairs-2026.03.03/qt0/data.dev.retrieval_accuracy.jsonl.gz",
            {
                "model_dir": "ncbi__MedCPT-Cross-Encoder",
                "train_data_format": "anchors",
                "query_template": "0",
                "train_query_template": "0",
                "partition": "dev",
            },
        ),
        (
            "../results/claim_verification/fine-tuning/config-8/completion/civicfact-2026.03.03/chat-default/prompts.jsonlimit200.0-shot.train.jsonl/meta-llama__Meta-Llama-3.1-8B-Instruct/evaluations/civicfactpairs-grand-bat-L5000-qt0-2026.03.03/chat-default/gen-5/prompts.jsonlimit200.0-shot.dev.processed.label_accuracy.jsonl.gz",
            {"eval_dataset": "civicfactpairs", "ret_query_template": "0"},
        ),
        (
            "results/evidence_retrieval/bi/evaluations/bm25-lucene/civicfactpairs-2026.03.03/qt0/data.dev.retrieval_accuracy.jsonl.gz",
            {"model_dir": "bm25-lucene", "query_template": "0"},
        ),
        (
            "../results/evidence_retrieval/cross/evaluations/Qwen__Qwen3-Reranker-4B/civicfactpairs-2026.03.03/qt3/data.dev.retrieval_accuracy.jsonl.gz",
            {"partition": "dev"},
        ),
        (
            "../results/evidence_retrieval/cross/evaluations/BAAI__bge-reranker-large/civicfactpairs-2026.03.03/qt3/data.dev.retrieval_accuracy.jsonl.gz",
            {"partition": "dev"},
        ),
        (
            "results/claim_verification/api/evaluations/openai__gpt-5.2/civicfactft_f0.01_s42-2026.03.03/chat-default/gen-8/fulltextretrieveandverify.0-shot/qt0/data.dev.retrieval_accuracy.jsonl.gz",
            {
                "model_dir": "openai__gpt-5.2",
                "query_template": "0",
                "generation_config": "gen-8",
            },
        ),
        (
            "results/evidence_retrieval/cross/evaluations/BAAI__bge-reranker-large",
            {"model_dir": "BAAI__bge-reranker-large"},
        ),
        (
            "results/claim_verification/fine-tuning/config-15/completion/civicfact-2026.03.03/chat-default/prompts.jsonlimit200.0-shot.train.jsonl/meta-llama__Llama-3.2-1B-Instruct/evaluations/civicfactpairs-random42-L10000-qt0-2026.03.03/chat-default/gen-5/prompts.jsonlimit200.0-shot.dev.processed.jsonl.gz",
            {"ret_model": "random42"},
        ),
        (
            "../results/claim_verification/fine-tuning/config-15/completion/civicfactxf1.0g8-2026.03.03/chat-default/prompts.verifyevidencetrain.0-shot.train.jsonl/meta-llama__Llama-3.2-3B-Instruct/checkpoints/checkpoint-1209/trainer_state.json",
            {"train_dataset": "civicfactxf1.0g8"},
        ),
        (
            "results/claim_verification/api/evaluations/openai__gpt-5.2/civicfactpairs-random42-L5000-qt0_f0.50_hard_s42-2026.03.03/chat-default/gen-4/prompts.jsonexampleslimit200.3-shot.dev.processed.jsonl.gz",
            {
                "eval_dataset": "civicfactpairs",
                "ret_model": "random42",
                "subsample_filter": "hard",
                "subsample_fraction": "0.5",
                "subsample_seed": "42",
            },
        ),
        (
            "results/claim_verification/api/evaluations/openai__gpt-5.2/civicfactpairs-random42-L5000-qt0_f0.5_s42-2026.03.03/chat-default/gen-4/prompts.jsonexampleslimit200.3-shot.dev.processed.jsonl.gz",
            {
                "eval_dataset": "civicfactpairs",
                "ret_model": "random42",
                "subsample_fraction": "0.5",
                "subsample_seed": "42",
            },
        ),
        (
            "results/claim_verification/fine-tuning/config-15/completion/civicfactxf1.0g8-2026.03.03/chat-default/prompts.verifyevidencetrain.0-shot.train.jsonl/meta-llama__Llama-3.2-3B-Instruct/evaluations/civicfactpairs-head-oracle-L5000-qt0_f1.0_hard_s42-2026.03.03/chat-default/gen-5/prompts.verifyevidencetrain.0-shot.dev.processed.jsonl.gz",
            {
                "eval_dataset": "civicfactpairs",
                "subsample_fraction": "1",
                "subsample_seed": "42",
            },
        ),
    ],
)
def test_parse_filepath_metadata(filepath, expected_elements):
    print(filepath)
    meta = parse_filepath_metadata(filepath)
    for k, v in sorted(meta.items()):
        print(k, v)
    for elem, val in expected_elements.items():
        assert elem in meta
        assert meta[elem] == val


@pytest.mark.parametrize(
    "claim,text",
    [
        (
            {
                "disease": "Juvenile Polyposis Syndrome",
                "evidenceDirection": "SUPPORTS",
                "evidenceType": "PREDISPOSING",
                "molecularProfile": "SMAD4 R361C",
                "significance": "predisposition",
                "variantOrigin": "RARE_GERMLINE",
            },
            "The rare germline variant, SMAD4 R361C, is associated with predisposition to Juvenile Polyposis Syndrome",
        ),
        (
            {
                "disease": "Lung Non-small Cell Carcinoma",
                "evidenceDirection": "SUPPORTS",
                "evidenceType": "PREDICTIVE",
                "molecularProfile": "ETS2 RS461155",
                "significance": "sensitivity/response",
                "therapies": ["Paclitaxel", "Cisplatin"],
                "therapyInteractionType": "COMBINATION",
                "variantOrigin": "COMMON_GERMLINE",
            },
            "The common germline variant, ETS2 RS461155, is predictive of sensitivity/response to a combination of Paclitaxel, and Cisplatin in Lung Non-small Cell Carcinoma",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "predictive",
                "significance": "sensitivity",
                "evidenceDirection": "SUPPORTS",
                "disease": "cancer",
                "therapies": ["drug"],
            },
            "MUT is predictive of sensitivity to drug in cancer",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "predictive",
                "significance": "sensitivity",
                "disease": "cancer",
                "therapies": ["drug A", "drug B"],
                "evidenceDirection": "SUPPORTS",
                "therapyInteractionType": "COMBINATION",
            },
            "MUT is predictive of sensitivity to a combination of drug A, and drug B in cancer",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "predictive",
                "significance": "sensitivity",
                "disease": "cancer",
                "therapies": ["drug A", "drug C", "drug B"],
                "therapyInteractionType": "SUBSTITUTES",
                "evidenceDirection": "SUPPORTS",
            },
            "MUT is predictive of sensitivity to any of drug A, drug C, or drug B in cancer",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "prognostic",
                "significance": "poor outcome",
                "disease": "cancer",
                "evidenceDirection": "SUPPORTS",
            },
            "MUT is associated with a prognosis of poor outcome in cancer",
        ),
        (
            {
                "molecularProfile": "MUT or MUT",
                "evidenceType": "prognostic",
                "significance": "poor outcome",
                "disease": "cancer",
                "evidenceDirection": "SUPPORTS",
                "variantOrigin": "COMMON_GERMLINE",
            },
            "MUT or MUT common germline variants are associated with a prognosis of poor outcome in cancer",
        ),
        (
            {
                "molecularProfile": "MUT or MUT",
                "evidenceType": "prognostic",
                "significance": "poor outcome",
                "disease": "cancer",
                "evidenceDirection": "SUPPORTS",
            },
            "MUT or MUT are associated with a prognosis of poor outcome in cancer",
        ),
        (
            {
                "molecularProfile": "MUT and MUT",
                "variantOrigin": "COMMON_GERMLINE",
                "evidenceType": "prognostic",
                "significance": "poor outcome",
                "disease": "cancer",
                "evidenceDirection": "SUPPORTS",
            },
            "a molecular profile consisting of MUT and MUT common germline variants is associated with a prognosis of poor outcome in cancer",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "diagnostic",
                "significance": "positive",
                "disease": "cancer",
                "evidenceDirection": "SUPPORTS",
            },
            "MUT is a positive diagnostic indicator in cancer",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "predisposing",
                "significance": "predisposition",
                "disease": "cancer",
                "evidenceDirection": "SUPPORTS",
            },
            "MUT is associated with predisposition to cancer",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "functional",
                "significance": "unaltered function",
                "evidenceDirection": "SUPPORTS",
            },
            "MUT results in unaltered function",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "functional",
                "significance": "neomorphic",
                "evidenceDirection": "SUPPORTS",
            },
            "MUT results in neomorphic functionality",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "functional",
                "significance": "dominant negative",
                "evidenceDirection": "SUPPORTS",
            },
            "MUT results in a dominant negative",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "functional",
                "significance": "dominant negative",
                "evidenceDirection": "DOES_NOT_SUPPORT",
            },
            "MUT does not result in a dominant negative",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "predisposing",
                "evidenceDirection": "SUPPORTS",
                "significance": "pathogenic",
                "disease": "disease",
            },
            "MUT is pathogenic in disease",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "predisposing",
                "evidenceDirection": "SUPPORTS",
                "significance": "uncertain significance",
                "disease": "disease",
            },
            "MUT has uncertain significance in disease",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "predisposing",
                "evidenceDirection": "DOES_NOT_SUPPORT",
                "significance": "uncertain significance",
                "disease": "disease",
            },
            "MUT does not have uncertain significance in disease",
        ),
        (
            {
                "molecularProfile": "MUT",
                "evidenceType": "predictive",
                "significance": "sensitivity",
                "evidenceDirection": "SUPPORTS",
                "disease": "cancer",
                "therapies": ["drug1", "drug2"],
                "therapyInteractionType": "SUBSTITUTES",
            },
            "MUT is predictive of sensitivity to drug1, or drug2 in cancer",
        ),
        (
            {
                "disease": "Renal Cell Carcinoma",
                "evidenceDirection": "SUPPORTS",
                "evidenceType": "PREDICTIVE",
                "molecularProfile": "FLCN c.1285dupC",
                "phenotypes": [
                    "Spontaneous pneumothorax",
                    "Fibrofolliculoma",
                    "Pulmonary cyst",
                ],
                "significance": "sensitivity/response",
                "therapies": ["Everolimus"],
                "variantOrigin": "RARE_GERMLINE",
            },
            "The rare germline variant, FLCN c.1285dupC, is predictive of sensitivity/response to Everolimus in Renal Cell Carcinoma accompanied by the Spontaneous pneumothorax, Fibrofolliculoma, Pulmonary cyst phenotypes",
        ),
        (
            {
                "disease": "Renal Cell Carcinoma",
                "evidenceDirection": "SUPPORTS",
                "evidenceType": "PREDICTIVE",
                "molecularProfile": "FLCN c.1285dupC",
                "phenotypes": ["Spontaneous pneumothorax"],
                "significance": "sensitivity/response",
                "therapies": ["Everolimus"],
                "variantOrigin": "RARE_GERMLINE",
            },
            "The rare germline variant, FLCN c.1285dupC, is predictive of sensitivity/response to Everolimus in Renal Cell Carcinoma accompanied by the Spontaneous pneumothorax phenotype",
        ),
        (
            {
                "disease": "Renal Cell Carcinoma",
                "evidenceDirection": "DOES_NOT_SUPPORT",
                "evidenceType": "PREDICTIVE",
                "molecularProfile": "FLCN c.1285dupC",
                "phenotypes": ["Spontaneous pneumothorax"],
                "significance": "sensitivity/response",
                "therapies": ["Everolimus"],
                "variantOrigin": "RARE_GERMLINE",
            },
            "The rare germline variant, FLCN c.1285dupC, is not predictive of sensitivity/response to Everolimus in Renal Cell Carcinoma accompanied by the Spontaneous pneumothorax phenotype",
        ),
    ],
)
def test_convert_with_template(claim, text):
    assert convert_with_template(claim) == text


@pytest.mark.parametrize(
    "range1,range2,overlap",
    [
        [(500, 600), (501, 502), 1],
        [(500, 600), (499, 502), 2],
        [(500, 600), (400, 401), 0],
        [(400, 401), (500, 600), 0],
        [(5400, 6400), (5400, 6400), 1000],
    ],
)
def test_compute_overlap(range1, range2, overlap):
    assert compute_overlap(range1, range2) == overlap
    assert compute_overlap(range2, range1) == overlap
