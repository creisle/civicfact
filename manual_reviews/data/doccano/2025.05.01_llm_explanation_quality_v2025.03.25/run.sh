# download doccano data
set -e

# format the initial review data dump to look like API data
python data/doccano/2025.05.01_llm_explanation_quality_v2025.03.25/review_1_initial/format.py

# download the re-examined data
python src/doccano/download_examples_data.py "CIViC-Fact full pipeline review - LLM responses" data/doccano/2025.05.01_llm_explanation_quality_v2025.03.25/review_2_comments/examples.jsonl.gz --passwords .doccano_users.json
python src/doccano/download_examples_data.py "REVIEW - LLM responses" data/doccano/2025.05.01_llm_explanation_quality_v2025.03.25/review_3_explanations/examples.jsonl.gz --passwords .doccano_users.json

# pull out the gold labels from the various doccano projects and compute the consensus
python data/doccano/2025.05.01_llm_explanation_quality_v2025.03.25/extract_gold_labels.py

# pull out the span annotations
python data/doccano/2025.05.01_llm_explanation_quality_v2025.03.25/extract_explanation_tags.py
