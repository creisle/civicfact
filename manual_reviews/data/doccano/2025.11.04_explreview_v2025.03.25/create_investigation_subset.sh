set -e
ACTIVE_DIR=data/doccano/2025.11.04_explreview_v2025.03.25
REPO_DIR=/projects/creisle_prj/git/civicfact
SUBSET_FILE=$REPO_DIR/data_builder/builds/explreview-2025.03.25/data.jsonl.gz

python src/doccano/create_response_review_input.py \
    --subset_selection $SUBSET_FILE \
    --inputs $REPO_DIR/experiments/results/claim_verification/evaluations/meta-llama__Meta-Llama-3.1-70B-Instruct/explreview-2025.03.25/chat-default/gen-2/prompts.openaimdlimit100.0-shot.{dev,train}.processed.jsonl.gz \
    --inputs $REPO_DIR/experiments/results/claim_verification/evaluations/mistralai__Mixtral-8x7B-Instruct-v0.1/explreview-2025.03.25/chat-default/gen-2/prompts.openaimdlimit100.0-shot.{dev,train}.processed.jsonl.gz \
    --inputs $REPO_DIR/experiments/results/claim_verification/api/evaluations/openai__gpt-4.1/explreview-2025.03.25/chat-default/gen-2/prompts.openaimdlimit100.0-shot.{dev,train}.processed.jsonl.gz \
    --num_expected_responses 3 \
    --response_per_file 1 \
    --output \
    $ACTIVE_DIR/inputs/detailed_responses.jsonl.gz \
    $ACTIVE_DIR/inputs/doccano_input.jsonl.gz
