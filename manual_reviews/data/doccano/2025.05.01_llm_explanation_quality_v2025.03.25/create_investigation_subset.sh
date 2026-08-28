set -e
ACTIVE_DIR=data/doccano/2025.05.01_llm_explanation_quality_v2025.03.25

SUBSET_FILE=$ACTIVE_DIR/subset_selection.jsonl.gz

if [ ! -f "$SUBSET_FILE" ];
then
    python src/doccano/select_review_subset.py \
        --inputs /projects/creisle_prj/git/llm_testing/results/claim_verification/fine-tuning/config-8/completion/civicfact-2025.03.25/chat-default/prompts.labelfirst.0-shot.train.jsonl/meta-llama__Meta-Llama-3.1-8B-Instruct/evaluations/civicfactpairs-grand-bat-L5000-2025.03.25/chat-default/gen-2/prompts.labelfirst.0-shot.{dev,train}.processed.jsonl.gz \
        --prop_train 0.6 \
        --prop_ft 0.75 \
        --max_review_count 1 \
        --ref_data builds/civicfact-2025.03.25/data.jsonl.gz \
        --output $SUBSET_FILE \
        --max_annotations_per_user 20 \
        --users 11 \
        --annotation_overlap 2 \
        --preferred_examples $ACTIVE_DIR/preferred_examples.jsonl

else
    echo "subset file already exists, skipping generating"
fi

if [ ! -f "${SUBSET_FILE/subset_selection/subset_selection.responses}" ];
then
    python src/doccano/create_response_review_input.py \
        --subset_selection $SUBSET_FILE \
        --inputs /projects/creisle_prj/git/llm_testing/results/claim_verification/fine-tuning/config-8/completion/civicfact-2025.03.25/chat-default/prompts.labelfirst.0-shot.train.jsonl/meta-llama__Meta-Llama-3.1-8B-Instruct/evaluations/civicfactpairs-grand-bat-L5000-2025.03.25/chat-default/gen-2/prompts.labelfirst.0-shot.{dev,train}.processed.jsonl.gz \
        --inputs /projects/creisle_prj/git/llm_testing/results/claim_verification/fine-tuning/config-8/completion/civicfact-2024.07.09/chat-default/prompts.reason.0-shot.train.jsonl/meta-llama__Meta-Llama-3.1-70B-Instruct/evaluations/civicfactpairs-grand-bat-L5000-2025.03.25/chat-default/gen-2/prompts.labelfirst.0-shot.{dev,train}.processed.jsonl.gz \
        --inputs $ACTIVE_DIR/subset_selection.gemini-2.5-pro-preview-03-25.jsonl.gz \
        --inputs $ACTIVE_DIR/subset_selection.gpt4.1.jsonl.gz \
        --inputs /projects/creisle_prj/git/llm_testing/results/claim_verification/evaluations/meta-llama__Meta-Llama-3.1-70B-Instruct/civicfactpairs-grand-bat-L5000-2025.03.25/chat-default/gen-2/prompts.labelfirst.0-shot.{dev,train}.processed.jsonl.gz \
        --num_expected_responses 5 \
        --response_per_file 1 \
        --output \
        ${SUBSET_FILE/subset_selection/subset_selection.responses} \
        ${SUBSET_FILE/subset_selection/subset_selection.doccano_input}
fi
