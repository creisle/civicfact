source .venv/bin/activate

# python src/doccano/convert_dataset_to_html.py \
#     --civic_dump ../data_builder/builds/civicfact-2025.03.25/civic-dump.jsonl.gz \
#     ../data_builder/builds/civicreview-2025.03.25v1/data.jsonl.gz \
#     data/doccano/2025.11.17_civicreview_v2025.03.25/inputs/doccano_input.v1.jsonl.gz \
#     --metadata civic.id gold_label_name document.content_source \
#     --acronyms ../data_builder/results/civic/acronyms.jsonl

# python src/doccano/convert_dataset_to_html.py \
#     --civic_dump ../data_builder/builds/civicfact-2025.03.25/civic-dump.jsonl.gz \
#     ../data_builder/builds/civicreview-2025.03.25v2/data.jsonl.gz \
#     data/doccano/2025.11.17_civicreview_v2025.03.25/inputs/doccano_input.jsonl.gz \
#     --metadata civic.id gold_label_name document.content_source \
#     --acronyms ../data_builder/results/civic/acronyms.jsonl \
#     --label_column status_labels

# python src/doccano/convert_dataset_to_html.py \
#     --civic_dump ../data_builder/builds/civicfact-2025.03.25/civic-dump.jsonl.gz \
#     ../data_builder/builds/civicreview-2025.03.25v3/nonredundant.jsonl.gz \
#     data/doccano/2025.11.17_civicreview_v2025.03.25/inputs/doccano_input.v3.jsonl.gz \
#     --metadata civic.id gold_label_name document.content_source \
#     --acronyms ../data_builder/results/civic/acronyms.jsonl
