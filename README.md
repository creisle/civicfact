# CIViC-Fact

This repository contains the code and data related to the CIViC-Fact project. 

See the [dataset documentation](data_builder/README.md#using-the-released-dataset) for download information, data formats, and examples of loading the released datasets.

The repository is structured as follows
- data_builder: workflow code for downloading, preprocessing, generating negatives, and analyzing the CIViC-Fact dataset
- data_builder/builds/*: jsonl.gz downloads of the CIViC-Fact dataset
- experiments: code for testing and training language models on CIViC-Fact
- manual_review: data from rounds of manual review of the dataset and fact-checking pipeline
