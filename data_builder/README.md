# CIViC-Fact Claim Verification Dataset

- [Data Format](#data-format)
  - [Evidence](#evidence)
  - [Claim](#claim)
  - [Reviews](#reviews)
- [Data Preprocessing](#data-preprocessing)
  - [Data Cleaning](#data-cleaning)
  - [Data Augmentation](#data-augmentation)
- [Getting Started Building the Dataset (Developers)](#getting-started-building-the-dataset-developers)
- [Workflow](#workflow)
- [Data License](#data-license)

The CIViC-Fact data set is a companion dataset to the CIViC database. For the purposes of this dataset which consider each "Evidence Item" from CIViC to be a claim. We provide evidence for each claim via selected excerpts of text from the article the claim was derived from. These selections were collected by a group of expert curators using the web tool hypothes.is. An example is given below. Here the pertinent fields of the claim are shown as a table (tab-delimited in the data files) and any relevant excerpts are concatenated via ellipsis. The dataset is published in JSONL format.

claim

| molecular profile | evidence type| significance | therapies | diseases |
|---|---|---|---|---|
| EGFR::RAD51 Fusion  | predictive | sensitivity/response   | Erlotinib |  Lung Adenocarcinoma|

evidence

> Patient 1, a 35-year-old female, was diagnosed with metastatic lung adenocarcinoma after presenting with generalized weakness and worsening vision. . . . A lymph node biopsy was sent for genomic profiling using an extensively validated hybrid capture-based NGS diagnostic platform (FoundationOne(r)) and found to harbor a novel EGFR rearrangement at exon 25, resulting in the formation of a fusion gene between EGFR and RAD51 . . . Within two weeks of erlotinib initiation, DIC had resolved (Supplementary Fig. S1A) and the patient experienced clinical improvement with a noticeable decrease in supraclavicular lymphadenopathy and a hard palate metastatic lesion. After six months of treatment, the primary left lung mass and largest two liver lesions had decreased by 69% per RECIST (Fig. 1C, Supplementary Fig. S1B), and the patient experienced an improvement in her functional status.

There are several steps in pre-processing the data we have done to both clean and augment the dataset. We aimed to keep as much metadata as possible in the dataset to increase utlity for downstream users.

The data builds themselves are found under the builds directory versioned by the date they were generated

## Data Format

A description of each field in the data files

If the claim maps directly to a CIViC evidence item, then the following fields are copied directly from that CIViC evidence item to include in the data file for posterity at the time of the build

- civic.description
- civic.evidenceItem
- civic.flagged `bool`
- civic.id
- civic.status `str`: ACCEPTED, REJECTED, or SUBMITTED

Fields that directly relate to the article metadata are below

- document.doi `str`
- document.license `str`: The license status in PMC
- document.pmcid `str`: PMC ID
- document.pmid `str`: pubmed ID
- document.title `str`

Remaining Fields

- claim `List[TEXT]`: The tabular claim from civic as a structured element containing metadata
- claim.flat `str`: The flattened claim
- claim.generated `bool`: If this field is true, the claim was generated as part of data augmentation and does not necessiarly map to a particular item in CIViC
- claim.id: generated id unique to this claim
- claim.source `str`: The source the claim was dervived from, generally CIViC for non-augmented data
- conversation: A representation of the comment history from hypothes.is
- display_id `UUID`: generated UUID
- evidence `List[EVIDENCE]`: The evidence as a structured element containing metadata
- evidence.flat `str`: The flattened evidence
- evidence.generated `bool`: If true, the evidence was not selected by a human curator via hypothes.is but is augmented data
- evidence.id `int`: generated id unique to this combination of evidence
- evidence.selection_method `str`: how evidence was selected
- evidence.source `str`: The source used to get the article content
- example `str`: When there are multiple possible sets of exceprts that could be evidence for a particular claim we differentiate possible different combinations with an example number
- flagged `bool`: If this claim/evidence pair was flagged either in civic or hypothesis. Indicates a possible error in the data and should be reviewed or used more carefully
- generated `bool`: True when the data was generated as part of data augmentation
- gold_label_name `str`: One of three possible labels: SUPPORTS, REVISIONS, or NEI indicating the stance of the evidence toward the claim
- hypothesis.tag: Tags from the original hypothesis file
- hypothesis.username `str`: The name of the user who curated the hypothes.is selections
- id `UUID`: generated UUID
- in_table: True if the evidence selection falls in a table
- original_example `UUID`: Links to another row if this is an augmented data point. This will point to the row it was dervived from
- partition `str`: The partition of the dataset this row belongs to. train, dev, test, or unassigned.
- reviews `List[REVIEW]`: A structured object containing reviews and their explanations of the stance of the evidence for this claim/evidence pair

### Evidence

The Evidence object has three main fields

```json
{
    "content": "str or list[list[str]]",
    "meta": {},
    "type": "table or unstructured"
}
```

An example of an evidence chunk object is given below

```json
{
  "content": "Summary of ERBB2 mutants analyzed along with the IC50 values against reversible inhibitors lapatinib and AEE788.",
  "meta": {
    "end": 5693.0,
    "hypothesis.comment": "gene, drug, response metric (IC50)",
    "hypothesis.id": "LLuN9J8rEey2Sw99rAb_0Q",
    "section": "article//results and discussion",
    "start": 5581.0
  },
  "type": "unstructured"
}
```

### Claim

The claim object follows a similar structure to the evidence object. An example is shown below. By default, all claim objects in the CIViC-Fact dataset are tabular. Tabular content is represented as a list of tuples where each evidence object represents a single row and each row is composed of (col,value) tuples.

```json
{
  "content": [
    ["molecularProfile", "ERBB2 H878Y"],
    ["evidenceType", "PREDICTIVE"],
    ["significance", "resistance"],
    ["therapies", "Lapatinib"],
    ["diseases", "Cancer"]
  ],
  "meta": {
    "synonyms": [
      {
        "name": "Lapatinib",
        "synonyms": [
          "GSK572016", "GW 2016", "GW2016", "GW572016",
        ],
        "type": "therapy",
        "xrefs": "NCI:C26653"
      },
      {
        "name": "ERBB2",
        "synonyms": [
          "CD340", "HER-2", "HER-2/neu", "HER2", "c-ERB-2", "c-ERB2",
        ],
        "type": "gene",
        "xrefs": "entrezId:2064"
      }
    ]
  },
  "type": "table"
}
```

## Data Preprocessing

### Data Cleaning

Text selected through hypothesis often includes formatting marks or loss of whitespace characters. To ensure the evidence we have in the dataset is representative of what a human reading the paper would view we take the text selected via hypothesis and match it to the BioC text of the article which we have obtained via pubmed. The pubmed XML file is downloaded either from the main PMC files or from the authors uploaded manuscript and then transformed into BioC using the biotext package (<https://github.com/jakelever/biotext>). We use the newly matched text to get the position of the selection within the article.

### Data Augmentation

In order to augment the dataset for LLM training we have expanded the dataset in several ways

1. CIViC evidence item descriptions are used as further examples of "evidence" for their claim
2. The NEI class is supplemented with claims generated based on NER entities from putator plus randomly selected exceprts from the same paper
3. The REFUTES class is augments by altering claims to their opposite. For example a claim/evidence pair with a label of SUPPORTS the significance of sensitivity can be changed to be an expected REFUTES label by modifying the significance of the claim to be resistance instead of sensitivity

## Getting Started Building the Dataset (Developers)

Install and dependencies are managed via [poetry](https://python-poetry.org/docs/)

```bash
poetry install
```

Run the tests

```bash
poetry run pytest tests
```

Or to execute the snakemake workflow (This requires API key access to the hypothesis groups)

```bash
poetry run snakemake ...
```

## Workflow

The general workflow to generate the CIViC-Fact claim verification and evidence retrieval sentence pairs datasets is shown below

![workflow diagram](./workflow.png)

## Data License

The CIViC-Fact dataset does not introduce any new licensing terms. Instead, it inherits the license of the source article from which each claim/evidence pair is derived. Source articles come from PMC (Open Access subset, author-uploaded manuscripts, or the historical OCR subset) or from PubMed (when only abstracts are used). Each entry includes a "document.license" field, which specifies the applicable license so users can determine whether and how a document may be reused (e.g., commercial users must exclude entries with CC BY-NC or CC BY-NC-SA licenses). To use the dataset in full without filtering, users should assume the most restrictive license applies. For build v2025.03.25, this is CC BY-NC-SA.
