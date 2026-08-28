# Evidence Retrieval Review

Review of evidence retrieval results using the 2024.12.22 data build. This was done with 1000 character chunks with 100 character overlap. However there was a bug where some chunks were merged incorrectly resulting in lost text from some chunks. This has since been fixed but will have affected a percentage of these annotations. This likely means that there will be more NEI labels that we would otherwise expect. As such this data was only used for label agreement comparison and not for calculating model accuracy or evidence retrieval success rates.

The instructions given to annotators are saved in the Instructions.pdf file in this folder

## Inter-Annotator Agreement Metrics

<https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/>

- Cohen’s Kappa Statistic: used to compare two annotators only
- Fleiss kappa (adaptation of Cohen’s kappa for 3 or more raters)
- percent agreement
