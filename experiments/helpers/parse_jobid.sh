#!/bin/bash
# helper script that parses slurm output for the job ID,
# and feeds it to back to snakemake/slurm for dependencies.
# This is required when you want to use the snakemake --immediate-submit option

# from https://stackoverflow.com/questions/59525402/how-to-immediately-submit-all-snakemake-jobs-to-slurm-cluster

if [[ "Submitted batch job" =~ "$@" ]]; then
  echo -n ""
else
  deplist=$(grep -Eo '[0-9]{1,10}' <<< "$@" | tr '\n' ',' | sed 's/.$//')
  echo -n "--dependency=aftercorr:$deplist"
fi;
