import json
import os
import re
import subprocess


def parse_deets(details_text):
    details = {}
    for chunk in re.split(r"\s+", details_text.strip()):
        try:
            tag, value = chunk.split("=", 1)
            details[tag] = value
        except ValueError as err:
            print(f"failed to extract chunk: {repr(chunk)}")
            continue

    # details = {tag.strip(): value.strip() for tag, value in zip(details[::2], details[1::2])}
    # if snakemake, get snakemake deets too
    if re.search(r"\.snakemake/.*sh", details["Command"]) and os.path.exists(details["Command"]):
        with open(details["Command"], "r") as fh:
            lines = fh.readlines()
            for line in lines:
                if line.startswith("# properties = "):
                    line = line[len("# properties = ") :].strip()
                    props = json.loads(line)
                    details["snakemake_properties"] = props
    return details


job_list = subprocess.check_output(["squeue", "-u", os.environ["USER"], "-o", "%i"]).decode()
job_list = [int(job.strip()) for job in job_list.split("\n") if re.match(r"\d+", job.strip())]

details = []
for jobid in job_list:
    job_details = subprocess.check_output(["scontrol", "show", "jobid", f"{jobid}"]).decode()
    details.append(parse_deets(job_details))

common_props = {}
for job in details:
    snakemake = job.get("snakemake_properties", {}).get("wildcards", {})
    for k, v in snakemake.items():
        common_props.setdefault(k, set()).add(v)

for k in list(common_props):
    if len(common_props[k]) > 1:
        del common_props[k]


print("common properties:", {k: list(v)[0] for k, v in common_props.items()})
jobs_by_key = {}
for job in details:
    snakemake = {}
    for k, v in job.get("snakemake_properties", {}).get("wildcards", {}).items():
        if k not in common_props:
            snakemake[k] = v
    job_name = job["JobName"].split(".")[0]
    dup_key = json.dumps({"name": job_name, **snakemake}, sort_keys=True)
    if job_name != "monitor":
        job["dup_key"] = dup_key
        print(
            f'JobID: {job["JobId"]} name: {job_name: <16} status: {job["JobState"]} ({job["Reason"]: <10}) {job["Priority"]} - {snakemake}'
        )
        jobs_by_key.setdefault(dup_key, []).append(job)


all_dups = []

for dup_key, jobs in jobs_by_key.items():
    if len(jobs) > 1:
        print("DUPLICATES", dup_key)
        for i, job in enumerate(sorted(jobs, key=lambda job: int(job["Priority"]), reverse=True)):
            print(
                f'- JobID: {job["JobId"]} status: {job["JobState"]} ({job["Reason"]: <10}) Priority:{job["Priority"]}'
            )
            if i > 0:
                all_dups.append(job["JobId"])


print("All lower Priority dups:", " ".join(all_dups))
