last_modified_date=$(find results -printf "%TY.%Tm.%Td\n" | sort -d | tail -n 1)

echo "The results were last modified at $last_modified_date"

working_dir=$(pwd)

version_dir="$working_dir/builds/civicfact-$last_modified_date"

if [ -d "$version_dir" ]; then
    echo "A build with this date already exists: $version_dir"
    exit 1
fi

archive_name="results-${last_modified_date}.tar.gz"
echo "archiving the current results directory: $archive_name"
tar --exclude="*.pkl" --exclude="*.backup.*" --exclude="*/pubtator/*" --exclude="*/sentence_similarity/splits/*"   -zcvf $archive_name results

echo "creating: ${version_dir}"
mkdir -p $version_dir
cd $version_dir
mv $working_dir/$archive_name .


cp $working_dir/results/civic_nli/data.goal-3.jsonl.gz data.jsonl.gz
cp $working_dir/data/civic-dump.jsonl civic-dump.jsonl
cp $working_dir/results/civic_nli/sentence_level_evidence.jsonl.gz retrieval.jsonl.gz
gzip civic-dump.jsonl
cp $working_dir/results/sentence_similarity/document_chunks.jsonl.gz document_chunks.jsonl.gz
