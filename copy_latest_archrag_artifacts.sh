#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-corpus/output}"
target_dir="${2:-archrag}"

required_files=(
  "create_final_entities.parquet"
  "create_final_relationships.parquet"
)

if [[ ! -d "$output_root" ]]; then
  echo "Output root not found: $output_root" >&2
  exit 1
fi

latest_artifacts=""
while IFS= read -r run_dir; do
  artifacts_dir="$run_dir/artifacts"
  [[ -d "$artifacts_dir" ]] || continue

  missing=0
  for file in "${required_files[@]}"; do
    if [[ ! -f "$artifacts_dir/$file" ]]; then
      missing=1
      break
    fi
  done

  if [[ "$missing" -eq 0 ]]; then
    latest_artifacts="$artifacts_dir"
    break
  fi
done < <(find "$output_root" -mindepth 1 -maxdepth 1 -type d | sort -r)

if [[ -z "$latest_artifacts" ]]; then
  echo "No completed GraphRAG artifacts found under: $output_root" >&2
  echo "Required files: ${required_files[*]}" >&2
  exit 1
fi

mkdir -p "$target_dir"

for file in "${required_files[@]}"; do
  cp -f "$latest_artifacts/$file" "$target_dir/$file"
done

echo "Copied ArchRAG input artifacts:"
echo "  source: $latest_artifacts"
echo "  target: $target_dir"
for file in "${required_files[@]}"; do
  echo "  - $target_dir/$file"
done
