#!/usr/bin/env bash
set -euo pipefail

# require Bash 4+ for associative arrays
if ((BASH_VERSINFO[0] < 4)); then
  echo "Bash 4 or higher is required." >&2
  exit 1
fi

# map tasks to their runtimes
declare -A tasks2time=(
  ["ner"]=3
  ["pos"]=3
  ["copa"]=1
  ["qa"]=1
)

# list of settings
settings=(base eu sr extended)

# for each setting, map disable_baselines and local_adapters
declare -A disable_baselines=(
  ["base"]=True
  ["eu"]=True
  ["sr"]=True
  ["extended"]=False
)

declare -A local_adapters=(
  ["base"]=""
  ["eu"]="eu"
  ["sr"]="sr"
  ["extended"]="eu sr"
)

for task in "${!tasks2time[@]}"; do
  runtime="${tasks2time[$task]}"
  for setting in "${settings[@]}"; do
    db_flag="${disable_baselines[$setting]}"
    adapters="${local_adapters[$setting]}"

    cmd=(python run_unseen_lang.py "$runtime" \
      --task "$task" \
      --disable_baselines "$db_flag" \
      --output_name "$setting")

    if [[ -n "$adapters" ]]; then
      cmd+=(--local_adapters $adapters)
    fi

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
  done
done
