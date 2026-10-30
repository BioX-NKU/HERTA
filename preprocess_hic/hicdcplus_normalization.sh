#!/bin/bash
# HiCDCPlus normalization entrypoint for HERTA Hi-C inputs.
#
# Example:
#   hic_file=/path/to/sample.hic \
#   resolution=10000 \
#   assembly=hg38 \
#   workdir_base=/path/to/output_root \
#   bash hicdcplus_normalization.sh

set -euo pipefail

trap 'echo "Error: command failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

if [ -z "${hic_file:-}" ] || [ -z "${resolution:-}" ] || [ -z "${assembly:-}" ]; then
  echo "Error: missing required environment variables."
  echo "Usage: hic_file='your_file.hic' resolution=10000 assembly='hg38' bash $0"
  echo ""
  echo "Optional variables:"
  echo "  workdir_base='/custom/path' (default: ./processed/hic)"
  echo "  feature_type='RE-based' or 'RE-agnostic' (default: RE-based)"
  echo "  R_ENV='archer' PYTHON_ENV='cuda'"
  exit 1
fi

workdir_base="${workdir_base:-./processed/hic}"
feature_type="${feature_type:-RE-based}"
R_ENV="${R_ENV:-archer}"
PYTHON_ENV="${PYTHON_ENV:-cuda}"
CONDA_SH="${CONDA_SH:-}"

if [ ! -f "$hic_file" ]; then
  echo "Error: Hi-C file does not exist: $hic_file"
  exit 1
fi

hic_file="$(cd -- "$(dirname -- "$hic_file")" && pwd)/$(basename -- "$hic_file")"

if [ -z "$CONDA_SH" ]; then
  CONDA_BASE="$(conda info --base)"
  CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
fi

if [ ! -f "$CONDA_SH" ]; then
  echo "Error: conda activation script does not exist: $CONDA_SH"
  exit 1
fi

if ! [[ "$resolution" =~ ^[0-9]+$ ]]; then
  echo "Error: resolution must be a positive integer: $resolution"
  exit 1
fi

if [[ ! "$assembly" =~ ^(hg19|hg38|mm9|mm10)$ ]]; then
  echo "Error: unsupported assembly: $assembly"
  exit 1
fi

if [[ ! "$feature_type" =~ ^(RE-based|RE-agnostic)$ ]]; then
  echo "Error: unsupported feature_type: $feature_type"
  echo "Supported values: RE-based, RE-agnostic"
  exit 1
fi

valid_resolutions=(1000 2000 5000 10000 25000 40000 50000 100000 250000 500000 1000000)
if [[ ! " ${valid_resolutions[*]} " =~ " ${resolution} " ]]; then
  echo "Warning: resolution $resolution is not in the standard HiCDCPlus test list."
  echo "Standard resolutions: ${valid_resolutions[*]}"
fi

get_resolution_label() {
  local res=$1
  if [ "$res" -ge 1000000 ]; then
    echo "$((res / 1000000))Mb"
  elif [ "$res" -ge 1000 ]; then
    echo "$((res / 1000))kb"
  else
    echo "${res}bp"
  fi
}

res_label=$(get_resolution_label "$resolution")
workdir="${workdir_base}/hicdc_normalization/${resolution}bp"

echo "============================================================="
echo "HiCDCPlus normalization"
echo "============================================================="
echo "Hi-C file: $hic_file"
echo "Resolution: $resolution bp ($res_label)"
echo "Assembly: $assembly"
echo "Feature type: $feature_type"
echo "Workdir base: $workdir_base"
echo "============================================================="

mkdir -p "$workdir"/{zvalue_intermediate_${res_label},zvalue_${res_label},features,intermediate,zvalue_intermediate,zvalue}
workdir="$(cd -- "$workdir" && pwd)"
cd "$workdir" || exit 1

R_SCRIPT="${SCRIPT_DIR}/step1_hicdcplus_normalization_run.R"
PYTHON_SCRIPT="${SCRIPT_DIR}/step2_python_save.py"

if [ ! -f "$R_SCRIPT" ]; then
  echo "Error: R script does not exist: $R_SCRIPT"
  exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
  echo "Error: Python script does not exist: $PYTHON_SCRIPT"
  exit 1
fi

conda_activate_env() {
  local env_name=$1
  set +u
  set +e
  source "$CONDA_SH"
  conda activate "$env_name"
  local status=$?
  set -e
  set -u
  return "$status"
}

conda_deactivate_env() {
  set +u
  set +e
  conda deactivate
  local status=$?
  set -e
  set -u
  return "$status"
}

set +u
source "$CONDA_SH"
set -u

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found"
  exit 1
fi

echo ""
echo "===================="
echo "Step 1: HiCDCPlus normalization"
echo "===================="

if [ "$feature_type" = "RE-agnostic" ]; then
  feature_suffix="RE-agnostic"
else
  feature_suffix="GATC_GANTC"
fi
feature_file="./features/${assembly}_${res_label}_${feature_suffix}_bintolen.txt.gz"
echo "Expected feature file: $feature_file"
if [ -f "$feature_file" ]; then
  echo "Existing feature file found: $(ls -lh "$feature_file" | awk '{print $5}')"
else
  echo "Feature file not found; the R script will generate it if needed."
fi

conda_activate_env "$R_ENV"

set +e
Rscript "$R_SCRIPT" "$hic_file" "$resolution" "$assembly" "$feature_type" "$workdir"
R_EXIT_CODE=$?
set -e
conda_deactivate_env

if [ "$R_EXIT_CODE" -ne 0 ]; then
  echo "Error: HiCDCPlus normalization failed with exit code $R_EXIT_CODE"
  exit 1
fi

echo ""
echo "===================="
echo "Checking intermediate files"
echo "===================="

python_files_missing=0
missing_files=()

if [[ "$assembly" == "hg19" || "$assembly" == "hg38" ]]; then
  chromosomes=($(seq 1 22) X)
elif [[ "$assembly" == "mm9" || "$assembly" == "mm10" ]]; then
  chromosomes=($(seq 1 19) X)
fi

for i in "${chromosomes[@]}"; do
  file_path="./intermediate/normalized_chr${i}.txt"
  if [ -f "$file_path" ]; then
    lines=$(wc -l < "$file_path" 2>/dev/null || echo "0")
    echo "chr${i}: $lines lines"
  else
    echo "Missing chr${i}"
    missing_files+=("chr${i}")
    python_files_missing=$((python_files_missing + 1))
  fi
done

if [ "$python_files_missing" -ne 0 ]; then
  echo "Error: missing $python_files_missing chromosome files: ${missing_files[*]}"
  exit 1
fi

echo ""
echo "===================="
echo "Step 2: Save sparse z-score matrices"
echo "===================="

conda_activate_env "$PYTHON_ENV"

python -c "import pandas, numpy, scipy.sparse" || {
  echo "Error: missing Python packages: pandas, numpy, scipy"
  conda_deactivate_env
  exit 1
}

set +e
canonical_args=()
if [ -n "${CELL_LINE_NAME:-}" ] || [ -n "${PROCESSED_INPUT_DIR:-}" ]; then
  if [ -z "${CELL_LINE_NAME:-}" ] || [ -z "${PROCESSED_INPUT_DIR:-}" ]; then
    echo "Set CELL_LINE_NAME and PROCESSED_INPUT_DIR together for a canonical training input copy." >&2
    exit 1
  fi
  canonical_args=(--cell_type_prefix "$CELL_LINE_NAME" --save_path "$PROCESSED_INPUT_DIR")
fi
python "$PYTHON_SCRIPT" --assembly "$assembly" --resolution "$resolution" --workdir "$workdir" "${canonical_args[@]}"
PY_EXIT_CODE=$?
set -e

if [ "$PY_EXIT_CODE" -ne 0 ]; then
  echo "Error: Python conversion failed with exit code $PY_EXIT_CODE"
  conda_deactivate_env
  exit 1
fi

expected_pickle_file="./zvalue_${res_label}/normalized_hic_zscore_dict_${res_label}.p"
fallback_pickle_file="./zvalue/normalized_hic_zscore_dict.p"

if [ -f "$expected_pickle_file" ]; then
  final_pickle_file="$expected_pickle_file"
elif [ -f "$fallback_pickle_file" ]; then
  final_pickle_file="$fallback_pickle_file"
else
  echo "Error: pickle output was not found"
  conda_deactivate_env
  exit 1
fi

echo ""
echo "===================="
echo "Final output"
echo "===================="
echo "$final_pickle_file"
ls -lh "$final_pickle_file"

set +e
python -c "
import pickle
with open('$final_pickle_file', 'rb') as handle:
    data = pickle.load(handle)
print(f'Chromosomes: {len(data)}')
print(f'Total nonzero interactions: {sum(m.nnz for m in data.values() if hasattr(m, \"nnz\")):,}')
"
VERIFY_EXIT_CODE=$?
set -e
conda_deactivate_env

if [ "$VERIFY_EXIT_CODE" -ne 0 ]; then
  echo "Error: pickle verification failed with exit code $VERIFY_EXIT_CODE"
  exit 1
fi
