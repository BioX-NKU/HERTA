#!/bin/bash
set -eo pipefail

# ==============================================================================
#             PART 1: User configuration
# ==============================================================================

# --- 1. Cell line to process ---
# Set the sample prefix used in the prepared input filenames.
export CELL_LINE_NAME="${CELL_LINE_NAME:?Set CELL_LINE_NAME to the sample prefix}"

# --- 2. Main working directories ---
# All outputs will be saved under a cell-line-specific subdirectory here.
export BASE_DIR="${BASE_DIR:-./processed}"
# Root directory containing one fragment subdirectory per sample.
export RAW_DATA_DIR="${RAW_DATA_DIR:-./raw/atac}"
# Directory containing this preprocessing package.
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Directory containing the pre-downloaded CTCF motif score files.
export CTCF_FILES_DIR="${CTCF_FILES_DIR:-./references}"
# Optional conda activation script. Defaults to the active conda installation.
export CONDA_SH="${CONDA_SH:-}"

# --- 3. Reference genome ---
export GENOME_ASSEMBLY="${GENOME_ASSEMBLY:-hg38}"
# Chromosome sizes file used by the Python preprocessing step.
export CHROM_SIZES_FILE="${CHROM_SIZES_FILE:-${CTCF_FILES_DIR}/${GENOME_ASSEMBLY}.chrom.sizes}"

# ==============================================================================
#          PART 2: Derived paths
# ==============================================================================

# Output directory for the current cell line.
export SAVE_LOC="${SAVE_LOC:-${BASE_DIR}/${CELL_LINE_NAME}_output}"
# Output directory for the ArchR step.
export ARCHR_LOC="${ARCHR_LOC:-${SAVE_LOC}/archr_data}"
# Input raw fragment file.
export INPUT_FRAG_FILE="${INPUT_FRAG_FILE:-${RAW_DATA_DIR}/${CELL_LINE_NAME}/fragments.tsv.gz}"
# Sorted bgzip-compressed fragment file.
export SORTED_FRAG_FILE="${SORTED_FRAG_FILE:-${SAVE_LOC}/${CELL_LINE_NAME}_bgz_sorted.tsv.gz}"


# ==============================================================================
#                       PART 3: Preprocessing workflow
# ==============================================================================

echo "========================================================"
echo "=== Processing cell line: ${CELL_LINE_NAME}"
echo "========================================================"

for required in "$INPUT_FRAG_FILE" "$CHROM_SIZES_FILE" "$CTCF_FILES_DIR/${GENOME_ASSEMBLY}_ctcf_motif_score.p"; do
    test -s "$required" || { echo "Missing input/reference: $required" >&2; exit 1; }
done
case "$GENOME_ASSEMBLY" in
    hg19|hg38) CHROM_PATTERN='^chr([1-9]|1[0-9]|2[0-2]|X|Y)[[:space:]]' ;;
    mm9|mm10) CHROM_PATTERN='^chr([1-9]|1[0-9]|X|Y)[[:space:]]' ;;
    *) echo "Unsupported genome: $GENOME_ASSEMBLY" >&2; exit 1 ;;
esac

# --- Step 1: Create output directories ---
echo "--> Step 1: Creating output directories..."
mkdir -p "${ARCHR_LOC}"
mkdir -p "${SAVE_LOC}/atac"
mkdir -p "${SAVE_LOC}/dna"
echo "Output directories created: ${SAVE_LOC}"
echo ""

# --- Step 2: Copy the CTCF motif file ---
echo "--> Step 2: Preparing CTCF motif file..."
cp "${CTCF_FILES_DIR}/${GENOME_ASSEMBLY}_ctcf_motif_score.p" "${SAVE_LOC}/dna/"
echo "CTCF motif file copied."
echo ""

# --- Step 3: Convert and sort the fragment file ---
echo "--> Step 3: Converting fragment format..."
echo "Processing raw fragment file: ${INPUT_FRAG_FILE}"

# Normalized flow: decompress -> remove comments -> remove optional prefix -> keep standard chromosomes -> sort -> bgzip.
read_fragments() {
    case "$INPUT_FRAG_FILE" in
        *.gz|*.bgz) gzip -cd "$INPUT_FRAG_FILE" ;;
        *) cat "$INPUT_FRAG_FILE" ;;
    esac
}
read_fragments | \
    grep -v '^#' | \
    sed 's/^GRCh38_//' | \
    grep -E "$CHROM_PATTERN" | \
    sort -k1,1V -k2,2n | \
    bgzip > "${SORTED_FRAG_FILE}"

echo "Fragment conversion finished: ${SORTED_FRAG_FILE}"
echo ""

# Move a stale tabix index aside if one exists.
if [ -f "${SORTED_FRAG_FILE}.tbi" ]; then
    if [ -n "${TRASH_DIR:-}" ]; then
        mkdir -p "${TRASH_DIR}"
        echo "Found a stale index file; moving it to ${TRASH_DIR} ..."
        mv "${SORTED_FRAG_FILE}.tbi" "${TRASH_DIR}/"
    else
        echo "Found a stale index file; removing it before re-indexing ..."
        rm -f "${SORTED_FRAG_FILE}.tbi"
    fi
fi
echo ""


# --- Step 4: Activate environments and run the core scripts ---
echo "--> Step 4: Activating conda environments and running the core analyses..."

# Load conda shell functions so that conda activate works inside this script.
if [ -z "${CONDA_SH}" ]; then
    CONDA_BASE="$(conda info --base)"
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
fi
source "${CONDA_SH}"

conda activate "${R_ENV:-archer}"
echo "Running ArchR script..."
Rscript "${SCRIPT_DIR}/ArchR_layer.R" \
  "${CELL_LINE_NAME}" \
  "${ARCHR_LOC}" \
  "${SORTED_FRAG_FILE}" \
  "${GENOME_ASSEMBLY}"
echo "ArchR step completed."

# Run the Python preprocessing script.
echo "Current environment: $CONDA_DEFAULT_ENV"
echo "Python interpreter: $(which python)"

echo "Running Python script to generate final files..."
conda deactivate
source "${CONDA_SH}"
conda activate "${PYTHON_ENV:-cuda}"
python "${SCRIPT_DIR}/scATAC_preparation.py" \
  --cell_type_prefix "${CELL_LINE_NAME}" \
  --fragment_file  "${SORTED_FRAG_FILE}" \
  --barcode_file "${ARCHR_LOC}/archr_filtered_barcode.csv" \
  --lsi_file "${ARCHR_LOC}/archr_filtered_lsi.csv" \
  --genome_assembly "${GENOME_ASSEMBLY}" \
  --chrom_sizes "${CHROM_SIZES_FILE}" \
  --save_path "${SAVE_LOC}"
echo ""
echo "*****************************************************************"
echo "All preprocessing steps for cell line [${CELL_LINE_NAME}] are complete."
echo "Final files are located at: ${SAVE_LOC}/atac"
echo "*****************************************************************"
