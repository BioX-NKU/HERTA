#!/usr/bin/env bash
# Build the final HERTA environments from ./env using conda only.
#
# Expected repository layout:
#   setup.sh
#   env/
#     cuda.yml
#     pycistopic.yml
#     archer.yml
#     archer_R_installed_packages.csv
#
# Usage:
#   bash setup.sh
#
# Optional:
#   RECREATE=1 bash setup.sh
#   TARGETS=cuda,pycistopic bash setup.sh
#   TARGETS=archer bash setup.sh

set -euo pipefail

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${ENV_DIR:-${ROOT_DIR}/env}"
TARGETS="${TARGETS:-cuda,pycistopic,archer}"
RECREATE="${RECREATE:-0}"
R_SNAPSHOT_POLICY="${R_SNAPSHOT_POLICY:-required}"
# ArchR::installExtraPackages installs optional GUI/integration packages
# (Cairo, harmony, presto, shiny...). Keep it opt-in so the core
# archer environment is reproducible and version-checked by the snapshot.
INSTALL_ARCHR_EXTRAS="${INSTALL_ARCHR_EXTRAS:-0}"
CRAN_REPO="${CRAN_REPO:-https://cloud.r-project.org}"
ARCHR_REF="${ARCHR_REF:-6feec354ad6c8052ddbc4626a2ca2d858ed465bf}"
ARCHR_EXPECTED_SHA="${ARCHR_EXPECTED_SHA:-6feec354ad6c8052ddbc4626a2ca2d858ed465bf}"
CONDA_SOLVER="${CONDA_SOLVER:-libmamba}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Error: these environments target linux-64. Run setup on the Linux server." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found in PATH. Start from a conda base shell." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Error: conda activation script not found under ${CONDA_BASE}" >&2
  exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"

contains_target() {
  local needle=$1
  case ",${TARGETS}," in
    *",${needle},"*) return 0 ;;
    *) return 1 ;;
  esac
}

env_exists() {
  local env_name=$1
  conda env list | awk -v name="${env_name}" 'NF && $1 !~ /^#/ { if ($1 == name || $NF == name || $NF ~ "/" name "$" ) found=1 } END { exit(found ? 0 : 1) }'
}


conda_run_env() {
  local env_name=$1
  shift
  if conda env list | awk -v name="${env_name}" 'NF && $1 !~ /^#/ { if ($1 == name) found=1 } END { exit(found ? 0 : 1) }'; then
    conda run -n "${env_name}" "$@"
  else
    local env_path="${CONDA_ENVS_DIR:-}/$env_name"
    if [[ -n "${CONDA_ENVS_DIR:-}" && -d "${env_path}" ]]; then
      conda run -p "${env_path}" "$@"
    else
      conda run -n "${env_name}" "$@"
    fi
  fi
}

conda_env_create_args() {
  if [[ -n "${CONDA_SOLVER}" ]] && conda env create --help 2>&1 | grep -q -- '--solver'; then
    printf '%s\0%s\0' --solver "${CONDA_SOLVER}"
  fi
}

smoke_test_env() {
  local env_name=$1
  case "${env_name}" in
    cuda)
      conda_run_env cuda python - <<'PYSMOKE'
import numpy, pandas, scipy, torch
print("cuda smoke ok", torch.__version__)
PYSMOKE
      ;;
    pycistopic)
      conda_run_env pycistopic python - <<'PYSMOKE'
import pkg_resources
import lda
from bs4 import BeautifulSoup
from pycisTopic.cistopic_class import create_cistopic_object
from pycisTopic.lda_models import run_cgs_models_mallet
print("pycistopic import smoke ok")
PYSMOKE
      conda_run_env pycistopic python "${ROOT_DIR}/preprocess_atac/preprocess_topic/topic_train.py" --help >/dev/null
      ;;
    archer)
      conda_run_env archer Rscript --version >/dev/null
      conda_run_env archer Rscript --vanilla - <<'RSMOKE'
cxx17 <- tryCatch(system2(file.path(R.home("bin"), "R"), c("CMD", "config", "CXX17"), stdout = TRUE), error = function(e) "")
if (!length(cxx17) || !nzchar(cxx17[[1]])) stop("R CXX17 compiler is not configured")
if (!nzchar(Sys.which(strsplit(cxx17[[1]], " ")[[1]][[1]]))) stop("R CXX17 compiler is not on PATH: ", cxx17[[1]])
if (!capabilities("libcurl")) stop("R was not built with libcurl support")
cat("archer base smoke ok; CXX17=", cxx17[[1]], "\n", sep = "")
RSMOKE
      ;;
  esac
}

create_env() {
  local env_name=$1
  local yml_path="${ENV_DIR}/${env_name}.yml"

  if [[ ! -f "${yml_path}" ]]; then
    echo "Error: missing environment file: ${yml_path}" >&2
    exit 1
  fi

  echo
  echo "============================================================"
  echo "Conda environment: ${env_name}"
  echo "YAML: ${yml_path}"
  echo "============================================================"

  if env_exists "${env_name}"; then
    if [[ "${RECREATE}" == "1" ]]; then
      echo "Removing existing environment because RECREATE=1: ${env_name}"
      if conda env list | awk -v name="${env_name}" 'NF && $1 !~ /^#/ { if ($1 == name) found=1 } END { exit(found ? 0 : 1) }'; then
        conda env remove -n "${env_name}" -y
      elif [[ -n "${CONDA_ENVS_DIR:-}" && -d "${CONDA_ENVS_DIR}/${env_name}" ]]; then
        conda env remove -p "${CONDA_ENVS_DIR}/${env_name}" -y
      else
        echo "Error: could not locate environment to recreate: ${env_name}" >&2
        exit 1
      fi
    else
      echo "Existing environment found. Checking it before reuse: ${env_name}"
      if ! smoke_test_env "${env_name}"; then
        echo "Error: existing environment '${env_name}' failed its smoke test." >&2
        echo "Fix: rerun with RECREATE=1 TARGETS=${env_name} bash setup.sh" >&2
        exit 1
      fi
      echo "Existing environment smoke test passed: ${env_name}"
      return
    fi
  fi

  mapfile -d '' create_args < <(conda_env_create_args)
  if [[ -n "${CONDA_ENVS_DIR:-}" ]]; then
    mkdir -p "${CONDA_ENVS_DIR}"
    conda env create "${create_args[@]}" -p "${CONDA_ENVS_DIR}/${env_name}" -f "${yml_path}"
  else
    conda env create "${create_args[@]}" -f "${yml_path}"
  fi
  smoke_test_env "${env_name}"
}

repair_archer_r_packages() {
  local snapshot="${ENV_DIR}/archer_R_installed_packages.csv"
  if [[ ! -f "${snapshot}" && -f "${ENV_DIR}/arcerr_R_installed_packages.csv" ]]; then
    snapshot="${ENV_DIR}/arcerr_R_installed_packages.csv"
  fi
  if [[ ! -f "${snapshot}" ]]; then
    echo "Error: missing R package version snapshot: ${ENV_DIR}/archer_R_installed_packages.csv" >&2
    exit 1
  fi

  echo
  echo "============================================================"
  echo "Checking archer R packages"
  echo "Snapshot: ${snapshot}"
  echo "Policy: ${R_SNAPSHOT_POLICY}"
  echo "============================================================"

  if [[ -n "${CONDA_ENVS_DIR:-}" ]]; then
    export TMPDIR="${TMPDIR:-${CONDA_ENVS_DIR}/.tmp}"
    mkdir -p "${TMPDIR}"
  fi

  set +u
  if [[ -n "${CONDA_ENVS_DIR:-}" && -d "${CONDA_ENVS_DIR}/archer" ]]; then
    conda activate "${CONDA_ENVS_DIR}/archer"
  else
    conda activate archer
  fi
  set -u

  CRAN_REPO="${CRAN_REPO}" \
  ARCHR_REF="${ARCHR_REF}" \
  ARCHR_EXPECTED_SHA="${ARCHR_EXPECTED_SHA}" \
  INSTALL_ARCHR_EXTRAS="${INSTALL_ARCHR_EXTRAS}" \
  R_SNAPSHOT_CSV="${snapshot}" \
  R_SNAPSHOT_POLICY="${R_SNAPSHOT_POLICY}" \
  Rscript --vanilla - <<'RSCRIPT'
options(repos = c(CRAN = Sys.getenv("CRAN_REPO", unset = "https://cloud.r-project.org")))

snapshot_path <- Sys.getenv("R_SNAPSHOT_CSV", unset = "")
snapshot_policy <- Sys.getenv("R_SNAPSHOT_POLICY", unset = "all")
if (!snapshot_policy %in% c("all", "required", "none")) {
  stop("R_SNAPSHOT_POLICY must be one of: all, required, none")
}

snapshot <- data.frame(Package = character(), Version = character())
if (snapshot_policy != "none") {
  snapshot <- read.csv(snapshot_path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!all(c("Package", "Version") %in% names(snapshot))) {
    stop("R snapshot CSV must contain Package and Version columns: ", snapshot_path)
  }
}

snapshot_version <- function(pkg) {
  if (snapshot_policy == "none" || nrow(snapshot) == 0) return(NA_character_)
  hit <- snapshot$Version[snapshot$Package == pkg]
  if (length(hit) == 0 || is.na(hit[[1]]) || hit[[1]] == "") return(NA_character_)
  hit[[1]]
}

install_cran_if_missing <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  for (pkg in missing) {
    target <- snapshot_version(pkg)
    if (!is.na(target) && requireNamespace("remotes", quietly = TRUE)) {
      remotes::install_version(pkg, version = target, dependencies = TRUE)
    } else {
      install.packages(pkg, dependencies = TRUE)
    }
  }
}

install_cran_if_missing(c("BiocManager", "remotes"))

check_compile_toolchain <- function() {
  cxx17 <- tryCatch(system2(file.path(R.home("bin"), "R"), c("CMD", "config", "CXX17"), stdout = TRUE), error = function(e) "")
  if (!length(cxx17) || !nzchar(cxx17[[1]])) stop("R CXX17 compiler is not configured; keep gcc/gxx/binutils in archer.yml")
  compiler <- strsplit(cxx17[[1]], " ")[[1]][[1]]
  if (!nzchar(Sys.which(compiler))) stop("R CXX17 compiler not found on PATH: ", compiler)
  if (!capabilities("libcurl")) stop("R libcurl support is missing; keep libcurl/curl in archer.yml")
  message("Compile toolchain OK: CXX17=", cxx17[[1]])
}

check_compile_toolchain()

bioc_pkgs <- c(
  "HiCDCPlus",
  "GenomicRanges",
  "InteractionSet",
  "S4Vectors",
  "IRanges",
  "GenomeInfoDb",
  "SummarizedExperiment",
  "BSgenome",
  "BSgenome.Hsapiens.UCSC.hg38",
  "Biostrings",
  "rtracklayer",
  "BiocIO",
  "chromVAR",
  "motifmatchr",
  "rhdf5"
)
missing_bioc <- bioc_pkgs[!vapply(bioc_pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_bioc) > 0) {
  BiocManager::install(missing_bioc, ask = FALSE, update = FALSE)
}

install_cran_if_missing(c("strawr", "tidyr", "data.table", "dplyr", "curl", "httr", "Rcpp", "nabor", "matrixStats", "magrittr", "uwot", "Seurat"))

if (!requireNamespace("ArchR", quietly = TRUE)) {
  remotes::install_github(
    "GreenleafLab/ArchR",
    ref = Sys.getenv("ARCHR_REF", unset = "6feec354ad6c8052ddbc4626a2ca2d858ed465bf"),
    repos = BiocManager::repositories(),
    upgrade = "never"
  )
}

if (Sys.getenv("INSTALL_ARCHR_EXTRAS", unset = "1") == "1") {
  ArchR::installExtraPackages()
}

archr_expected_sha <- Sys.getenv("ARCHR_EXPECTED_SHA", unset = "")
if (nzchar(archr_expected_sha) && requireNamespace("ArchR", quietly = TRUE)) {
  archr_desc <- utils::packageDescription("ArchR")
  archr_sha <- archr_desc$GithubSHA1
  if (is.null(archr_sha) || !nzchar(archr_sha)) {
    stop("ARCHR_EXPECTED_SHA was set but installed ArchR does not record GithubSHA1")
  }
  if (!identical(tolower(archr_sha), tolower(archr_expected_sha))) {
    stop("ArchR SHA mismatch: expected ", archr_expected_sha, " got ", archr_sha)
  }
}

required <- c(
  "ArchR",
  "strawr",
  "HiCDCPlus",
  "GenomicRanges",
  "InteractionSet",
  "S4Vectors",
  "IRanges",
  "GenomeInfoDb",
  "SummarizedExperiment",
  "BSgenome",
  "BSgenome.Hsapiens.UCSC.hg38",
  "Biostrings",
  "rtracklayer",
  "BiocIO",
  "chromVAR",
  "motifmatchr",
  "rhdf5",
  "tidyr",
  "data.table",
  "dplyr",
  "curl",
  "httr",
  "Rcpp",
  "nabor",
  "matrixStats",
  "magrittr",
  "uwot",
  "Seurat"
)

missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("Missing required R packages: ", paste(missing, collapse = ", "))
}

if (snapshot_policy != "none") {
  check_pkgs <- if (snapshot_policy == "all") snapshot$Package else required
  mismatches <- character()
  missing_snapshot <- character()
  for (pkg in unique(check_pkgs)) {
    expected <- snapshot_version(pkg)
    if (is.na(expected)) next
    if (!requireNamespace(pkg, quietly = TRUE)) {
      missing_snapshot <- c(missing_snapshot, pkg)
      next
    }
    actual <- as.character(utils::packageVersion(pkg))
    if (!identical(actual, expected)) {
      mismatches <- c(mismatches, paste0(pkg, " expected ", expected, " got ", actual))
    }
  }
  if (length(missing_snapshot) > 0 || length(mismatches) > 0) {
    if (length(missing_snapshot) > 0) {
      message("Missing R packages from snapshot:")
      message(paste0("  - ", missing_snapshot, collapse = "\n"))
    }
    if (length(mismatches) > 0) {
      message("R package version mismatches:")
      message(paste0("  - ", mismatches, collapse = "\n"))
    }
    stop("R package versions do not match snapshot: ", snapshot_path)
  }
}

message("archer R package check passed")
RSCRIPT

  Rscript --vanilla -e 'required <- c("ArchR", "strawr", "HiCDCPlus", "GenomicRanges", "InteractionSet", "data.table", "dplyr", "tidyr", "Seurat", "uwot", "rhdf5"); missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]; if (length(missing)) stop("Missing required R packages after repair: ", paste(missing, collapse=", ")); message("archer package smoke test passed")'
}

if contains_target "cuda"; then
  create_env cuda
fi

if contains_target "pycistopic"; then
  create_env pycistopic
fi

if contains_target "archer"; then
  create_env archer
  repair_archer_r_packages
fi

echo
echo "============================================================"
echo "HERTA environments are ready"
echo "============================================================"
echo "Activate with:"
echo "  conda activate cuda"
echo "  conda activate pycistopic"
echo "  conda activate archer"
