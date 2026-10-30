#!/usr/bin/env Rscript
# HiCDCPlus normalization for HERTA Hi-C inputs.

library(curl)
library(httr)
library(strawr)
library(dplyr)
library(HiCDCPlus)
library(data.table)
library(GenomicRanges)
library(InteractionSet)

options(scipen = 9999)
options(timeout = 3600)
setDTthreads(1)

args <- commandArgs(trailing = TRUE)

if (length(args) < 3) {
  stop(
    "Usage: step1_hicdcplus_normalization_run.R ",
    "<hic_file> <resolution> <assembly> [feature_type] [workdir]"
  )
}

hicfile_path <- normalizePath(args[1], mustWork = TRUE)
resolution <- as.integer(args[2])
assembly <- args[3]
feature_type <- if (length(args) >= 4) args[4] else "RE-based"
workdir <- if (length(args) >= 5) normalizePath(args[5], mustWork = FALSE) else getwd()
dir.create(workdir, showWarnings = FALSE, recursive = TRUE)

if (is.na(resolution) || resolution <= 0) {
  stop("Resolution must be a positive integer.")
}

if (!(feature_type %in% c("RE-based", "RE-agnostic"))) {
  stop("Unsupported feature_type: ", feature_type, ". Use RE-based or RE-agnostic.")
}

get_resolution_label <- function(res) {
  if (res >= 1000000) {
    return(paste0(as.integer(res / 1000000), "Mb"))
  }
  if (res >= 1000) {
    return(paste0(as.integer(res / 1000), "kb"))
  }
  paste0(res, "bp")
}

res_label <- get_resolution_label(resolution)

cat("\n=============================================================\n")
cat("HiCDCPlus normalization\n")
cat("=============================================================\n")
cat("Hi-C file:", hicfile_path, "\n")
cat("Resolution:", resolution, "bp (", res_label, ")\n", sep = "")
cat("Assembly:", assembly, "\n")
cat("Feature type:", feature_type, "\n")
cat("Workdir:", workdir, "\n")
cat("=============================================================\n\n")

if (assembly %in% c("mm9", "mm10")) {
  species <- "Mmusculus"
  chrom_length <- 19
} else if (assembly %in% c("hg19", "hg38")) {
  species <- "Hsapiens"
  chrom_length <- 22
} else {
  stop("Unsupported assembly: ", assembly)
}

outdir <- file.path(workdir, "features")
outpth <- file.path(workdir, "intermediate")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
dir.create(outpth, showWarnings = FALSE, recursive = TRUE)

chromosomes <- paste("chr", seq(1, chrom_length, 1), sep = "")
chromosomes[[chrom_length + 1]] <- "chrX"

set.seed(1010)

cat("=============================================================\n")
cat("Step 1: Build genomic features\n")
cat("=============================================================\n")

if (feature_type == "RE-agnostic") {
  cat("Using RE-agnostic mode: generate uniform binned gi_list directly.\n")
  gi_list <- generate_binned_gi_list(
    binsize = resolution,
    chrs = chromosomes,
    gen = species,
    gen_ver = assembly,
    Dthreshold = 2e6
  )
} else {
  cat("Using RE-based mode: construct GATC/GANTC features.\n")
  output_suffix <- "GATC_GANTC"
  feature_prefix <- file.path(outdir, paste0(assembly, "_", res_label, "_", output_suffix))
  feature_file <- paste0(feature_prefix, "_bintolen.txt.gz")

  construct_features(
    output_path = feature_prefix,
    gen = species,
    gen_ver = assembly,
    sig = c("GATC", "GANTC"),
    bin_type = "Bins-uniform",
    binsize = resolution,
    chrs = chromosomes
  )

  gi_list <- generate_bintolen_gi_list(
    bintolen_path = feature_file,
    gen = species,
    gen_ver = assembly
  )
}

if (length(gi_list) == 0) {
  stop("Feature construction produced an empty gi_list.")
}

cat("Feature chromosomes:", paste(names(gi_list), collapse = ", "), "\n\n")

add_hic_counts_smart <- function(gi_list, hic_path, chrs = NULL) {
  HiCDCPlus:::gi_list_validate(gi_list)
  binsize <- HiCDCPlus:::gi_list_binsize_detect(gi_list)
  Dthreshold <- HiCDCPlus:::gi_list_Dthreshold.detect(gi_list)

  if (is.null(chrs)) {
    chrs <- sort(names(gi_list))
  } else {
    chrs <- chrs[chrs %in% names(gi_list)]
    if (length(chrs) == 0) {
      stop("None of the requested chromosomes exists in the gi_list object.")
    }
  }

  for (chrom in chrs) {
    cat("Processing ", chrom, "...\n", sep = "")

    chr_numeric <- gsub("^chr", "", chrom)
    attempts <- list(
      list(label = "numeric chromosome name", value = chr_numeric),
      list(label = "chr-prefixed chromosome name", value = chrom)
    )

    count_matrix <- NULL
    errors <- character()

    for (attempt in attempts) {
      result <- tryCatch(
        strawr::straw(
          norm = "NONE",
          fname = path.expand(hic_path),
          chr1loc = attempt$value,
          chr2loc = attempt$value,
          unit = "BP",
          binsize = binsize
        ),
        error = function(e) e
      )

      if (!inherits(result, "error") && !is.null(result) && nrow(result) > 0) {
        count_matrix <- as.data.frame(result)
        cat("  Read counts with ", attempt$label, ".\n", sep = "")
        break
      }

      if (inherits(result, "error")) {
        errors <- c(errors, paste0(attempt$label, ": ", conditionMessage(result)))
      } else {
        errors <- c(errors, paste0(attempt$label, ": no rows returned"))
      }
    }

    if (is.null(count_matrix) || nrow(count_matrix) == 0) {
      warning(
        paste0(
          chrom,
          " has no readable counts and will be dropped. Attempts: ",
          paste(errors, collapse = " | ")
        )
      )
      gi_list[[chrom]] <- NULL
      next
    }

    required_cols <- c("x", "y", "counts")
    missing_cols <- setdiff(required_cols, colnames(count_matrix))
    if (length(missing_cols) > 0) {
      stop(
        "strawr output for ",
        chrom,
        " is missing columns: ",
        paste(missing_cols, collapse = ", ")
      )
    }

    count_matrix <- count_matrix %>%
      dplyr::filter(abs(.data$y - .data$x) <= Dthreshold) %>%
      dplyr::rename(startI = "x", startJ = "y")

    if (nrow(count_matrix) == 0) {
      warning(paste0(chrom, " has no counts within Dthreshold and will be dropped."))
      gi_list[[chrom]] <- NULL
      next
    }

    gi_list[[chrom]] <- HiCDCPlus:::add_2D_features(gi_list[[chrom]], count_matrix)
    cat("  Completed ", nrow(count_matrix), " interactions.\n", sep = "")
  }

  if (length(gi_list) == 0) {
    stop("No chromosomes had readable Hi-C counts.")
  }

  gi_list
}

cat("=============================================================\n")
cat("Step 2: Add Hi-C counts\n")
cat("=============================================================\n")

gi_list <- add_hic_counts_smart(gi_list, hic_path = hicfile_path, chrs = names(gi_list))

first_chrom <- names(gi_list)[1]
cat("Counts added. First retained chromosome:", first_chrom, "\n")
print(utils::head(mcols(gi_list[[first_chrom]])$counts))

cat("=============================================================\n")
cat("Step 3: Fit HiCDCPlus\n")
cat("=============================================================\n")

gi_list <- expand_1D_features(gi_list)
gi_list <- HiCDCPlus_parallel(gi_list, ncore = 2)

cat("HiCDCPlus fitting completed\n\n")

cat("=============================================================\n")
cat("Step 4: Save per-chromosome normalized tables\n")
cat("=============================================================\n")

for (chrom in names(gi_list)) {
  file_name <- file.path(outpth, paste0("normalized_", chrom, ".txt"))
  gi_data <- gi_list[[chrom]]

  if (length(gi_data) > 0) {
    anchor1_gr <- anchors(gi_data, type = "first")
    anchor2_gr <- anchors(gi_data, type = "second")
    gi_mcols <- mcols(gi_data)

    required_cols <- c("counts", "mu", "sdev")
    missing_cols <- setdiff(required_cols, colnames(gi_mcols))
    if (length(missing_cols) > 0) {
      stop(chrom, " is missing model columns: ", paste(missing_cols, collapse = ", "))
    }

    df <- data.frame(
      start1 = start(anchor1_gr),
      start2 = start(anchor2_gr),
      counts = gi_mcols$counts,
      mu = gi_mcols$mu,
      sdev = gi_mcols$sdev,
      stringsAsFactors = FALSE
    )

    write.table(df, file = file_name, sep = "\t",
                row.names = FALSE, col.names = TRUE, quote = FALSE)

    cat("  ", chrom, ": ", nrow(df), " interactions\n", sep = "")
  }
}

cat("\n=============================================================\n")
cat("HiCDCPlus normalization completed\n")
cat("=============================================================\n")
cat("Resolution:", resolution, "bp (", res_label, ")\n", sep = "")
cat("Feature type:", feature_type, "\n")
cat("Chromosomes:", length(gi_list), "\n")
cat("Intermediate tables:", outpth, "\n")
cat(
  "Python output: ",
  file.path(workdir, paste0("zvalue_", res_label), paste0("normalized_hic_zscore_dict_", res_label, ".p")),
  "\n",
  sep = ""
)
cat("=============================================================\n")
