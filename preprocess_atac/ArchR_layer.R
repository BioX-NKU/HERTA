#!/usr/bin/env Rscript
# Export QC-filtered barcodes and LSI coordinates from an ArchR project.
# Usage: Rscript preprocess_atac/ArchR_layer.R sample output_dir fragments.tsv.gz hg38

library(ArchR)
library(tidyr)

args <- commandArgs(trailing = TRUE)
if (length(args) < 4) stop("Usage: ArchR_layer.R sample archr_path fragments genome")

#####################################
#      Step 0. Initial set-ups      #
#####################################

#1. Initial set-up
sample_name <- args[1]
archr_path <- normalizePath(args[2], mustWork = TRUE)
input_path <- normalizePath(args[3], mustWork = TRUE)
genome_assembly <- args[4]

setwd(archr_path)

#2. Hyper-parameters
# addArchRGenome("mm10")  # nolint
addArchRGenome(genome_assembly)
archr_seed <- as.integer(Sys.getenv("ARCHR_SEED", "1"))
archr_threads <- as.integer(Sys.getenv("ARCHR_THREADS", "1"))
min_tss <- as.numeric(Sys.getenv("ARCHR_MIN_TSS", "4"))
min_frags <- as.integer(Sys.getenv("ARCHR_MIN_FRAGS", "1000"))
cluster_resolution <- as.numeric(Sys.getenv("ARCHR_CLUSTER_RESOLUTION", "2"))
if (any(!is.finite(c(archr_seed, archr_threads, min_tss, min_frags, cluster_resolution))) ||
    archr_threads < 1 || min_tss < 0 || min_frags < 1 || cluster_resolution <= 0) stop("Invalid ArchR parameter override")
set.seed(archr_seed)
addArchRThreads(threads = archr_threads)
tile_mat_params <- list()
tile_mat_params$tileSize <- 500

#########################################
#      Step 1. Load fragmens files      #
#########################################

#1. Load fragments files

ArrowFiles <- createArrowFiles( # nolint
    inputFiles = input_path,
    sampleNames = sample_name,
    TileMatParams = tile_mat_params,
    minTSS = min_tss, # ArchR 1.0.3 parameter name; default remains 4
    minFrags = min_frags,
    addTileMat = TRUE,
    addGeneScoreMat = FALSE
)

print("Finished creating ArrowFiles")

#2. Create archr project
proj <- ArchRProject(
    ArrowFiles = ArrowFiles,
    # outputDirectory = paste0(archr_path, "/archr_out/"), #nolint
    outputDirectory = paste0(archr_path, "/"),
    copyArrows = FALSE
)

print("Finished creating Archr project")

#3. Run lsi
proj <- addIterativeLSI(
  ArchRProj = proj, useMatrix = "TileMatrix",
  name = "IterativeLSI", iterations = 2,
  clusterParams = list(
    resolution = c(0.2), sampleCells = 10000, n.start = 10
  ),
  varFeatures = 25000, dimsToUse = 1:30, seed = archr_seed, threads = archr_threads
)

print("Finished running lsi")

#3.1. Clustering
proj <- addClusters(
    input = proj, reducedDims = "IterativeLSI",
    method = "Seurat", name = "Clusters",
    resolution = cluster_resolution, seed = archr_seed, force = TRUE
)

print("Finished clustering")

#3.2. tsne
# proj <- addTSNE(
#     ArchRProj = proj, reducedDims = "IterativeLSI",
#     name = "TSNE", perplexity = 30
# )

####################################
#      Step 2. Save all files      #
####################################

#1. Save
final_bc <- rownames(proj@cellColData)
write.csv(final_bc, file = paste0(archr_path,  "/archr_filtered_barcode.csv"))
lsi <- getReducedDims(proj)
write.csv(lsi, file = paste0(archr_path, "/archr_filtered_lsi.csv"))
writeLines(capture.output(sessionInfo()), file.path(archr_path, "sessionInfo.txt"))
writeLines(capture.output(list(seed=archr_seed, threads=archr_threads,
    minTSS=min_tss, minFrags=min_frags, cluster_resolution=cluster_resolution,
    OPENBLAS_NUM_THREADS=Sys.getenv("OPENBLAS_NUM_THREADS"))),
    file.path(archr_path, "preprocessing_parameters.txt"))
saveArchRProject(ArchRProj = proj, outputDirectory = archr_path, load = FALSE)
