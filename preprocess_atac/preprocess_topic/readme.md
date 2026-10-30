# Topic preprocessing

Train a dictionary with `topic_train.py`, or use the dictionary paired with your
HERTA checkpoint. Keep `Topic50.pkl` and `inferencer.mallet` from the same training
run; a different dictionary requires a matching HERTA model.

Run from the repository root in the `pycistopic` environment:

```bash
conda activate pycistopic
```

## Train a new dictionary

Use 500 bp ATAC tiles from the selected cell types:

```bash
python preprocess_atac/preprocess_topic/topic_train.py \
  --train_cells gm12878 imr90 \
  --tile_dir /path/to/data/atac \
  --save_dir /path/to/topic_model \
  --mallet_path /path/to/mallet-2.0.8/bin/mallet
```

The model files are `/path/to/topic_model/Topic50.pkl` and
`/path/to/topic_model/inferencer.mallet`.

## Infer genomic topic features

Create the cell-order sidecar from the metacell-mask and tile barcode files:

```bash
python preprocess_atac/preprocess_topic/prepare_identity.py \
  --mask-cell-ids mask_barcodes.npy --tile-cell-ids tile_barcodes.npy \
  --out identity.json
```

Generate genomic topic features:

```bash
python preprocess_atac/preprocess_topic/topic_inference.py \
  --cell-type sample --tile-file sample_tile_500bp_dict.p \
  --mask-file sample_metacell_mask.csv --identity-json identity.json \
  --model-file /path/to/topic_model/Topic50.pkl \
  --inferencer-path /path/to/topic_model/inferencer.mallet \
  --chromosomes chr{1..21} --output-dir topic_output \
  --mallet-path /path/to/mallet-2.0.8/bin/mallet
```

Input tiles use 500 bp bins. Mask and tile cells are matched by barcode. The
included Java helper looks up numeric region names in the trained MALLET alphabet.
Each metacell is inferred across the requested chromosomes, then per-metacell
Top-k weights are aggregated into genomic features. `equal/sample_topics_aggregated.npz`
is the HERTA input. The output includes chromosome names, `(n_bins, 50)` matrices
and bin counts; row zero starts at genomic coordinate zero.

Use `--help` for batch sizes, iteration count, seed, selection and work-file options.
The Python environment needs NumPy, SciPy and the pycisTopic dependencies required
to load the trained model; MALLET 2.0.8 and a Java 11+ JDK are used for inference.
