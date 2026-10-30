# Environments

Run `bash setup.sh` on Linux with conda. To install a subset or use another location:

```bash
TARGETS=cuda,pycistopic bash setup.sh
TARGETS=archer bash setup.sh
CONDA_ENVS_DIR=/path/to/conda_envs bash setup.sh
```

- `cuda`: model training and inference; Python 3.9, PyTorch 2.5.1, CUDA 12.1.
- `pycistopic`: topic dictionary training and feature inference. Install MALLET
  separately and supply its binary with `--mallet-path` (or `--mallet_path`).
- `archer`: R preprocessing with ArchR and HiCDCPlus. The installer checks the
  C++17 compiler, libcurl, required R packages and the pinned ArchR revision.

`cuda.yml` uses SymPy 1.13.1 for PyTorch 2.5.1 compatibility. The topic environment
contains the pycisTopic compatibility dependencies. The R package CSV is read by
`setup.sh` when checking the preprocessing environment.

To recreate an environment, use `RECREATE=1 TARGETS=<name> bash setup.sh`.
Optional ArchR extras can be enabled with `INSTALL_ARCHR_EXTRAS=1`.
