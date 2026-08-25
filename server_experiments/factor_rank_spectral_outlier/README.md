# Llama-3.1-8B Stage 1.6 launcher

Run from the repository root. `MODEL_PATH` is required. Optional variables are
`OUTPUT_ROOT`, `OPERANDS_DIR`, `DEVICE`, `CPU_THREADS`, and
`SKIP_COLLECTION=1`.

The launcher preserves the preregistered seven-module set, Split A/B protocol,
rank-256 bulk, removal budgets, and ten random seeds. Results are resumable per
module.

Install PyTorch for the server CUDA version separately, then install the small
analysis dependency set from `requirements-server.txt` if it is not already
present.
