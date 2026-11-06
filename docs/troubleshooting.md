# Troubleshooting

- **Python dependency resolution:** use Python 3.12 and the CPU lock file after installing the official CPU PyTorch wheels. Use matching official CUDA wheels for GPU environments.
- **Dataset checksum mismatch:** do not change or bypass the expected checksum. Re-download the named release and verify the source. Do not execute bundled archive scripts.
- **Source images unavailable:** registry ingestion, queries, bootstrap statistics and delivered result inspection still work. Sensitivity and new training require the prepared local dataset.
- **Database locked:** allow the current writer to finish. The application waits up to 30 seconds. Do not copy or remove WAL files as a workaround.
- **Duplicate/overlap import error:** inspect sample IDs, content hashes and group membership. Keep split separation; do not weaken the validator to make an import pass.
- **Invalid aggregate counts:** recompute counts from the corresponding per-image results. Metrics must use the same masks, resolution and threshold.
- **Existing export/backup destination:** choose a new destination. The application deliberately preserves existing files.
- **Unknown database schema:** retain the original file. The current release supports schema version 1 and does not attempt speculative migrations.
- **Large full-resolution training:** reduce batch size, use the compact variant for development, or use a GPU. Reduced experiments must keep their configuration and interpretation explicit.
- **Semantic checkpoint lacks lane instances:** use the semantic prediction command. Instance decoding requires actual instance-supervised labels and training.
- **Apparent high accuracy with weak F1:** background dominates thin-lane images. Inspect confusion counts and lane IoU rather than accuracy alone.
- **Plot font cache warning:** set `MPLCONFIGDIR` to a writable directory, such as a subdirectory under `runs/`.
