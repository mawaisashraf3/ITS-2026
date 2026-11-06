# Delivered experiment registry

`experiments.sqlite3` is a consistent online-backup snapshot of the actual experiment records. It contains 1 dataset, 930 samples, 4 model runs, 12 conditions, 7,320 observations, and 33 artifact references. It contains metadata and counts, not source images or private credentials.

Copy the snapshot before using administrative commands:

```bash
cp results/database/experiments.sqlite3 runs/example.sqlite3
python -m rmldnet.storage.cli --database runs/example.sqlite3 integrity
python -m rmldnet.storage.cli --database runs/example.sqlite3 metrics --model swin_b
```

`verification.json` records the snapshot SHA-256, actual import timings, idempotent re-import result, and successful database/JSON recovery comparisons. The recorded timings are current-machine observations, not a controlled cross-system benchmark. Artifact paths are relative to `results/via`; raw-image paths are relative to the prepared dataset root.
