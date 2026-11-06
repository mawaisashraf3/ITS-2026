# Database design, validation and recovery

The schema is packaged in `rmldnet/storage/schema.sql`. SQLite provides transactional single-file storage, foreign keys, SQL views, online backup, and portability without another service. The runtime uses WAL mode and a 30-second lock timeout.

| Table | Purpose and constraints |
|---|---|
| datasets | Unique content-derived dataset identity and JSON release metadata |
| samples | Source ID, paths, SHA-256, group ID and instance-supervision flag; unique dataset/content pair |
| splits | Exactly named train/validation/test partitions per dataset |
| split_memberships | One membership per sample; composite foreign keys prevent cross-dataset assignment |
| experiments | Unique content-derived run key, configuration, seed, checkpoint and result hashes |
| conditions | Fog parameter, selected threshold, sample count, elapsed time and integer confusion counts |
| observations | Per-condition/per-sample confusion counts; trigger requires the correct dataset's test split |
| artifacts | Experiment file references, types, sizes and hashes |

The `metrics` view derives precision, recall, F1, accuracy, lane IoU and mIoU. Zero denominators return SQL NULL. Ingestion rejects duplicate IDs/content, cross-split groups, missing observations, malformed matrices, non-finite parameters, contradictory aggregate metrics and inconsistent pixel totals. SQL constraints additionally enforce foreign keys, positive sizes, valid JSON and integer nonnegative counts.

## Lifecycle

1. Initialize the registry automatically on the first command.
2. Ingest a complete result directory. Parsing and validation finish before writes begin.
3. Repeat ingestion safely: unchanged dataset/run identities do not create duplicates.
4. Query the metrics view through read-only connections. The authorizer denies mutation and attached databases.
5. Check database/page integrity, foreign keys and per-image/aggregate reconciliation.
6. Export a portable JSON relational snapshot or create an online SQLite backup.
7. Recover into a new database, or import JSON into an empty database.

Commands and SQL examples appear in the main README. Backups use `Connection.backup`, so committed WAL contents are included. Copying only the main SQLite file while WAL mode is active is not an equivalent backup. Existing backup/export/recovery destinations are rejected. Recovery and JSON import verify relational and numeric integrity before reporting success. Keep backups outside the live database path and verify restores periodically.

Images, masks and checkpoints are referenced rather than stored as database blobs. Restoring the registry restores experiment records, not missing external datasets or artifact files. Preserve those separately with their recorded hashes.

The application uses parameterized SQL for values and a fixed table whitelist for exports/imports. SQLite files and exports can contain user-supplied dataset paths and configurations; generated working databases remain under ignored `runs/` by default. The public example contains only already-published experiment metadata.
