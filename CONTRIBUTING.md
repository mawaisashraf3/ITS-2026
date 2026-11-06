# Contributing

Use Python 3.12, install the pinned environment, and run `python -m pytest -q` before submitting changes. Keep computational modules small and document numerical assumptions. Format Python with Ruff; static checks use `ruff check --select F,E9 rmldnet tests scripts`.

New algorithms need meaningful numerical or integration checks. Dataset changes must preserve licenses, record hashes and splits, and prevent training/test overlap. Keep raw datasets and working databases outside tracked source. Publish measured outputs only when commands, configurations, counts and selection procedures are recorded. Label synthetic data and controlled corruptions explicitly.

Database schema changes require a versioned migration, foreign-key tests, rollback tests and backup/restore verification. Never silently reinterpret an existing schema. Avoid changing previously published measurements without preserving a clear correction record.

Describe the concrete behavior change, validation performed, and remaining limitations in contributions. External code, model weights and data require appropriate attribution and compatible terms.
