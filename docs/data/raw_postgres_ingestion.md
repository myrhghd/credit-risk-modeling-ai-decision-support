# PostgreSQL Raw Ingestion

This stage loads the Home Credit training Parquet files into PostgreSQL `raw` tables.
It also creates the project schemas `raw`, `staging`, `features`, `analytics`, and
`monitoring`, but only `raw` is populated.

## Source To Table Mapping

Files are grouped by logical table. The `train_` prefix is removed, and final
partition suffixes are removed when present:

| Source pattern | Raw table |
| --- | --- |
| `train_base.parquet` | `raw.base` |
| `train_static_0_*.parquet` | `raw.static_0` |
| `train_static_cb_0.parquet` | `raw.static_cb_0` |
| `train_applprev_1_*.parquet` | `raw.applprev_1` |
| `train_applprev_2.parquet` | `raw.applprev_2` |
| `train_credit_bureau_a_1_*.parquet` | `raw.credit_bureau_a_1` |
| `train_credit_bureau_a_2_*.parquet` | `raw.credit_bureau_a_2` |
| `train_credit_bureau_b_1.parquet` | `raw.credit_bureau_b_1` |
| `train_credit_bureau_b_2.parquet` | `raw.credit_bureau_b_2` |
| `train_debitcard_1.parquet` | `raw.debitcard_1` |
| `train_deposit_1.parquet` | `raw.deposit_1` |
| `train_other_1.parquet` | `raw.other_1` |
| `train_person_1.parquet` | `raw.person_1` |
| `train_person_2.parquet` | `raw.person_2` |
| `train_tax_registry_a_1.parquet` | `raw.tax_registry_a_1` |
| `train_tax_registry_b_1.parquet` | `raw.tax_registry_b_1` |
| `train_tax_registry_c_1.parquet` | `raw.tax_registry_c_1` |

## Schema Strategy

Partitioned files must have identical column names in the same order and identical
Arrow data types before consolidation. Incompatible partitions fail before loading.
PostgreSQL DDL is generated from Arrow metadata and uses native column names,
including `case_id`, `num_group1`, `num_group2`, date fields, `WEEK_NUM`, `MONTH`,
`target`, numeric fields, categoricals, and booleans.

Raw source tables are `UNLOGGED`. This is appropriate for the local raw Home Credit
layer because the Parquet source is immutable and reproducible, local development
cares more about reducing WAL and disk amplification than crash durability, and
downstream curated or production-serving layers can use regular logged tables.

The loader keeps metadata in regular logged tables:

- `raw.ingestion_run`: one row per ingestion run.
- `raw.ingestion_logical_table`: one row per logical table in a run.
- `raw.ingestion_source_file`: one row per source file in a logical table.

`raw.ingestion_source_file` has a primary key on
`(run_id, logical_table, source_file)`, so one run cannot silently duplicate a
successful source file record.

## Atomic Load Strategy

Each logical table is loaded into a fresh `raw.__load_<table>_<run_id>` table. All
source partitions must load and reconcile there before the permanent
`raw.<logical_table>` is changed. Promotion happens by renaming the prior permanent
table aside, renaming the completed load table to the permanent name, and dropping
the prior table inside one PostgreSQL transaction.

If loading or reconciliation fails, the permanent table is not replaced. Incomplete
source files and the logical table are marked failed, and the failed load table is
dropped. Leftover managed load tables can be inspected or removed with the cleanup
command below.

## Commands

Run from the repository root inside the existing `credit-risk-ai` conda environment.
Connection settings are read from `.env` without printing them.

```bash
python scripts/raw_postgres.py list-tables
python scripts/raw_postgres.py init-schemas
python scripts/raw_postgres.py reset --table other_1
python scripts/raw_postgres.py ingest --table other_1 --batch-size 50000
python scripts/raw_postgres.py validate --table other_1
python scripts/raw_postgres.py cleanup-loads
```

After tests and smoke ingestion pass, run the full raw ingestion explicitly:

```bash
python scripts/raw_postgres.py reset
python scripts/raw_postgres.py ingest --batch-size 50000
python scripts/raw_postgres.py validate
```

The loader does not expose normal append mode for raw source tables. To reload
`raw.other_1` from the earlier smoke test, run:

```bash
python scripts/raw_postgres.py reset --table other_1
```

## Memory Safety

The implementation reads Parquet files with `pyarrow.parquet.ParquetFile.iter_batches`
and streams each batch to PostgreSQL through `COPY FROM STDIN`. It does not load a
full large table into pandas or memory. The recommended local full-load batch size
is `50000` rows per Arrow batch; the CLI keeps `--batch-size` configurable.

## Validation

After each logical table load, reconciliation checks:

- PostgreSQL row count equals Parquet metadata row count.
- PostgreSQL columns match the expected Arrow schema columns in order.
- Key columns present in the source schema also exist in PostgreSQL.

The CLI prints a final row and column summary for each validated table.
