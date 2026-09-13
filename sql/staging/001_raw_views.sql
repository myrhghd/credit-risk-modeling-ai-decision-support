CREATE SCHEMA IF NOT EXISTS staging;

CREATE OR REPLACE VIEW staging.base AS
SELECT *
FROM raw.base;

CREATE OR REPLACE VIEW staging.raw_ingestion_runs AS
SELECT
    run_id,
    started_at,
    finished_at,
    run_status,
    message
FROM raw.ingestion_run;

CREATE OR REPLACE VIEW staging.raw_logical_tables AS
SELECT
    run_id,
    logical_table,
    load_table,
    source_row_count,
    loaded_row_count,
    logical_table_status,
    started_at,
    finished_at,
    message
FROM raw.ingestion_logical_table;

CREATE OR REPLACE VIEW staging.raw_source_files AS
SELECT
    run_id,
    logical_table,
    source_file,
    source_row_count,
    loaded_row_count,
    source_file_status,
    started_at,
    finished_at,
    message
FROM raw.ingestion_source_file;
