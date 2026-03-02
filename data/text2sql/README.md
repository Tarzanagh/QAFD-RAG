# Text-to-SQL Data

This directory contains the data for the Text-to-SQL benchmark, following the
[Spider2-lite](https://github.com/xlang-ai/Spider2) directory layout.

## Structure

```
data/text2sql/
├── spider2-lite.jsonl              # Benchmark questions (instance_id, db, question)
├── README.md
└── databases/
    ├── sqlite/                     # Local SQLite databases
    │   └── <DB_Name>/
    │       ├── <DB_Name>.sqlite            # SQLite database file
    │       └── <DB_Name>_db_summary.json   # Auto-generated schema summary
    │
    ├── bigquery/                   # BigQuery dataset schemas
    │   └── <dataset_group>/
    │       └── <dataset_group>_bigquery_summary.json   # Schema summary
    │
    └── snowflake/                  # Snowflake database schemas
        └── <DATABASE>/
            └── <DATABASE>_db_summary.json              # Schema summary
```

## Sample Databases

One example db_summary is included per backend type:

| Backend   | Database | Summary file                              | Description                  |
|-----------|----------|-------------------------------------------|------------------------------|
| sqlite    | Pagila   | `Pagila/Pagila_db_summary.json`           | DVD rental store (16 tables) |
| bigquery  | austin   | `austin/austin_bigquery_summary.json`     | Austin 311 service requests  |
| snowflake | AUSTIN   | `AUSTIN/AUSTIN_db_summary.json`           | Austin 311 service requests  |

## Adding a New SQLite Database

1. Place the `.sqlite` file:
   ```
   data/text2sql/databases/sqlite/MyDB/MyDB.sqlite
   ```
2. The schema summary is auto-generated on first run, or generate manually:
   ```bash
   python -m src.indexing.extract_db_summary \
       --db-path data/text2sql/databases/sqlite/MyDB/MyDB.sqlite
   ```

## Adding a BigQuery Database

1. Generate the summary using the BigQuery extractor:
   ```bash
   python -m src.indexing.extract_db_summary_bigquery \
       --datasets project_id.dataset_id \
       --output-dir data/text2sql/databases/bigquery/my_dataset/
   ```

## Adding a Snowflake Database

1. Generate the summary using the Snowflake extractor:
   ```bash
   python -m src.indexing.extract_db_summary_snowflake \
       --databases MY_DATABASE \
       --output-dir data/text2sql/databases/snowflake/MY_DATABASE/
   ```

## Full Spider2-lite Data

To run the full benchmark, clone [Spider2](https://github.com/xlang-ai/Spider2)
and symlink or copy the databases into `databases/`.
