# Text-to-SQL Data

This directory contains the data for the Text-to-SQL benchmark, supporting
[Spider2-lite](https://github.com/xlang-ai/Spider2) and [Bird](https://bird-bench.github.io/) datasets.

## Structure

```
data/text2sql/
├── README.md
├── spider2-lite/
│   ├── spider2-lite.jsonl                          # Spider2-lite questions (546 instances)
│   ├── golden_lite_spider_total.json               # Gold labels for evaluation
│   ├── sqlite/                                     # Local SQLite databases
│   │   └── <DB_Name>/
│   │       ├── <DB_Name>.sqlite                    # SQLite database file
│   │       └── <DB_Name>_db_summary.json           # Auto-generated schema summary
│   ├── bigquery/                                   # BigQuery dataset schemas
│   │   └── <dataset_group>/
│   │       └── <dataset_group>_bigquery_summary.json
│   └── snowflake/                                  # Snowflake database schemas
│       └── <DATABASE>/
│           └── <DATABASE>_db_summary.json
└── bird/
    ├── bird.jsonl                                  # Bird questions (129 instances)
    └── databases/
        └── <DB_Name>/
            ├── <DB_Name>.sqlite
            └── <DB_Name>_db_summary.json
```

## Included Example Databases

| Backend   | Benchmark    | Database   | Description                  |
|-----------|-------------|------------|------------------------------|
| SQLite    | Spider2-lite | **Pagila**     | DVD rental store (16 tables) |
| SQLite    | Bird         | **superhero**  | Superhero database (10 tables) |
| BigQuery  | Spider2-lite | austin     | Austin 311 service requests (schema only) |
| Snowflake | Spider2-lite | AUSTIN     | Austin 311 service requests (schema only) |

## Adding a New SQLite Database

1. Place the `.sqlite` file:
   ```
   data/text2sql/spider2-lite/sqlite/MyDB/MyDB.sqlite
   ```
2. The schema summary is auto-generated on first run, or generate manually:
   ```bash
   python -m src.indexing.extract_db_summary \
       --db-path data/text2sql/spider2-lite/sqlite/MyDB/MyDB.sqlite
   ```

## Adding a BigQuery Database

1. Generate the summary using the BigQuery extractor:
   ```bash
   python -m src.indexing.extract_db_summary_bigquery \
       --datasets project_id.dataset_id \
       --output-dir data/text2sql/spider2-lite/bigquery/my_dataset/
   ```

## Adding a Snowflake Database

1. Generate the summary using the Snowflake extractor:
   ```bash
   python -m src.indexing.extract_db_summary_snowflake \
       --databases MY_DATABASE \
       --output-dir data/text2sql/spider2-lite/snowflake/MY_DATABASE/
   ```

## Full Spider2-lite Data

To run the full benchmark, clone [Spider2](https://github.com/xlang-ai/Spider2)
and copy the databases into `spider2-lite/sqlite/`.
