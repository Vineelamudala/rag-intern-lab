-- Runs once, when the Docker volume is first created.
-- Chunk tables are created by `python scripts/ingest.py` (or `--init-db`),
-- because their vector dimension depends on the configured embedding model.
CREATE EXTENSION IF NOT EXISTS vector;
