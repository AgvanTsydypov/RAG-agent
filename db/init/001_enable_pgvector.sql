-- Runs automatically the FIRST time the database volume is initialized.
-- Enables the pgvector extension so the `vector` column type becomes available.
CREATE EXTENSION IF NOT EXISTS vector;