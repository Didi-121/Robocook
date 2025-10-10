-- Extensiones para manejo de vectores
CREATE EXTENSION IF NOT EXISTS vector;

-- INGREDIENTS
CREATE TABLE ingredients (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(384),
    text_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX ingredients_text_idx ON ingredients USING GIN (text_vector);

-- PREFERENCES
CREATE TABLE preferences (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(384),
    text_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX preferences_text_idx ON preferences USING GIN (text_vector);

-- RESTRICTIONS
CREATE TABLE restrictions (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(384),
    text_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX restrictions_text_idx ON restrictions USING GIN (text_vector);

-- TASKS
CREATE TABLE task (
                    id SERIAL PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    description TEXT,
                    embedding VECTOR(384)
);