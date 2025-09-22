CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE ingredients (
                             id SERIAL PRIMARY KEY,
                             text TEXT NOT NULL,
                             embedding VECTOR(384)
);

CREATE TABLE preferences (
                             id SERIAL PRIMARY KEY,
                             text TEXT NOT NULL,
                             embedding VECTOR(384)
);

CREATE TABLE restrictions (
                              id SERIAL PRIMARY KEY,
                              text TEXT NOT NULL,
                              embedding VECTOR(384)
);


ALTER TABLE ingredients ADD COLUMN text_vector tsvector;
UPDATE ingredients SET text_vector = to_tsvector('english', text);
CREATE INDEX ingredients_text_idx ON ingredients USING GIN (text_vector);

ALTER TABLE preferences ADD COLUMN text_vector tsvector;
UPDATE preferences SET text_vector = to_tsvector('english', text);
CREATE INDEX preferences_text_idx ON preferences USING GIN (text_vector);

ALTER TABLE restrictions ADD COLUMN text_vector tsvector;
UPDATE restrictions SET text_vector = to_tsvector('english', text);
CREATE INDEX restrictions_text_idx ON restrictions USING GIN (text_vector);