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

