import psycopg2
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded")

# PostgreSQL connection
conn = psycopg2.connect(
    dbname='postgres',
    user='postgres',
    password='postgres',
    host='localhost',
    port='5432'
)
cursor = conn.cursor()
print("Connected to PostgreSQL")

# Paths 
kb_files = {
    'ingredients': r'ingest\ingredients.txt',
    'preferences': r'ingest\preferences.txt',
    'restrictions': r'ingest\restrictions.txt'
}

def ingest_file(table_name, file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        embedding = model.encode(line).tolist()
        cursor.execute(
            f"INSERT INTO {table_name} (text, embedding) VALUES (%s, %s)",
            (line, embedding)
        )
    print(f"Ingested {len(lines)} entries into '{table_name}'.")

for table, path in kb_files.items():
    ingest_file(table, path)

conn.commit()
cursor.close()
conn.close()
