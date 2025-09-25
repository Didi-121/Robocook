from sentence_transformers import SentenceTransformer
import psycopg2
import json
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

truncate = input("Want to truncate existing tables? (y/n): ")

if truncate.lower() == 'y':
    cursor.execute("TRUNCATE TABLE ingredients, preferences, restrictions, task RESTART IDENTITY;")
    print("Tables truncated.")

print("Ingesting kb files...")
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
            f"""
            INSERT INTO {table_name} (text, embedding, text_vector)
            VALUES (%s, %s, to_tsvector('english', %s))
            """,
            (line, embedding, line)
        )
        
    print(f"Ingested {len(lines)} entries into '{table_name}'.")

for table, path in kb_files.items():
    ingest_file(table, path)

print("Ingesting tasks...")

with open(r"ingest/task.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Insert embeddings into table
for task_name, examples in data.items():
    if isinstance(examples, str):
        examples = [examples]  # handle single string case

    # If there will be more than one example
    for example in examples:
        embedding = model.encode(example).tolist()
        cursor.execute(
            """
            INSERT INTO task (task_name, description, embedding)
            VALUES (%s, %s, %s)
            """,
            (task_name, example, embedding)
    )

print("Ingestion complete")

conn.commit()
cursor.close()
conn.close()
