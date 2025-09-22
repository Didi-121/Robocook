import psycopg2
import re

# Conexión
conn = psycopg2.connect(
    dbname='postgres',
    user='postgres',
    password='postgres',
    host='localhost',
    port='5432'
)
cur = conn.cursor()

TABLES = ['ingredients', 'preferences', 'restrictions']
QUERIES = [
    "Im Luna and i want something with apple", 
    "Give Sofia chicken recipe",
    "Sofia wants something with beans",
    "Im Mateo and I want something with fish",
    "Im Diego and i want something with onion"
]

def extract_tsquery(text):
    tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {'im', 'and', 'i', 'want', 'something', 'give', 'with', 'recipe', 'mateo', 'sofia', 'luna', 'diego'}
    keywords = [t for t in tokens if t not in stopwords]
    return ' & '.join(keywords)

for query in QUERIES:
    tsquery = extract_tsquery(query)
    print(f"\nQuery: '{query}' → tsquery: '{tsquery}'")

    for table in TABLES:
        sql = f"""
            SELECT id, text, ts_rank(text_vector, to_tsquery('english', %s)) AS rank
            FROM {table}
            WHERE text_vector @@ to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT 5;
        """
        cur.execute(sql, (tsquery, tsquery))
        results = cur.fetchall()

        print(f"Table: {table}")
        if results:
            for doc_id, text, rank in results:
                print(f"ID: {doc_id} | Rank: {rank:.4f} | Text: {text}")
        else:
            print("No lexical matches found.")

cur.close()
conn.close()