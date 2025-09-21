import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
conn = psycopg2.connect(
     dbname='postgres',
    user='postgres',
    password='postgres',
    host='localhost',
    port='5432'
)

model = SentenceTransformer("all-MiniLM-L6-v2")  

def precision_at_k(results, expected, k):
    """Calculate Precision@K"""
    retrieved = [r[0] for r in results[:k]]
    relevant = set(expected)
    hits = len([doc for doc in retrieved if doc in relevant])
    return hits / k

def run_evaluation(TABLE_NAME, ids, queries, cur):
    
    i = 0
    for q in queries:
        q_emb = q_emb = np.array(model.encode(q), dtype=np.float32).tolist()

        print(f"Query: {q}")
        print(f"Expected ID: {ids[i]}")

        # Cosine similarity / l2 depending on postgres index
        cur.execute(f"""
            SELECT id, text
            FROM {TABLE_NAME}
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """, (q_emb, K))

        distance_results = cur.fetchall()
        p_cosine = precision_at_k(distance_results, ids[i] , K)

        # Dot product
        cur.execute(f"""
            SELECT id, text
            FROM {TABLE_NAME}
            ORDER BY embedding <#> (%s)::vector
            LIMIT %s;
        """, (q_emb, K))
        dot_results = cur.fetchall()
        p_dot = precision_at_k(dot_results, ids[i], K)

        # Print results
        print("Distance  results (l2 or cosine):")
        for r in distance_results:
            print(f"  {r[0]}: {r[1]}")
        print(f"Precision@{K} (Cosine): {p_cosine:.2f}")

        print("Dot product results:")
        for r in dot_results:
            print(f"  {r[0]}: {r[1]}")
        print(f"Precision@{K} (Dot): {p_dot:.2f}")
        i += 1

queries = [
    "Im Luna and i want something with apple", 
    "Give Sofia chicken recipe",
    "Sofia wants something with beans",
    "Im Mateo and I want something with fish",
    "Im Diego and i want something with onion"
]
expected_ingredients_ids =  [
    [46, 47, 44],  # Prompt 1: Apple → Apple, Mango, Plantain
    [21, 22, 10],  # Prompt 2: Chicken → Chicken, Beef, Onion
    [0, 3, 25],    # Prompt 3: Beans → Beans, Corn tortillas, Fresh cheese
    [23, 39, 5],   # Prompt 4: Fish → Fish, Potatoes, Vegetable oil
    [9, 10, 8]     # Prompt 5: Onion → Onion, Tomato, Garlic
]

expected_preferences_ids = [
    [0, 5, 4],     # Luna: apples, mangoes, chocolate cake
    [22, 27, 28],  # Sofia: fried chicken, bean tacos, burritos
    [27, 28, 23],  # Sofia: bean tacos, burritos, grilled cheese
    [16, 17, 11],  # Mateo: fish sticks, potatoes, tacos
    [36, 31, 37]   # Diego: onion rings, enchiladas, tomatoes
]

K = 3  # top-k results

cur = conn.cursor()
print("Evaluating on ingredients table")
run_evaluation("ingredients", expected_ingredients_ids, queries,cur)
print("Evaluating on preferences table")
run_evaluation("preferences", expected_preferences_ids,  queries,cur)
cur.close()
conn.close()