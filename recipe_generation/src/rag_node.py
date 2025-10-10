from sentence_transformers import SentenceTransformer
import ollama
import psycopg2
import logging
import json
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Rag:
    def __init__(self): 
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info(f"Loaded embedding model")
       
        self.conn = psycopg2.connect(
            dbname='knowledge_bases',
            user='postgres',
            password='postgres',
            host='localhost',
            port='5432'
        )
        self.cur = self.conn.cursor()
        logger.info("Connected to the database successfully")
        self.tables = ['ingredients', 'preferences', 'restrictions']
        
        self.top_k = 3
        self.alpha = 0.5  # Weighting factor for hybrid retrieval
        self.ollama_model = "llama3.2:3b-instruct-q4_0" 

        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        with open("cache.json", "r") as f:
            self.cache = json.load(f)

    def save_cache(self):
        with open("cache.json", "w") as f:
            json.dump(self.cache, f)

    def get_embedding(self, text: str):
        if text in self.cache:
            logger.debug("Using cached embedding")
            return self.cache[text]
        logger.info("Computing new embedding")
        embedding = self.embedding_model.encode(text).tolist()
        self.cache[text] = embedding
        self.save_cache()
        return embedding

    def classify_task(self) -> str:
        self.cur.execute("""
            SELECT task_name
            FROM task
            ORDER BY  embedding <=> %s::vector
            LIMIT 1;
        """, (self.query_embedding,))
        task = self.cur.fetchone()[0]
        return task

    def hybrid_retrieval(self) -> dict[str,str]:
        results_per_table: dict[str, list[str]] = {}
        for table in self.tables:  
            #Lexical search using ts-rank simulating BM25
            self.cur.execute(f"""
                SELECT id, text, ts_rank(text_vector, plainto_tsquery('english', %s)) AS rank
                FROM {table}
                WHERE text_vector @@ plainto_tsquery('english', %s)
                LIMIT %s;
            """, (self.raw_query, self.raw_query, self.top_k))
            lexical_results = self.cur.fetchall()

            # Vector similarity (dot product)
            self.cur.execute(f"""
                SELECT id, text, (embedding <#> (%s)::vector) * -1 AS score
                FROM {table}
                ORDER BY embedding <#> (%s)::vector
                LIMIT %s;
            """, (self.query_embedding, self.query_embedding, self.top_k))
            vector_results = self.cur.fetchall()

            # Merge results as { doc_id: (text, lexical_rank, vector_score) }
            scores = {}
            for doc_id, text, rank in lexical_results:
                scores[doc_id] = {"text": text, "lexical": rank, "vector": 0}

            for doc_id, text, score in vector_results:
                if doc_id not in scores:
                    scores[doc_id] = {"text": text, "lexical": 0, "vector": score}
                else:
                    scores[doc_id]["vector"] = score

            # Combine scores
            combined = []
            for doc_id, vals in scores.items():
                #Explained formula in documentation
                final_score = self.alpha * vals["vector"] + (1 - self.alpha) * vals["lexical"]
                combined.append((doc_id, vals["text"], final_score))

            #Save only the top texts in dict
            combined.sort(key=lambda x: x[2], reverse=True)

            #ignore doc_id and score
            results_per_table[table] = [text for _, text, _ in combined[:self.top_k]]

        return results_per_table
        
    def generate_recipe(self) -> str:
        context = self.hybrid_retrieval()
        prompt = (
            "Generate only one detailed cooking recipe.\n\n"
            f"Available ingredients: {' '.join(context.get('ingredients', []))}\n"
            f" {' '.join(context.get('preferences', []))}\n"
            f" {'  '.join(context.get('restrictions', []))}\n\n"
        )
        logger.debug(f"Recipe generation prompt:\n{prompt}")

        response = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"]
    
    def natural_language_input_query(self) -> str:
        prompt = (
            f"You will receive a natural language request. Your task is to:\n"
            f"1. Identify which table from the following list best matches the request: {', '.join(self.tables)}.\n"
            f"2. Rewrite the request using the correct format based on its type:\n"
            f"   - If the request expresses a restriction (e.g., allergies or dislikes), use: 'dont use <ingredient> for <person>'\n"
            f"   - If the request expresses a preference, use: '<person> likes <ingredient>'\n"
            f"   - If the request asks about available ingredients, use: '<ingredient>'\n\n"
            f"Return your answer in this exact format:\n"
            f"<table_name>,<rewritten_request>\n"
            f"Do not explain your reasoning. Do not include numbering, extra text, or line breaks.\n"
            f"Only return the string.\n\n"
            f"Request: {self.raw_query}"
        )
        response = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        table = response["message"]["content"].split(',')[0].strip()
        rewritten_request = response["message"]["content"].split(',')[1].strip()
        embedding = self.get_embedding(rewritten_request)
        self.cur.execute(
            f"""
            INSERT INTO {table} (text, embedding, text_vector)
            VALUES (%s, %s, to_tsvector('english', %s))
            """,
            (rewritten_request, embedding, rewritten_request)
        )
        self.conn.commit()  

        return f"Inserted {rewritten_request} into {table}"

    def natural_language_output_query(self) -> str:
        context = self.hybrid_retrieval()
        prompt = (
            f"You will receive a natural language request.\n"
            f"Use only the information provided in the context below to answer.\n"
            f"Do not invent ingredients, preferences, or restrictions that are not listed.\n"
            f"Respond clearly and concisely based on the context.\n\n"
            f"Context:\n"
            f"Available ingredients: {' '.join(context.get('ingredients', []))}\n"
            f"Preferences: {' '.join(context.get('preferences', []))}\n"
            f"Restrictions: {' '.join(context.get('restrictions', []))}\n\n"
            f"Request: {self.raw_query}"
        )
        response = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"]

    def start(self, input_text):
        self.raw_query = input_text
        self.query_embedding = self.get_embedding(self.raw_query)
        
        task = self.classify_task()
        logger.debug(f"Classified task: {task}")

        if task == "generate_recipe":
            logger.info(self.generate_recipe())
        elif task == "input_query":
            logger.info(self.natural_language_input_query())
        elif task == "output_query":
            logger.info(self.natural_language_output_query())
        
            
rag = Rag()

if __name__ == "__main__":
    rag = Rag()
    try:
        while True:
            user_input = input("Enter your request (or 'exit' to quit): ")
            if user_input.lower() == 'exit':
                break
            rag.start(user_input)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print("An error occurred. Please try again.")
    finally:
        logger.debug("Closing database connection")
        rag.cur.close()
        rag.conn.close()

# Im Luna and i want something with apple 