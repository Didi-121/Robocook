from urllib import response
from sentence_transformers import SentenceTransformer
import ollama
import psycopg2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Rag:
    def __init__(self): 
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info(f"Loaded embedding model")
       
        self.conn = psycopg2.connect(
            dbname='postgres',
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
        self.ollama_model = "llama3.2"  

    def rewrite_prompt(self, prompt: str) -> str:
        pass

    def cache_search(self, prompt:str) -> str:
        pass

    def clasify_task(self, query: str) -> str:
        pass

    def hybrid_retrieval(self) -> dict[str,str]:
        results_per_table: dict[str, list[str]] = {}
        for table in self.tables:  # self.tables = ['ingredients', 'preferences', 'restrictions']
            #Lexical search 
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

            # Merge results
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
                final_score = self.alpha * vals["vector"] + (1 - self.alpha) * vals["lexical"]
                combined.append((doc_id, vals["text"], final_score))

            combined.sort(key=lambda x: x[2], reverse=True)
            #Save only the top texts in dict
            results_per_table[table] = [text for _, text, _ in combined[:self.top_k]]

        return results_per_table
        
    def generate_recipe(self) -> str:
        context = self.hybrid_retrieval()
        prompt = (
            "Generate a detailed cooking recipe.\n\n"
            f"Ingredients (must use): {', '.join(context.get('ingredients', []))}\n"
            f"Preferences (try to satisfy): {', '.join(context.get('preferences', []))}\n"
            f"Restrictions (avoid): {', '.join(context.get('restrictions', []))}\n\n"
        )
        logger.debug(f"Recipe generation prompt:\n{prompt}")
        response = ollama.chat(
            model=self.ollama_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"]

    def start(self):
        try: 
            self.raw_query = input("Input: ")
            """
            query = rag.rewtite_prompt(query)
            cache_score, similar = rag.cache_search(query)
            if cache_score > n:
                response = similar
            """
            self.query_embedding = self.embedding_model.encode(self.raw_query).tolist()
            task = "generate_recipe" #self.clasify_task(query_embedding)
            logger.debug(f"Classified task: {task}")

            if task == "generate_recipe":
                print(self.generate_recipe())
                return 
            elif task == "input_query":
                return 
            elif task == "output_query":
                return
            else:
                return "I'm sorry, I cannot assist with that request."
            
        finally:
            self.cur.close()
            self.conn.close()

rag = Rag()
rag.start()
# Im Luna and i want something with apple 