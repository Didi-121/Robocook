#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sentence_transformers import SentenceTransformer
import ollama
import psycopg2
import logging
import json
import os
import time 
from datetime import datetime

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
            host='postgres',
            port='5432'
        )
        self.cur = self.conn.cursor()
        logger.info("Connected to the database successfully")
        self.tables = ['ingredients', 'preferences', 'restrictions']
        
        self.top_k = 3
        self.alpha = 0.5  # Weighting factor for hybrid retrieval

        self.ollama_client = ollama.Client(host='http://ollama:11434') 
        self.ollama_model = "llama3.2:3b-instruct-q4_0" 

        cache_dir = "/workspace/recipe_generation/scripts/cache.json"
        if not os.path.exists(cache_dir):
            with open(cache_dir, "w") as f:
                json.dump({}, f)
        with open(cache_dir, "r") as f:
            self.cache = json.load(f)

    def initialize_ollama(self):
        """Initialize Ollama client with retry logic"""
        self.ollama_client = ollama.Client(host='http://ollama:11434') 
        self.ollama_model = "llama3.2:3b-instruct-q4_0"
        
        max_retries = 15
        for attempt in range(max_retries):
            try:
                models = self.ollama_client.list()
                available_models = [model['name'] for model in models['models']]
                
                if  self.ollama_model in available_models:
                    logger.info(f"Ollama initialized successfully. Model {self.ollama_model} is available")
                    return
                else:
                    logger.warning(f"Model {self.ollama_model} not found. Available: {available_models}")
                    
            except Exception as e:
                logger.debug(f"Ollama connection attempt {attempt + 1}/{max_retries}: {e}")
                
            if attempt == max_retries - 1:
                raise Exception(f"Could not connect to Ollama after {max_retries} attempts")
                
            time.sleep(3)

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

        response = self.ollama_client.chat(
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
        response = self.ollama_client.chat(
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
            f"You will answer a question.\n"
            f"Use only the information provided in the context below to answer.\n"
            f"Do not invent ingredients, preferences, or restrictions that are not listed.\n"
            f"Respond clearly and concisely based on the context.\n\n"
            f"Context:\n"
            f"Available ingredients: {' '.join(context.get('ingredients', []))}\n"
            f"Preferences: {' '.join(context.get('preferences', []))}\n"
            f"Restrictions: {' '.join(context.get('restrictions', []))}\n\n"
            f"the question is: {self.raw_query}"
        )
        logger.info(f"Output query prompt:\n{prompt}")
        response = self.ollama_client.chat(
            model=self.ollama_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"]
    
    def get_current_time(self):
        return str( datetime.now().strftime("%H:%M:%S") ) 
    
    def get_current_date(self):
        return str ( datetime.now().strftime("%d-%m-%Y") )

    def start(self, input_text):
        self.raw_query = input_text
        self.query_embedding = self.get_embedding(self.raw_query)
        
        task = self.classify_task()
        logger.debug(f"Classified task: {task}")

        if task == "generate_recipe":
            return self.generate_recipe()
        elif task == "input_query":
            return self.natural_language_input_query()
        elif task == "output_query":
            return self.natural_language_output_query()
        elif task == "get_current_time":
            logger.info(f"Current time is: {self.get_current_time()}")
            return self.get_current_time()
        elif task == "get_current_date":
            logger.info(f"Current date is: {self.get_current_date()}")
            return self.get_current_date()

class RagNode(Node):
    def __init__(self):
        super().__init__('rag_node')
        logger.info("Initializing ROS node")
        self.rag = Rag()
        self.sub = self.create_subscription(String, 'rag/request', self.handle_request, 10)
        self.pub = self.create_publisher(String, 'rag/response', 10)

    def handle_request(self, msg: String):
        try:
            logger.info(f"Received request: {msg.data}")
            data = msg.data.strip().lower()
            if data == 'exit':
                logger.info('Exit command received, shutting down node...')
                rclpy.shutdown()
                return
            response = self.rag.start(data)
            self.pub.publish(String(data=response))
            logger.info("Response published: " + response)

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            self.pub.publish(String(data="An error occurred"))

def main():
    rclpy.init()
    node = RagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("Exit by keyboard interrupt")
    finally:
        logger.info("Shutting down; closing DB")
        node.rag.cur.close()
        node.rag.conn.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
