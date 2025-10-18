# Robocook
AI-powered recipe creator that adapts to what you have, what you need, and what you've already tried.


This app generates unique recipes augmented with context.  Also it can handle queries  to provide answers based on the context or save information with a standard format. The context that the app receives is: 
- Available ingredients 
- Preferences 
- Restrictions 

Also the app is capable of doing other task as:
- Get current time 
- Get current date 

### Knowledge / data base  
The DB used is PostgreSQL with the Pg vector extension. This is useful for embeddings operations required by the rag.

 Ingredients Table

| COLUMN      | TYPE        | DESCRIPTION                                    |
| ----------- | ----------- | ---------------------------------------------- |
| id          | SERIAL      | Unique identifier for each ingredient          |
| text        | TEXT        | Ingredient name or description (e.g., "apple") |
| embedding   | VECTOR(384) | Semantic embedding of the  text                |
| text_vector | TSVECTOR    | Text search vector                             |

Preferences table

| COLUMN      | TYPE        | DESCRIPTION                                                   |
| ----------- | ----------- | ------------------------------------------------------------- |
| id          | SERIAL      | Unique identifier for each ingredient                         |
| text        | TEXT        | Raw ingredient name or description (e.g., "Luna likes apple") |
| embedding   | VECTOR(384) | Semantic embedding vector of the preference text              |
| text_vector | TSVECTOR    | Text search vector in English                                 |

Restrictions table

| COLUMN      | TYPE        | DESCRIPTION                                                |
| ----------- | ----------- | ---------------------------------------------------------- |
| id          | SERIAL      | Unique identifier for each restriction                     |
| text        | TEXT        | Restriction statement (e.g., "Jhon is allergic to tomato") |
| embedding   | VECTOR(384) | Semantic embedding vector of the restriction text          |
| text_vector | TSVECTOR    | Text search vector in English                              |


Task table 

| COLUMN      | TYPE        | DESCRIPTION                            |
| ----------- | ----------- | -------------------------------------- |
| id          | SERIAL      | Unique identifier for each restriction |
| task_name   | TEXT        | Task type                              |
| description | TEXT        | Example of the task                    |
| embedding   | VECTOR(384) | Text search vector in English          |

### Embedding model
The embeddings will be made from names and short sentences. Thus using a 384-dimensional embedding model such as `all-MiniLM-L6-v2` provides a good balance of semantic accuracy and efficiency.

### Task distinction
For task distinction an embedding is generated from the query and the most similar task gathered from the knowledge base is gathered. The embedding is then reused for the classified task,  avoiding  duplication and maintains semantic coherence. This method is based on [1]

#### Hybrid Retrieval 
For vector similarity evaluations were conducted using cosine similarity, Euclidean distance (L2), and inner product (dot product), all tested with a top-k retrieval metric (k=3). Each method yielded identical results, with a Precision@3 score of 0.33 across all queries. Cosine distance was chosen because embedding magnitudes vary, and this metric focuses more on semantic similarity.

For lexical search, PostgreSQL’s full-text search (`ts_rank` with `to_tsquery`) is used for simulating BM25 to find documents that contain relevant keywords, giving them a score based on keyword frequency and importance. 

Finally, the system merges them with a weighted formula (`alpha * vector + (1 - alpha) * lexical`) to produce a balanced ranking that captures both keyword matches and semantic meaning. This formula was gathered from [2]. 
#### LLM
The model `llama3.2:3b-instruct-q4_0` was chosen because it is fine-tuned to follow instructions effectively, making it ideal for chatbot and assistant-style interactions. Its 4-bit quantization enables faster inference and reduced memory usage, which makes it ideal for local deployment and real-time responsiveness.

#### Task 
- Generation of recipes: Context is gathered by  hybrid retrieval and added to a prompt in order to send it to the LLM.
- Input queries: the request is classified by table, and rewrite it to the required format, then the data is saved in the DB. 
- Output queries: context is obtained by hybrid retrieval and given to the LLM to give a response based on the context. 
- Get current date or time: the date or time is obtained from the standard library datetime.

#### Query rewriting 
Distinct prompts were made for the query rewriting by the LLM and the results were:
The query intension  was affected, hardening the task distinction. Even when it was specified that things like ingredients or names in the query were important, the LLM in some cases delete this elements. The query rewriting is only useful in the input query task, so this method was added to the prompt of this task, but is avoided for the others.
### Cache :
The cache is used to retrieve embeddings but no for  questions - answer pairs because of the approach of the project. The app generates new recipes, so caching the generated recipes is redundant. A json file will be used as cache for efficiency. 
### ROS Humble usage 
This version of Ros 2 is used to create the package recipe_generation with both nodes rag_interaction.py and rag_node.py . Ros is used in both scripts to create the node, a subscriber and a publisher. This nodes communicate at this topics: 
### Dockers: 
There are 3 services used in this project: 
- Robocook_app: Contains the image of ubuntu 22.04 with Ros humble, python and the necesary dependencies to run the nodes. 
- Postgres: Contains the image of postgresql with the pgvector extention.  
- ollama: contains the ollama image with llama.3.2.

Refereces: 

[1] "Usage - Semantic Textual Similarity — Sentence-Transformers documentation," _sbert.net_. [Online]. Available: [https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html).

[2] A. D. Jassim, "A state-of-the-art survey on question-answering systems," _arXiv.org_, Oct. 21, 2022. [Online]. Available: [https://arxiv.org/abs/2210.11934](https://arxiv.org/abs/2210.11934).