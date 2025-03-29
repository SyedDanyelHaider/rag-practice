
import os

DB_USERNAME = 'ibtehajali'
DB_PASSWORD = 'password'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'mydatabase'    

DATABASE_URL = f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

EMBEDDINGS_MODEL_NAME = 'all-MiniLM-L6-v2'  

LOG_FILE = 'logs/pipeline.log'
LOG_LEVEL = 'INFO'

POD_ID = "tlikdz9ypqfwaj" 
PORT = "11434"

OLLAMA_BASE_URL = f"https://{POD_ID}-{PORT}.proxy.runpod.net/api/generate"


