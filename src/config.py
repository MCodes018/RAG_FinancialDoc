
from pathlib import Path
from dotenv import load_dotenv
import os   

load_dotenv()
# Project Directories

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

CHUNK_DIR = DATA_DIR / "chunks"

VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# SEC Configuration
SEC_USER_NAME = os.getenv("SEC_USER_NAME")
SEC_USER_EMAIL = os.getenv("SEC_USER_EMAIL")

# Embedding Model
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Chunking Parameters
FIXED_CHUNK_SIZE = 800
FIXED_CHUNK_OVERLAP = 100

# Retrieval Parameters
TOP_K = 5

#LLM
OLLAMA_MODEL= "llama3.2:3b"