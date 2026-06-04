from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db" / "conferences.db"
DATA_DIR = BASE_DIR / "data"

# LLM
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# pipeline
N_YEARS = 3
