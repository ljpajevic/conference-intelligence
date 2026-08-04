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
# Previously llama-3.3-70b-versatile — deprecated by Groq on June 17, 2026
GROQ_MODEL = "openai/gpt-oss-120b"

# pipeline
# how many recent years of historical papers to scrape, used only for CoNEXT
N_YEARS = 3
