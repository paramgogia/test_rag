"""Central configuration for the Infosys RAG chatbot."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
OUTPUTS_DIR = ROOT_DIR / "outputs"

VECTORSTORE_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Models
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIM = 768          # request 768-dim output via MRL
CHAT_MODEL = "gemini-2.5-flash"

# Chunking
CHUNK_SIZE = 2000      # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks
# Retrieval
TOP_K = 6              # chunks per query
MAX_HISTORY_TURNS = 6  # how many past turns to feed back

# Vector store filenames
FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"

# Source document map (display name -> filename in /data)
SOURCE_FILES = {
    "Annual Report FY25": "infosys-ar-25.pdf",
    "Q1 FY26 Press Release": "ifrs-usd-press-release_q1.pdf",
    "Q2 FY26 Press Release": "ifrs-usd-press-release_q2.pdf",
    "Q3 FY26 Press Release": "ifrs-usd-press-release_q3.pdf",
    "Q4 FY26 Press Release": "ifrs-usd-press-release_q4.pdf",
    "Investor Sheet (Multi-year Financials)": "investor-sheet.xls",
    "BSE 500209 Stock Price FY26": "500209.csv",
}
