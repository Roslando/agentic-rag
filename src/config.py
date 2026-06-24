from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
MARKDOWN_DIR = DATA_DIR / "markdown_docs"
QDRANT_PATH = os.getenv("QDRANT_PATH", str(DATA_DIR / "qdrant_db"))
PARENT_STORE_DIR = DATA_DIR / "parent_store"

# ── Qdrant ─────────────────────────────────────────────────────────────────────
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "company_docs")

# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
EMBEDDING_DIM = 768

# ── Chunking ───────────────────────────────────────────────────────────────────
CHILD_CHUNK_SIZE = 400      # tokens cibles pour les chunks enfants
CHILD_CHUNK_OVERLAP = 80
PARENT_CHUNK_SIZE = 1500    # tokens cibles pour les chunks parents
PARENT_CHUNK_OVERLAP = 100

# ── Retrieval ──────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 10        # nb d'enfants récupérés (→ + de parents distincts après dédup)

# ── Reranking (cross-encoder) ───────────────────────────────────────────────────
# On récupère un large vivier d'enfants, on les re-note avec un cross-encoder
# (vraie pertinence 0-1), on filtre par seuil, puis on étend aux parents.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # léger (~80 Mo), anglais
RERANK_CANDIDATES = 12      # enfants candidats récupérés AVANT reranking
                            # (12 au lieu de 20 : ~40% de paires en moins à scorer
                            #  par le cross-encoder → recherche plus rapide, sans
                            #  perte notable de recall car le top domine)
RERANK_THRESHOLD = 0.15     # score sigmoïde min (0-1) ; testé : hors-sujet≈0.00,
                            # pertinent 0.27-0.95 → 0.15 sépare proprement
RERANK_KEEP = 3             # nb max de parents donnés au LLM après filtrage
                            # (3 au lieu de 5 : limite la taille du prompt — le
                            #  top 3 reranké couvre l'essentiel)

# ── Budget de contexte (anti-dépassement de prompt) ─────────────────────────────
# Chaque recherche ne renvoie au LLM que ce nb de tokens max de documents (docs
# entiers, meilleur-classé d'abord, sans couper un bloc de code). Avec jusqu'à
# MAX_SEARCH_ITERATIONS recherches empilées dans l'historique, le prompt total
# reste sous la limite gratuite d'OpenRouter (~13K tokens). 3 × 3000 + overhead.
SEARCH_CONTEXT_TOKEN_BUDGET = 3000

# ── LLM ────────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

# OpenRouter fallback (compatible OpenAI API)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4048"))

# ── Agent ──────────────────────────────────────────────────────────────────────
MAX_SEARCH_ITERATIONS = 3   # nombre max de tours de recherche avant réponse forcée
RECURSION_LIMIT = 25

# ── Observabilité ──────────────────────────────────────────────────────────────
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
ENABLE_LANGFUSE = bool(LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY)
