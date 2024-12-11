# ----------------------------- Import Required Modules -----------------------------
import os
import logging
from langchain_community.embeddings import SentenceTransformerEmbeddings  # Embedding Model
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader  # For Loading Documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For Splitting Texts
from langchain_community.vectorstores import Qdrant  # For Qdrant Vector Store Management

# ----------------------------- Configuration Section -----------------------------
# Shared Configurations (Can align with `app.py` for consistency)
DATA_DIR = "data/"  # Directory containing input PDF files.
FILE_PATTERN = "**/*.pdf"  # File pattern to scan for documents.
CHUNK_SIZE = 700  # The size of each text chunk.
CHUNK_OVERLAP = 70  # Overlap between consecutive chunks.
QDRANT_URL = "http://localhost:6333"  # URL for the Qdrant vector database.
COLLECTION_NAME = "vector_db_2"  # Name of the Qdrant collection for vectors.
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"  # Embedding model optimized for biomedical content.

# ----------------------------- Logging Configuration -----------------------------
# Unified logging format for better clarity and troubleshooting.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ----------------------------- Function Definitions -----------------------------
def load_documents(data_dir: str, pattern: str) -> list:
    """
    Loads PDF documents from the specified directory using DirectoryLoader.
    Works similarly to document ingestion in `app.py`.

    Args:
        data_dir (str): Path to the data directory.
        pattern (str): File glob pattern (e.g., '**/*.pdf').

    Returns:
        list: List of loaded documents.
    """
    logger.info(f"📄 Loading documents from '{data_dir}' matching '{pattern}'...")
    loader = DirectoryLoader(data_dir, glob=pattern, show_progress=True, loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    logger.info(f"✅ Successfully loaded {len(documents)} document(s).")
    return documents


def split_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    Splits documents into smaller chunks to prepare for embedding.

    Args:
        documents (list): List of loaded documents.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap size between chunks.

    Returns:
        list: List of split text chunks.
    """
    logger.info(f"✂️ Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"✅ Successfully split documents into {len(chunks)} chunk(s).")
    logger.debug(f"Sample Chunk: {chunks[0] if chunks else 'No Chunks Found'}")
    return chunks


def create_vector_store(text_chunks: list, embeddings, url: str, collection_name: str):
    """
    Creates a Qdrant vector store using the provided text chunks and embeddings.
    Mirrors the database setup in `app.py`.

    Args:
        text_chunks (list): List of text chunks to embed.
        embeddings: Preloaded embedding model for text-to-vector transformation.
        url (str): Qdrant server URL.
        collection_name (str): Name of the Qdrant collection.

    Returns:
        Qdrant: Qdrant vector store instance.
    """
    logger.info(f"🚀 Connecting to Qdrant at '{url}' and creating collection '{collection_name}'...")
    qdrant = Qdrant.from_documents(
        text_chunks, embeddings, url=url, prefer_grpc=False, collection_name=collection_name
    )
    logger.info("✅ Qdrant vector store successfully created.")
    return qdrant

# ----------------------------- Main Execution Block -----------------------------
if __name__ == "__main__":
    try:
        logger.info("🔧 Initializing SentenceTransformer Embedding Model...")
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("✅ Embedding model loaded successfully.")

        # Step 1: Load Documents
        documents = load_documents(DATA_DIR, FILE_PATTERN)

        # Step 2: Split Documents into Chunks
        text_chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

        # Step 3: Create Qdrant Vector Store
        create_vector_store(text_chunks, embeddings, QDRANT_URL, COLLECTION_NAME)

        logger.info("🎉 Document ingestion and vector store creation completed successfully!")

    except Exception as e:
        logger.error(f"❌ An error occurred during the process: {e}")
