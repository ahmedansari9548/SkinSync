from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import os

# Load environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
COLLECTION_NAME = "vector_db_2"

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_HOST, prefer_grpc=False)

# Load documents from the "documents" folder (expects PDFs)
documents = []
pdf_folder = "/app/documents"
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
        documents.extend(loader.load())

# Split text into chunks for embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert to embeddings
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Store embeddings in Qdrant
db = Qdrant.from_documents(docs, embeddings, client=client, collection_name=COLLECTION_NAME)

print("✅ Documents successfully loaded into Qdrant!")
