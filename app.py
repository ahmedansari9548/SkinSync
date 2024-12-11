# --------------------- Import Required Modules ---------------------
from langchain import PromptTemplate  # Template for formatting input to the language model.
from langchain_community.llms import LlamaCpp  # For running local LLM models (like BioMistral).
from langchain.chains import RetrievalQA  # Chain for retrieval-based question answering.
from langchain_community.embeddings import SentenceTransformerEmbeddings  # For generating vector embeddings.
from fastapi import FastAPI, Request, Form, Response  # FastAPI for building the API.
from fastapi.responses import HTMLResponse  # HTMLResponse to render pages.
from fastapi.templating import Jinja2Templates  # Jinja2 templating engine for dynamic web pages.
from fastapi.staticfiles import StaticFiles  # To serve static assets like CSS and JS files.
from fastapi.encoders import jsonable_encoder  # JSON encoder for FastAPI responses.
from qdrant_client import QdrantClient  # Client for interacting with Qdrant vector database.
from langchain_community.vectorstores import Qdrant  # LangChain wrapper for Qdrant vector database.
import os  # OS module to manage file paths.
import json  # JSON module for working with structured data.

# --------------------- Initialize FastAPI App ---------------------
app = FastAPI()

# Configure Jinja2 templates and static files.
templates = Jinja2Templates(directory="templates")  # Templates directory for HTML files.
app.mount("/static", StaticFiles(directory="static"), name="static")  # Serve static assets like CSS/JS.

# --------------------- Load Local Language Model ---------------------
# Path to the local LLM model file (BioMistral in gguf format).
local_llm = "BioMistral-7B.Q4_K_M.gguf"

# Initialize LlamaCpp to load the local LLM.
llm = LlamaCpp(
    model_path=local_llm,  # Path to the LLM file.
    temperature=0.3,  # Controls output randomness; lower values = more deterministic.
    max_tokens=2048,  # Limits the maximum tokens in the output.
    top_p=1  # Nucleus sampling parameter (0.0-1.0); higher values include more tokens.
)
print("LLM Initialized....")  # Confirm the model has been loaded successfully.

# --------------------- Define Prompt Template ---------------------
# A custom template for formatting input to the language model.
# Ensures the model uses retrieved context to answer user questions.
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

# --------------------- Setup Embedding Model and Qdrant ---------------------
# Load the SentenceTransformer Embedding model for biomedical text (PubMed).
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Connect to the Qdrant vector database server (runs locally).
url = "http://localhost:6333"  # Qdrant is assumed to be running on localhost, port 6333.
client = QdrantClient(url=url, prefer_grpc=False)  # Initialize the Qdrant client over HTTP.

# Load the Qdrant collection containing vectorized documents.
db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="vector_db_2"  # The name of the vector collection created previously.
)

# --------------------- Setup Retriever and QA Chain ---------------------
# The retriever fetches the top-k most relevant chunks from Qdrant based on the query.
retriever = db.as_retriever(search_kwargs={"k": 1})  # Retrieve the top 1 most relevant document.

# Wrap the prompt template into a LangChain-compatible PromptTemplate.
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# --------------------- FastAPI Endpoints ---------------------

# Route: Render the main HTML page (index.html)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Renders the main user interface (index.html) using the Jinja2 template engine.
    - Templates are stored in the 'templates' directory.
    - Static files like CSS/JS are served from the 'static' directory.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# Route: POST endpoint to generate a response using the LLM.
@app.post("/get_response")
async def get_response(query: str = Form(...)):
    """
    Handles POST requests for querying the system.
    - Accepts the user's query via a form input.
    - Uses the retriever to fetch relevant document chunks from Qdrant.
    - Generates a detailed answer using the local LLM (BioMistral) based on the retrieved context.
    - Returns the answer and source document details in JSON format.
    """
    # Initialize a RetrievalQA chain with the specified configuration.
    chain_type_kwargs = {"prompt": prompt}  # Use the defined prompt template.
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,  # Local LLM model (BioMistral).
        chain_type="stuff",  # Combines retrieved documents into a single context.
        retriever=retriever,  # Qdrant-based retriever for context fetching.
        return_source_documents=True,  # Include the source documents in the response.
        chain_type_kwargs=chain_type_kwargs,  # Pass the custom prompt.
        verbose=True  # Enable detailed logging for debugging.
    )

    # Query the chain to generate a response.
    response = qa(query)  # Includes the answer and source document(s).
    print(response)  # Debugging: Print the raw response to the console.

    # Extract and prepare the output data.
    answer = response['result']  # The generated answer.
    source_document = response['source_documents'][0].page_content  # Content of the first relevant document.
    doc = response['source_documents'][0].metadata['source']  # Metadata about the source document.

    # Convert the data to JSON format.
    response_data = jsonable_encoder(
        json.dumps({"answer": answer, "source_document": source_document, "doc": doc})
    )

    # Return the JSON response to the client.
    res = Response(response_data)
    return res

# --------------------- End of Script ---------------------
