import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

# Load the env file
load_dotenv()

# Get the key from the env file
API_key = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free" 
PDF_FILENAME = "machine_learning_wp-5_copy.pdf"

vector_db = None
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db
    if os.path.exists(PDF_FILENAME):
        print(f"Indexing {PDF_FILENAME}...")
        loader = PyMuPDFLoader(PDF_FILENAME)
        docs = loader.load()
        splitter = RecursiveCharacter_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
        chunks = splitter.split_documents(docs)
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=local_embeddings,
            persist_directory="./chroma_db_static"
        )
        print("Indexing complete.")
    yield
    # Shutdown logic (optional)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

llm = ChatOpenAI(
    openai_api_key=API_key ,
    openai_api_base=BASE_URL,
    model_name=MODEL_NAME,
    temperature=0.1,        
    max_tokens=400,
    default_headers={"HTTP-Referer": "http://localhost:8000", "X-Title": "RAG App"}
)
from fastapi.responses import HTMLResponse

@app.get("/ask")
async def ask_question(query: str):
    if not vector_db:
        raise HTTPException(status_code=500, detail="Database not ready.")
    
    docs = vector_db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"Use the context to answer: {query}\n\nContext: {context}"
    
    try:
        response = llm.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        # Catch the 429 error and explain it
        if "429" in str(e):
            return {"error": "The free AI model is busy."}
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)