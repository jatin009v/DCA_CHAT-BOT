import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Updated imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

import google.generativeai as genai

# Load environment variables
load_dotenv()

app = FastAPI(title="BCA & MCA Admission Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
embeddings = None
vectorstore = None
llm = None


def init_langchain():
    global embeddings, vectorstore, llm

    try:
        # ✅ Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ✅ Vector DB check
        if os.path.exists("./chroma_db"):
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            print("✅ Chroma DB Loaded")
        else:
            print("❌ chroma_db not found. Please run ingest.py before deployment.")

        # ✅ Gemini setup
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            llm = genai.GenerativeModel("gemini-1.5-flash")  # stable model
            print("✅ Gemini Initialized")
        else:
            print("❌ GEMINI_API_KEY missing")

    except Exception as e:
        print(f"❌ Init Error: {str(e)}")


# Initialize on startup
init_langchain()


class ChatQuery(BaseModel):
    message: str
    history: list = []


@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectorstore_ready": vectorstore is not None,
        "gemini_ready": llm is not None
    }


@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vector DB not found")

    if not llm:
        raise HTTPException(status_code=500, detail="Gemini not initialized")

    try:
        # 🔍 Step 1: Similarity Search
        docs = vectorstore.similarity_search(query.message, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 🧠 Step 2: Prompt
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a professional BCA & MCA Admission Assistant.

Use the context to answer the question.
If answer not found, say politely you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        final_prompt = prompt_template.format(
            context=context,
            question=query.message
        )

        # 🤖 Step 3: Generate Response
        response = llm.generate_content(
            final_prompt,
            generation_config={
                "temperature": 0.4,
                "max_output_tokens": 500
            }
        )

        if response and response.text:
            return {"response": response.text}
        else:
            return {"response": "No response generated"}

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"response": "Server error, try again later"}


# ✅ 🔥 MOST IMPORTANT PART FOR RENDER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
