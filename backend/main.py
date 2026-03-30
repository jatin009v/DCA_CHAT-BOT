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

# Load env
load_dotenv()

app = FastAPI(title="BCA & MCA Admission Chatbot API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals (lazy load)
embeddings = None
vectorstore = None
llm = None


# ✅ LAZY LOAD VECTOR DB
def get_vectorstore():
    global embeddings, vectorstore

    if vectorstore is None:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            print("✅ Vector DB Loaded")
        except Exception as e:
            print("❌ Vector DB Error:", e)
            vectorstore = None

    return vectorstore


# ✅ LAZY LOAD GEMINI
def get_llm():
    global llm

    if llm is None:
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                llm = genai.GenerativeModel("gemini-2.5-flash")
                print("✅ Gemini Loaded")
            else:
                print("❌ API KEY missing")
        except Exception as e:
            print("❌ Gemini Error:", e)
            llm = None

    return llm


class ChatQuery(BaseModel):
    message: str
    history: list = []


# ✅ ROOT (important for Render)
@app.get("/")
def home():
    return {"message": "API is running 🚀"}


# ✅ HEALTH CHECK
@app.get("/health")
def health():
    return {"status": "ok"}


# ✅ CHAT API
@app.post("/chat")
async def chat_endpoint(query: ChatQuery):

    vectorstore = get_vectorstore()
    llm = get_llm()

    if not vectorstore:
        return {"response": "Vector DB not ready yet"}

    if not llm:
        return {"response": "AI model not ready"}

    try:
        docs = vectorstore.similarity_search(query.message, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

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
        print("Error:", e)
        return {"response": "Server error"}


# ✅ 🔥 RENDER FIX (MOST IMPORTANT)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
