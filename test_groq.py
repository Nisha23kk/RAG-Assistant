from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv(dotenv_path="D:/research-assistant-rag/.env")

api_key = os.getenv("GROQ_API_KEY")
print("Loaded API Key:", api_key[:10] + "..." if api_key else None)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=api_key
)

response = llm.invoke("What is Retrieval-Augmented Generation (RAG)?")
print(response.content)