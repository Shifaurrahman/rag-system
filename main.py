from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize embeddings and load the FAISS index
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY3"))
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Define the LLMChain for question answering
qa_prompt = """
You are an AI assistant providing detailed answers based on the given context.
Answer the question based only on the context provided:
{context}
Question: {question}
If the information is not available in the context, respond with:
"Sorry, I don't have much information about it."
Answer:
"""

qa_chain = LLMChain(
    llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY3"), max_tokens=1024),
    prompt=PromptTemplate.from_template(qa_prompt)
)

# Create FastAPI app
app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request schema for the question
class Query(BaseModel):
    question: str

@app.post("/query/")
async def answer_question(query: Query):
    try:
        # Perform a similarity search in the FAISS index
        relevant_docs = db.similarity_search(query.question)
        
        # Check if relevant documents were retrieved
        if not relevant_docs:
            raise HTTPException(status_code=404, detail="No relevant information found.")

        # Build context from relevant documents
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate an answer using the RAG model
        result = qa_chain.run({'context': context, 'question': query.question})
        
        # Return the result
        return {"answer": result}

    except Exception as e:
        # Return a 500 error if any unexpected error occurs
        raise HTTPException(status_code=500, detail=str(e))
