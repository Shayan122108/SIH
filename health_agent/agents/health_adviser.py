import os
from typing import TypedDict, List
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables from .env file
load_dotenv()

## 1. Define the State for the Graph
class RAGState(TypedDict):
    question: str
    documents: List[str]
    answer: str

## 2. Setup LLM, Embeddings, and the Retriever
# Initialize the LLM for generating answers
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize the embeddings model to MATCH the one used for ingestion
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# --- Load the local FAISS index ---
try:
    vector_store = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True # Required for loading local FAISS indexes
    )
except RuntimeError as e:
    print("Error loading FAISS index. Did you run `scripts/ingest_data.py` first?")
    print(e)
    exit()

# Create a retriever from the vector store to perform searches
retriever = vector_store.as_retriever()

## 3. Define the Nodes of the Graph
def retrieve_node(state: RAGState) -> RAGState:
    """Retrieves relevant documents from the vector store."""
    print("---NODE: RETRIEVING DOCUMENTS---")
    question = state["question"]
    retrieved_docs = retriever.invoke(question)
    state['documents'] = [doc.page_content for doc in retrieved_docs]
    print(f"Retrieved {len(state['documents'])} documents.")
    return state

def generate_node(state: RAGState) -> RAGState:
    """Generates a final answer using the LLM based on the retrieved documents."""
    print("---NODE: GENERATING ANSWER---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI health adviser from the Government of India, operating in Warangal, Telangana.
        Your task is to answer the user's health-related question accurately and safely.
        
        Use only the following context to answer the question. Do not use any other information.
        If the context does not contain the answer, state that you do not have enough information from the provided documents.
        
        Be concise, clear, and easy to understand. Always encourage the user to consult a medical professional for personal health issues.
        
        Today's Date: {current_date}.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        """
    )
    
    context_str = "\n\n---\n\n".join(documents)
    chain = prompt | llm
    
    result = chain.invoke({
        "context": context_str,
        "question": question,
        "current_date": datetime.now().strftime('%B %d, %Y')
    })
    
    state['answer'] = result.content
    return state

## 4. Assemble the Graph
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

health_adviser_app = workflow.compile()

## 5. Example Usage
if __name__ == "__main__":
    print("--- Health Adviser Agent ---")
    print("Ask a health-related question. Type 'exit' to quit.")
    
    while True:
        user_question = input("\nYOU: ")
        if user_question.lower() == 'exit':
            break
            
        inputs = {"question": user_question}
        final_state = health_adviser_app.invoke(inputs)
        
        print("\nADVISER:")
        print(final_state['answer'])