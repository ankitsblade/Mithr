import os
import uuid
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel

from langgraph.graph import StateGraph, END

# --- 1. Import Shared Components ---
print("Importing shared components...")
# We only need the standard LLM, vector_store, supabase client, and the response prompt
from shared_components import (
    llm,
    vector_store,
    supabase
)
from prompts import RESPONSE_GENERATION_PROMPT
print("Shared components imported successfully.")


# --- 2. Load Environment Variables and Initial Setup ---
load_dotenv()
SESSION_ID = str(uuid.uuid4())
print(f"New session started: {SESSION_ID}")


# --- 3. Agent State Definition ---

class SimpleRAGState(BaseModel):
    """Defines the state for the simple RAG workflow."""
    messages: list = []
    context: str = ""
    user_query: str = ""

# --- 4. Core RAG Logic ---

def retrieve_context(state: SimpleRAGState) -> dict:
    """
    Performs a global vector search across all documents to find the most relevant context.
    """
    print("--- RETRIEVING CONTEXT (Global Search) ---")
    query = state.user_query
    
    # Perform a similarity search across the entire vector store
    # k=5 means we retrieve the top 5 most relevant chunks overall
    docs = vector_store.similarity_search(query, k=5)
    
    # Aggregate the content from the retrieved documents
    context = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source_file', 'N/A')}\n\n{doc.page_content}" for doc in docs])
    
    return {"context": context}

def generate_response(state: SimpleRAGState) -> dict:
    """
    Generates the final answer to the user based on the retrieved context.
    """
    print("--- GENERATING FINAL RESPONSE ---")
    query = state.user_query
    context = state.context

    generation_prompt = RESPONSE_GENERATION_PROMPT.format(context=context, query=query)
    response = llm.invoke(generation_prompt).content
    return {"messages": [("ai", response)]}

def log_conversation(state: SimpleRAGState) -> dict:
    """Logs the user query and the final AI response to Supabase."""
    print("--- LOGGING CONVERSATION ---")
    user_query = state.user_query
    ai_response = state.messages[-1][1] 
    
    try:
        supabase.table("conversation_logs").insert({
            "session_id": SESSION_ID,
            "user_query": user_query,
            "ai_response": ai_response
        }).execute()
        print("Successfully logged conversation to Supabase.")
    except Exception as e:
        print(f"Error logging conversation: {e}")
        
    return {}

# --- 5. Assembling the Workflow using LangGraph ---

workflow = StateGraph(SimpleRAGState)

# Define the nodes in the simple, linear workflow
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("generate_response", generate_response)
workflow.add_node("log_conversation", log_conversation)

# Define the workflow edges
workflow.set_entry_point("retrieve_context")
workflow.add_edge("retrieve_context", "generate_response")
workflow.add_edge("generate_response", "log_conversation")
workflow.add_edge("log_conversation", END)

agent_app = workflow.compile()


# --- 6. Main Execution Block ---

if __name__ == "__main__":
    print("\n\n--- Mahindra University AI Assistant (Simple RAG) ---")
    print("Ask me anything about the university. Type 'exit' to end.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        initial_state = {"user_query": user_input, "messages": []}
        final_state = agent_app.invoke(initial_state)
        
        ai_response = final_state['messages'][-1][1]
        print(f"AI: {ai_response}")
