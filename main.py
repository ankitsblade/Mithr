import os
import uuid
from dotenv import load_dotenv
from typing import List, Set
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI

# --- 1. Import Shared Components ---
print("Importing shared components...")
from shared_components import (
    llm,
    vector_store,
    supabase
)
from prompts import (
    DOCUMENT_ROUTING_PROMPT,
    VALIDATION_PROMPT,
    RESPONSE_GENERATION_PROMPT
)
print("Shared components imported successfully.")


# --- 2. Load Environment Variables and Initial Setup ---
load_dotenv()
SESSION_ID = str(uuid.uuid4())
print(f"New session started: {SESSION_ID}")

# --- Initialize a dedicated, more powerful LLM for validation ---
print("Initializing powerful LLM for validation...")
try:
    validation_llm = AzureChatOpenAI(
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        # Assumes you have set these in your .env file for the powerful model
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_BIG_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
    )
    print("Dedicated validation LLM initialized.")
except Exception as e:
    print(f"Warning: Could not initialize the powerful validation LLM. Using the standard agent LLM as a fallback. Error: {e}")
    # If the powerful model isn't configured, fall back to the standard one
    validation_llm = llm


# --- 3. Agent State Definition ---

class DocumentAgentState(BaseModel):
    """Defines the state for the document routing agent."""
    messages: list = []
    cumulative_context: str = ""
    user_query: str = ""
    selected_document: str = ""
    searched_documents: Set[str] = Field(default_factory=set)
    validation_decision: str = ""
    search_iterations: int = 0

# --- 4. Agent Logic and Tools ---

def get_available_documents(searched_docs: Set[str]) -> List[str]:
    """
    Gets the list of available source documents, excluding those already searched.
    """
    markdown_directory = "sitemap_crawl_results"
    if not os.path.isdir(markdown_directory):
        print(f"Warning: Document directory '{markdown_directory}' not found.")
        return []
    all_docs = {f for f in os.listdir(markdown_directory) if f.endswith('.md')}
    return sorted(list(all_docs - searched_docs))

def route_to_document(state: DocumentAgentState) -> dict:
    """
    Analyzes the user query and decides which document to search next.
    """
    print(f"\n--- ITERATION {state.search_iterations + 1} ---")
    print("--- ROUTER: Deciding which document to search ---")
    query = state.user_query
    
    available_docs = get_available_documents(state.searched_documents)
    if not available_docs:
        print("ROUTER: No more documents to search.")
        return {"selected_document": "NONE"}

    doc_list_str = "\n".join([f"- `{doc}`" for doc in available_docs])
    routing_prompt = DOCUMENT_ROUTING_PROMPT.format(query=query, documents=doc_list_str)
    
    # Use the standard, fast LLM for routing
    decision = llm.invoke(routing_prompt).content.strip()
    cleaned_decision = next((doc for doc in available_docs if doc in decision), "NONE")

    print(f"Router decision: '{cleaned_decision}'")
    
    newly_searched = set(state.searched_documents)
    if cleaned_decision != "NONE":
        newly_searched.add(cleaned_decision)
    
    return {"selected_document": cleaned_decision, "searched_documents": newly_searched}

def filtered_vector_search(state: DocumentAgentState) -> dict:
    """
    Performs a vector search filtered to only the document selected by the router.
    """
    print(f"--- TOOL: Performing filtered vector search on '{state.selected_document}' ---")
    query = state.user_query
    selected_doc = state.selected_document

    if selected_doc == "NONE":
        print("Router did not select a document. Skipping search.")
        return {"cumulative_context": state.cumulative_context}

    try:
        docs = vector_store.similarity_search(
            query,
            k=4,
            filter={'source_file': selected_doc}
        )
        new_context = "\n\n".join([doc.page_content for doc in docs])
        
        updated_context = state.cumulative_context + "\n\n---\n\n" + new_context if state.cumulative_context else new_context
        return {"cumulative_context": updated_context}

    except Exception as e:
        print(f"Error during filtered vector search: {e}")
        return {"cumulative_context": state.cumulative_context}

def validate_search_result(state: DocumentAgentState) -> dict:
    """
    Uses a powerful LLM to check if the retrieved context is a good answer and increments iteration count.
    """
    print("--- VALIDATING SEARCH RESULT ---")
    query = state.user_query
    context = state.cumulative_context
    iterations = state.search_iterations + 1

    remaining_docs = get_available_documents(state.searched_documents)
    if not remaining_docs:
        print("No more documents to search. Accepting current answer as final.")
        return {"validation_decision": "final_answer", "search_iterations": iterations}

    MAX_ITERATIONS = 3
    if iterations >= MAX_ITERATIONS:
        print(f"Max search iterations ({MAX_ITERATIONS}) reached. Accepting current answer as final.")
        return {"validation_decision": "final_answer", "search_iterations": iterations}

    validation_prompt = VALIDATION_PROMPT.format(query=query, context=context)
    # Use the dedicated, powerful LLM for validation
    decision = validation_llm.invoke(validation_prompt).content.strip()
    print(f"Validation decision: '{decision}'")

    return {"validation_decision": decision, "search_iterations": iterations}

def decide_next_step(state: DocumentAgentState) -> str:
    """
    Determines the next step based on the validation result.
    """
    if state.validation_decision == "try_another_document":
        return "route_to_document"
    else:
        return "generate_response"

def generate_response(state: DocumentAgentState) -> dict:
    """
    Generates the final answer to the user based on the retrieved context.
    """
    print("--- GENERATING FINAL RESPONSE ---")
    query = state.user_query
    context = state.cumulative_context

    generation_prompt = RESPONSE_GENERATION_PROMPT.format(context=context, query=query)
    response = llm.invoke(generation_prompt).content
    return {"messages": [("ai", response)]}

def log_conversation(state: DocumentAgentState) -> dict:
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

# --- 5. Assembling the Agentic Workflow using LangGraph ---

workflow = StateGraph(DocumentAgentState)

workflow.add_node("route_to_document", route_to_document)
workflow.add_node("filtered_vector_search", filtered_vector_search)
workflow.add_node("validate_search_result", validate_search_result)
workflow.add_node("generate_response", generate_response)
workflow.add_node("log_conversation", log_conversation)

workflow.set_entry_point("route_to_document")
workflow.add_edge("route_to_document", "filtered_vector_search")
workflow.add_edge("filtered_vector_search", "validate_search_result")

workflow.add_conditional_edges(
    "validate_search_result",
    decide_next_step,
    {
        "route_to_document": "route_to_document",
        "generate_response": "generate_response",
    }
)

workflow.add_edge("generate_response", "log_conversation")
workflow.add_edge("log_conversation", END)

agent_app = workflow.compile()


# --- 6. Main Execution Block ---

if __name__ == "__main__":
    print("\n\n--- Mahindra University AI Assistant (Iterative RAG) ---")
    print("Ask me anything about the university. Type 'exit' to end.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        initial_state = {"user_query": user_input, "messages": []}
        final_state = agent_app.invoke(initial_state)
        
        ai_response = final_state['messages'][-1][1]
        print(f"AI: {ai_response}")
