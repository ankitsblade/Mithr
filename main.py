import os
import uuid
import re
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI

# --- 1. Import Shared Components ---
print("Importing shared components...")
from shared_components import (
    llm, # Standard model
    neo4j_graph,
    vector_store,
    supabase
)
from prompts import (
    CYPHER_GENERATION_PROMPT,
    ROUTING_PROMPT,
    VALIDATE_GRAPH_RESPONSE_PROMPT,
    RESPONSE_GENERATION_PROMPT
)
print("Shared components imported successfully.")


# --- 2. Load Environment Variables and Initial Setup ---
load_dotenv()
SESSION_ID = str(uuid.uuid4())
print(f"New session started: {SESSION_ID}")

# --- Initialize a dedicated, more powerful LLM for Cypher generation ---
print("Initializing powerful LLM for Cypher generation...")
try:
    cypher_llm = AzureChatOpenAI(
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_BIG_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
    )
    print("Dedicated Cypher generation LLM initialized.")
except Exception as e:
    print(f"Warning: Could not initialize the powerful Cypher LLM. Using the standard agent LLM as a fallback. Error: {e}")
    cypher_llm = llm

# --- 3. Agent State and Tool Definition ---

class GraphState(BaseModel):
    """Defines the state of the agent's workflow."""
    messages: list = []
    context: str = ""
    user_query: str = ""
    next_node: str = ""
    # FIX: Add a counter for Cypher regeneration attempts
    cypher_attempts: int = 0

# --- Agent Tools ---

def vector_search_tool(state: GraphState) -> dict:
    """Performs a vector search on the Supabase store."""
    print("--- TOOL: Vector Search ---")
    query = state.user_query
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def graph_search_tool(state: GraphState) -> dict:
    """
    Performs a preliminary vector search to gather context, then uses that context
    to generate a more accurate Cypher query.
    """
    print("--- TOOL: Graph Search ---")
    query = state.user_query
    
    # FIX: Add a preliminary vector search to provide the Cypher LLM with more context.
    print("--- Performing preliminary vector search for Cypher context ---")
    try:
        docs = vector_store.similarity_search(query, k=2)
        vector_context = "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Preliminary vector search failed: {e}")
        vector_context = "" # Continue without vector context if it fails

    regeneration_hint = ""
    if state.cypher_attempts > 0:
        regeneration_hint = "You have tried to answer this question before and failed. Please analyze the question again and formulate a new, different query strategy."

    # Dynamically create an enhanced prompt with the added vector context
    enhanced_cypher_prompt = f"""{CYPHER_GENERATION_PROMPT}

Here is some additional context from a vector search that might help you identify the entities, properties, and relationships involved:
---
{vector_context}
---
"""

    cypher_generation_prompt = enhanced_cypher_prompt.format(
        schema=neo4j_graph.schema,
        question=query,
        regeneration_hint=regeneration_hint
    )
    
    try:
        raw_cypher_response = cypher_llm.invoke(cypher_generation_prompt).content
        
        match = re.search(r"```(?:cypher)?\s*\n(.*?)\n```", raw_cypher_response, re.DOTALL)
        if match:
            cypher_query = match.group(1).strip()
        else:
            cypher_query = raw_cypher_response.strip()

        if cypher_query.startswith('"') and cypher_query.endswith('"'):
            cypher_query = cypher_query[1:-1]

        print(f"Generated Cypher: {cypher_query}")

        if not cypher_query or cypher_query.startswith("//"):
            print("No valid Cypher query generated.")
            return {"context": "[]"} 
        
        result = neo4j_graph.query(cypher_query)
        context = json.dumps(result, indent=2)
        return {"context": context}
    except Exception as e:
        print(f"Error executing Cypher query: {e}.")
        return {"context": "[]"}


def direct_answer_tool(state: GraphState) -> dict:
    """Handles conversational turns where no retrieval is needed."""
    print("--- TOOL: Direct Answer ---")
    return {"context": "No retrieval needed for this query."}

# --- 4. Agent Logic and Graph Definition ---

def route_query(state: GraphState) -> dict:
    """
    Analyzes the user query and decides which tool to use.
    """
    print("--- ROUTER: Deciding next step ---")
    query = state.user_query
    routing_prompt = ROUTING_PROMPT.format(query=query)
    decision = llm.invoke(routing_prompt).content
    print(f"Router decision: {decision}")
    
    if "graph_search" in decision:
        return {"next_node": "graph_search", "cypher_attempts": 0}
    return {"next_node": decision}

def select_next_node(state: GraphState) -> str:
    """
    Reads the routing decision from the state for conditional branching.
    """
    return state.next_node

def validate_graph_response(state: GraphState) -> dict:
    """
    Uses an LLM to check if the data retrieved from the graph is a good answer.
    """
    print("--- VALIDATING GRAPH RESPONSE ---")
    
    if not state.context or state.context == "[]":
        print("Context is empty. Deciding to regenerate or fallback.")
        if state.cypher_attempts < 1:
             return {"next_node": "regenerate_cypher"}
        else:
             return {"next_node": "fallback_to_vector"}

    validation_prompt = VALIDATE_GRAPH_RESPONSE_PROMPT.format(
        question=state.user_query,
        context=state.context
    )
    decision = llm.invoke(validation_prompt).content
    print(f"Validation decision: {decision}")
    
    if "regenerate_cypher" in decision:
        if state.cypher_attempts < 1:
             return {"next_node": "regenerate_cypher"}
        else:
             print("Max Cypher retries reached. Forcing fallback.")
             return {"next_node": "fallback_to_vector"}

    return {"next_node": decision}

def select_validation_decision(state: GraphState) -> str:
    """
    Reads the validation decision from the state for conditional branching.
    """
    return state.next_node

def increment_cypher_attempts(state: GraphState) -> dict:
    """
    Increments the counter for Cypher generation attempts.
    """
    print("--- INCREMENTING CYPHER ATTEMPTS ---")
    attempts = state.cypher_attempts + 1
    return {"cypher_attempts": attempts}

def generate_response(state: GraphState) -> dict:
    """
    Generates the final answer to the user based on the retrieved context.
    """
    print("--- GENERATING FINAL RESPONSE ---")
    query = state.user_query
    context = state.context
    generation_prompt = RESPONSE_GENERATION_PROMPT.format(context=context, query=query)
    response = llm.invoke(generation_prompt).content
    return {"messages": [("ai", response)]}

def log_conversation(state: GraphState) -> dict:
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

workflow = StateGraph(GraphState)

workflow.add_node("vector_search", vector_search_tool)
workflow.add_node("graph_search", graph_search_tool)
workflow.add_node("direct_answer", direct_answer_tool)
workflow.add_node("generate_response", generate_response)
workflow.add_node("log_conversation", log_conversation)
workflow.add_node("route_query", route_query)
workflow.add_node("validate_graph_response", validate_graph_response)
workflow.add_node("increment_cypher_attempts", increment_cypher_attempts)

workflow.set_entry_point("route_query")

workflow.add_conditional_edges(
    "route_query",
    select_next_node,
    {
        "vector_search": "vector_search",
        "graph_search": "graph_search",
        "direct_answer": "generate_response",
    }
)

workflow.add_edge("graph_search", "validate_graph_response")

workflow.add_conditional_edges(
    "validate_graph_response",
    select_validation_decision,
    {
        "good_answer": "generate_response",
        "regenerate_cypher": "increment_cypher_attempts",
        "fallback_to_vector": "vector_search",
    }
)

workflow.add_edge("increment_cypher_attempts", "graph_search")

workflow.add_edge("vector_search", "generate_response")
workflow.add_edge("generate_response", "log_conversation")
workflow.add_edge("log_conversation", END)

agent_app = workflow.compile()


# --- 6. Main Execution Block ---

if __name__ == "__main__":
    print("\n\n--- Mahindra University AI Assistant ---")
    print("Ask me anything about the university. Type 'exit' to end.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        initial_state = {"user_query": user_input, "messages": []}
        final_state = agent_app.invoke(initial_state)
        
        ai_response = final_state['messages'][-1][1]
        print(f"AI: {ai_response}")
