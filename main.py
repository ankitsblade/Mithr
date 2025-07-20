import os
import uuid
import re
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

# --- 1. Import Shared Components ---
print("Importing shared components...")
from ingest_data import (
    llm,
    neo4j_graph,
    vector_store,
    supabase
)
from prompts import (
    CYPHER_GENERATION_PROMPT,
    ROUTING_PROMPT,
    RESPONSE_GENERATION_PROMPT
)
print("Shared components imported successfully.")


# --- 2. Load Environment Variables and Initial Setup ---
load_dotenv()
SESSION_ID = str(uuid.uuid4())
print(f"New session started: {SESSION_ID}")


# --- 3. Agent State and Tool Definition ---

class GraphState(BaseModel):
    """Defines the state of the agent's workflow."""
    messages: list = []
    context: str = ""
    user_query: str = ""
    next_node: str = ""
    fallback_to_vector: bool = False

# --- Agent Tools ---

def vector_search_tool(state: GraphState) -> dict:
    """Performs a vector search on the Supabase store."""
    print("--- TOOL: Vector Search ---")
    query = state.user_query
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context, "fallback_to_vector": False}

def graph_search_tool(state: GraphState) -> dict:
    """
    Converts a natural language query to a Cypher query, parses the output,
    and executes it. If it fails, it sets a flag to fallback to vector search.
    """
    print("--- TOOL: Graph Search ---")
    query = state.user_query
    
    # Use the imported prompt
    cypher_generation_prompt = CYPHER_GENERATION_PROMPT.format(
        schema=neo4j_graph.schema,
        question=query
    )
    
    try:
        raw_cypher_response = llm.invoke(cypher_generation_prompt).content
        
        match = re.search(r"```(?:cypher)?\s*\n(.*?)\n```", raw_cypher_response, re.DOTALL)
        if match:
            cypher_query = match.group(1).strip()
        else:
            cypher_query = raw_cypher_response.strip()

        if cypher_query.startswith('"') and cypher_query.endswith('"'):
            cypher_query = cypher_query[1:-1]

        print(f"Generated Cypher: {cypher_query}")

        if not cypher_query or cypher_query.startswith("//"):
            print("No valid Cypher query generated. Triggering fallback.")
            return {"context": "", "fallback_to_vector": True}
        
        result = neo4j_graph.query(cypher_query)
        context = json.dumps(result, indent=2)
        return {"context": context, "fallback_to_vector": False}
    except Exception as e:
        print(f"Error in graph search: {e}. Triggering fallback.")
        return {"context": "", "fallback_to_vector": True}


def direct_answer_tool(state: GraphState) -> dict:
    """Handles conversational turns where no retrieval is needed."""
    print("--- TOOL: Direct Answer ---")
    return {"context": "No retrieval needed for this query."}

# --- 4. Agent Logic and Graph Definition ---

def route_query(state: GraphState) -> dict:
    """
    Analyzes the user query, decides which tool to use, and saves the decision to the state.
    """
    print("--- ROUTER: Deciding next step ---")
    query = state.user_query

    # Use the imported prompt
    routing_prompt = ROUTING_PROMPT.format(query=query)
    
    decision = llm.invoke(routing_prompt).content
    print(f"Router decision: {decision}")
    
    if "vector_search" in decision:
        return {"next_node": "vector_search"}
    if "graph_search" in decision:
        return {"next_node": "graph_search"}
    return {"next_node": "direct_answer"}

def select_next_node(state: GraphState) -> str:
    """
    Reads the routing decision from the state and returns it for conditional branching.
    """
    return state.next_node

def check_graph_search_result(state: GraphState) -> str:
    """
    Checks if the graph search was successful or if a fallback is needed.
    """
    print("--- CHECKING GRAPH SEARCH RESULT ---")
    if state.fallback_to_vector:
        print("Decision: Fallback to vector search.")
        return "fallback"
    else:
        print("Decision: Continue to generate response.")
        return "continue"

def generate_response(state: GraphState) -> dict:
    """
    Generates the final answer to the user based on the retrieved context.
    """
    print("--- GENERATING FINAL RESPONSE ---")
    query = state.user_query
    context = state.context

    # Use the imported prompt
    generation_prompt = RESPONSE_GENERATION_PROMPT.format(
        context=context,
        query=query
    )
    
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

workflow.add_conditional_edges(
    "graph_search",
    check_graph_search_result,
    {
        "fallback": "vector_search",
        "continue": "generate_response",
    }
)

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

        initial_state = {"user_query": user_input, "messages": [], "context": "", "next_node": "", "fallback_to_vector": False}
        final_state = agent_app.invoke(initial_state)
        
        ai_response = final_state['messages'][-1][1]
        print(f"AI: {ai_response}")
