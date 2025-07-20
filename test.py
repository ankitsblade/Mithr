import os
import uuid
import re # Import the regular expression module
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_neo4j import Neo4jGraph

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from supabase.client import Client, create_client
from neo4j import GraphDatabase

import json

# --- 1. Load Environment Variables and Initial Setup ---
load_dotenv()

# Generate a unique session ID for the conversation
SESSION_ID = str(uuid.uuid4())

# --- 2. Pydantic Models for Structured Data ---
# These models ensure the LLM provides data in a predictable, structured format.

class Node(BaseModel):
    """Represents a node in the knowledge graph."""
    id: str = Field(description="A unique identifier for the node, often the name of the entity.")
    type: str = Field(description="The type or category of the node (e.g., 'Professor', 'Department', 'Course').")
    properties: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of attributes for the node.")

class Relationship(BaseModel):
    """Represents a relationship between two nodes in the knowledge graph."""
    source: Node = Field(description="The source node of the relationship.")
    target: Node = Field(description="The target node of the relationship.")
    type: str = Field(description="The type of the relationship (e.g., 'TEACHES', 'IS_PART_OF').")
    properties: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of attributes for the relationship.")

class KnowledgeGraph(BaseModel):
    """Represents the entire knowledge graph extracted from a text chunk."""
    nodes: List[Node] = Field(description="A list of nodes in the graph.")
    relationships: List[Relationship] = Field(description="A list of relationships connecting the nodes.")

# --- 3. Global Clients and Configuration ---

# Azure OpenAI LLM for Chat and Generation
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)

# Azure OpenAI Embeddings Model
embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

# Neo4j Graph Database Connection
neo4j_graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Supabase Client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Supabase Vector Store
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings_model,
    table_name="documents",
    query_name="match_documents", # Supabase RPC function for similarity search
)

# --- 4. Data Ingestion and Graph Creation ---

def create_graph_from_text(text_chunk: str):
    """
    Uses an LLM to extract entities and relationships from a text chunk
    and populates the Neo4j graph.
    """
    print(f"\n--- Processing chunk for graph extraction ---\n{text_chunk[:200]}...")

    structured_llm = llm.with_structured_output(KnowledgeGraph, method="function_calling")

    prompt = f"""
    You are an expert data architect. Your task is to extract entities and their relationships from the provided text, which is from the Mahindra University website.
    Represent the extracted information as a knowledge graph with nodes and relationships.
    Identify key entities such as people (professors, deans), departments, schools (e.g., School of Engineering), courses, admission criteria, facilities, etc.
    Extract relationships like 'TEACHES', 'HEADS', 'IS_PART_OF', 'OFFERS_COURSE', 'REQUIRES'.
    Ensure the output is a valid JSON that conforms to the provided schema.

    Text to process:
    ---
    {text_chunk}
    ---
    """
    try:
        graph_document = structured_llm.invoke(prompt)
        neo4j_graph.add_graph_documents([graph_document], include_source=False)
        print("Successfully added extracted data to Neo4j.")
    except Exception as e:
        print(f"Error extracting or adding graph data: {e}")


def ingest_data():
    """
    Main function to handle data ingestion from the text file into both
    Supabase (for vector search) and Neo4j (for graph search).
    """
    print("Starting data ingestion process...")
    try:
        with open("/home/ankitsblade/Development/mithr_test/another_attempt/corpus.txt", "r") as f:
            text_content = f.read()
    except FileNotFoundError:
        print("ERROR: mahindra_university_corpus.txt not found. Please create this file.")
        return

    chunks = [text_content[i:i + 2000] for i in range(0, len(text_content), 2000)]
    print(f"Split content into {len(chunks)} chunks.")

    print("\n--- Ingesting data into Supabase Vector Store ---")
    vector_store.add_texts(chunks)
    print("Data ingestion into Supabase complete.")

    print("\n--- Ingesting data into Neo4j Knowledge Graph ---")
    for chunk in chunks:
        create_graph_from_text(chunk)
    print("Data ingestion into Neo4j complete.")


# --- 5. Agent State and Tool Definition ---

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
    
    cypher_generation_prompt = f"""
    You are a Neo4j expert. Given the following schema and a user question, generate a Cypher query to answer the question.
    Do not provide any explanation or markdown formatting, just the raw Cypher query.
    If you cannot generate a query based on the schema, return an empty string.

    Schema:
    {neo4j_graph.schema}

    Question: {query}
    """
    
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

# --- 6. Agent Logic and Graph Definition ---

def route_query(state: GraphState) -> dict:
    """
    Analyzes the user query, decides which tool to use, and saves the decision to the state.
    """
    print("--- ROUTER: Deciding next step ---")
    query = state.user_query

    routing_prompt = f"""
    You are an expert routing agent. Given a user query, determine the best tool to use to answer it.
    Your options are:
    1. 'vector_search': For open-ended, semantic questions about policies, descriptions, or general information (e.g., "What is the campus life like?").
    2. 'graph_search': For specific, factual questions about connections and relationships (e.g., "Who is the head of the Computer Science department?", "Which courses does Dr. Anand teach?").
    3. 'direct_answer': For conversational greetings, farewells, or simple statements (e.g., "Hello", "Thanks").

    Return only one of the following strings: 'vector_search', 'graph_search', or 'direct_answer'.

    User Query: "{query}"
    """
    
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

    generation_prompt = f"""
    You are a helpful assistant for Mahindra University.
    Answer the user's query based on the provided context.
    If the context is empty or doesn't contain the answer, state that you couldn't find the information.
    Be concise and clear.

    Context:
    ---
    {context}
    ---
    Query: {query}
    """
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

# --- 7. Assembling the Agentic Workflow using LangGraph ---

workflow = StateGraph(GraphState)

workflow.add_node("vector_search", vector_search_tool)
workflow.add_node("graph_search", graph_search_tool)
workflow.add_node("direct_answer", direct_answer_tool)
workflow.add_node("generate_response", generate_response)
workflow.add_node("log_conversation", log_conversation)
workflow.add_node("route_query", route_query)
# The check_graph_result function is only used for conditional routing, not as a standalone node.

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

# FIX: Create a conditional edge directly from 'graph_search' to handle the fallback.
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


# --- 8. Main Execution Block ---

if __name__ == "__main__":
    run_ingestion = input("Do you want to run the data ingestion process? (This can take a while and should only be done once) [y/N]: ")
    if run_ingestion.lower() == 'y':
        ingest_data()

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
