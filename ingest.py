import os
import re
import time # Import the time module to add delays
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_neo4j import Neo4jGraph

from supabase.client import Client, create_client
# Import the prompt from the new prompts.py file
from prompts import GRAPH_EXTRACTION_PROMPT

# --- 1. Load Environment Variables and Initial Setup ---
print("Loading environment variables...")
load_dotenv()

# --- 2. Pydantic Models for Structured Data ---
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
# These clients are defined here so they can be imported by the main agent script
print("Initializing clients...")
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0,
)

embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

neo4j_graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings_model,
    table_name="documents",
    query_name="match_documents",
)
print("Clients initialized.")

# --- 4. Data Ingestion and Graph Creation Functions ---

def create_graph_from_text(text_chunk: str):
    """
    Uses an LLM to extract entities and relationships from a text chunk
    and populates the Neo4j graph.
    """
    print(f"\n--- Processing chunk for graph extraction ---\n{text_chunk[:200]}...")

    structured_llm = llm.with_structured_output(KnowledgeGraph, method="function_calling")

    # Use the imported prompt, formatted with the text chunk
    prompt = GRAPH_EXTRACTION_PROMPT.format(text_chunk=text_chunk)
    
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
        with open("/home/ankitsblade/Development/mithr_test/final_scraped.txt", "r") as f:
            text_content = f.read()
        print("Successfully read corpus file.")
    except FileNotFoundError:
        print("ERROR: final_scraped.txt not found. Please create this file.")
        return

    chunks = [text_content[i:i + 2000] for i in range(0, len(text_content), 2000)]
    print(f"Split content into {len(chunks)} chunks.")

    # FIX: Implement batching to handle API rate limits
    BATCH_SIZE = 16  # The number of chunks to process in each batch
    DELAY_BETWEEN_BATCHES = 2 # Seconds to wait between batches

    print("\n--- Ingesting data into Supabase Vector Store in batches ---")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}...")
        vector_store.add_texts(batch)
        print(f"Batch {i//BATCH_SIZE + 1} ingested. Waiting for {DELAY_BETWEEN_BATCHES} seconds...")
        time.sleep(DELAY_BETWEEN_BATCHES)

    print("Data ingestion into Supabase complete.")

    print("\n--- Ingesting data into Neo4j Knowledge Graph ---")
    print("Clearing existing graph to ensure a fresh start with the new, robust data...")
    neo4j_graph.query("MATCH (n) DETACH DELETE n")
    print("Graph cleared.")
    
    # The graph creation can also be rate-limited if needed, but it's less likely to be an issue.
    # For now, we'll keep it as is, but a similar batching/delay could be added here.
    for chunk in chunks:
        create_graph_from_text(chunk)
    print("Data ingestion into Neo4j complete.")


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    ingest_data()
    print("\nAll data has been successfully ingested.")
