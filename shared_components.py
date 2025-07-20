import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_neo4j import Neo4jGraph

from supabase.client import Client, create_client
# Import the prompts from the prompts.py file
from prompts import GRAPH_EXTRACTION_PROMPT, CYPHER_GENERATION_PROMPT, ROUTING_PROMPT, RESPONSE_GENERATION_PROMPT

# --- 1. Load Environment Variables ---
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
# These clients are defined here so they can be imported by any other script
print("Initializing shared clients...")
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
print("Shared clients initialized.")
