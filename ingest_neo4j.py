import os
import time
from dotenv import load_dotenv

# Import the text splitter and Pydantic models
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import only the necessary components from the shared file
from shared_components import neo4j_graph, KnowledgeGraph
# Import the prompt from the prompts.py file
from prompts import GRAPH_EXTRACTION_PROMPT
# Import the AzureChatOpenAI class to create a new, dedicated LLM instance
from langchain_openai import AzureChatOpenAI

def create_graph_from_text(text_chunk: str, llm_instance):
    """
    Uses a dedicated LLM instance to extract entities and relationships from a text chunk
    and populates the Neo4j graph.
    """
    print(f"\n--- Processing chunk for graph extraction ---\n{text_chunk[:200]}...")

    structured_llm = llm_instance.with_structured_output(KnowledgeGraph, method="function_calling")

    # Use the imported prompt, formatted with the text chunk
    prompt = GRAPH_EXTRACTION_PROMPT.format(text_chunk=text_chunk)
    
    try:
        graph_document = structured_llm.invoke(prompt)
        neo4j_graph.add_graph_documents([graph_document], include_source=False)
        print("Successfully added extracted data to Neo4j.")
    except Exception as e:
        print(f"Error extracting or adding graph data: {e}")


def ingest_into_neo4j():
    """
    Main function to handle data ingestion from the text file into the
    Neo4j knowledge graph using a dedicated, powerful LLM.
    """
    load_dotenv()
    
    # --- Initialize a dedicated, powerful LLM for ingestion ---
    print("Initializing powerful LLM for ingestion...")
    try:
        ingestion_llm = AzureChatOpenAI(
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            # Use the new environment variables for the ingestion model
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_BIG_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
        )
        print("Ingestion LLM initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize the ingestion LLM. Please check your .env file. Error: {e}")
        return

    print("Starting Neo4j ingestion process...")
    try:
        with open("final_scraped.txt", "r") as f:
            text_content = f.read()
        print("Successfully read corpus file.")
    except FileNotFoundError:
        print("ERROR: Corpus file 'final_scraped.txt' not found. Please check the path.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text_content)
    print(f"Split content into {len(chunks)} chunks using RecursiveCharacterTextSplitter.")

    print("\n--- Ingesting data into Neo4j Knowledge Graph ---")
    print("Clearing existing graph to ensure a fresh start...")
    neo4j_graph.query("MATCH (n) DETACH DELETE n")
    print("Graph cleared.")
    
    for chunk in chunks:
        # Pass the dedicated LLM instance to the function
        create_graph_from_text(chunk, ingestion_llm)
        # Optional: add a small delay here as well if you hit chat model rate limits
        # time.sleep(1) 
    print("Data ingestion into Neo4j complete.")


if __name__ == "__main__":
    ingest_into_neo4j()
    print("\nNeo4j ingestion process has finished.")
