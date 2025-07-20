import time
# Import the new text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import the necessary pre-initialized client from our shared components file
from shared_components import vector_store

def ingest_into_supabase():
    """
    Main function to handle data ingestion from the text file into the
    Supabase vector store.
    """
    print("Starting Supabase ingestion process...")
    try:
        # Note: The user provided a full path in the last request.
        # Using a relative path is often more portable.
        with open("final_scraped.txt", "r") as f:
            text_content = f.read()
        print("Successfully read corpus file.")
    except FileNotFoundError:
        print("ERROR: Corpus file not found. Please check the path.")
        return

    # FIX: Use RecursiveCharacterTextSplitter for more accurate, context-aware chunking.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text_content)
    print(f"Split content into {len(chunks)} chunks using RecursiveCharacterTextSplitter.")


    # Implement batching to handle API rate limits
    BATCH_SIZE = 16  # The number of chunks to process in each batch
    DELAY_BETWEEN_BATCHES = 2 # Seconds to wait between batches

    print("\n--- Ingesting data into Supabase Vector Store in batches ---")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{len(chunks)//BATCH_SIZE + 1}...")
        try:
            vector_store.add_texts(batch)
            print(f"Batch {i//BATCH_SIZE + 1} ingested. Waiting for {DELAY_BETWEEN_BATCHES} seconds...")
            time.sleep(DELAY_BETWEEN_BATCHES)
        except Exception as e:
            print(f"An error occurred during batch {i//BATCH_SIZE + 1}: {e}")
            # Optional: decide if you want to stop or continue on error
            continue

    print("Data ingestion into Supabase complete.")


if __name__ == "__main__":
    ingest_into_supabase()
    print("\nSupabase ingestion process has finished.")
