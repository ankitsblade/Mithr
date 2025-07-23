import os
import time
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import the necessary pre-initialized client from our shared components file
from shared_components import vector_store

def clean_markdown_content(text: str) -> str:
    """
    Performs a more robust cleaning of the markdown text.
    1. Removes all markdown-style links (both standard and image).
    2. Cleans up leftover empty list items and excessive blank lines.
    """
    # Step 1: Remove all markdown links (e.g., [text](url) and ![alt](url))
    text_no_links = re.sub(r'!*\[[^\]]*\]\([^\)]+\)', '', text)

    # Step 2: Process lines to remove leftover artifacts
    lines = text_no_links.split('\n')
    cleaned_lines = []
    for line in lines:
        # This regex checks if a line consists ONLY of whitespace and/or markdown list markers (*, -, +)
        # If it does, we skip it to remove the empty list item.
        if not re.match(r'^\s*[\*\-\+]\s*$', line):
            cleaned_lines.append(line)

    # Rejoin the text and reduce multiple blank lines down to a single one
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()


def extract_title_from_markdown(content: str) -> str:
    """
    Extracts the first H1 header (# Title) from the markdown content.
    Falls back to the first non-empty line if no H1 is found.
    """
    match = re.search(r'^#\s+(.*)', content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    
    first_line = next((line for line in content.split('\n') if line.strip()), "Untitled Document")
    return first_line.strip()

def ingest_into_supabase():
    """
    Reads all markdown files from a specified directory, cleans them, splits them into chunks,
    extracts metadata, and ingests them into a Supabase table.
    """
    markdown_directory = "sitemap_crawl_results"
    if not os.path.isdir(markdown_directory):
        print(f"ERROR: Directory not found: '{markdown_directory}'")
        print("Please make sure a directory named 'sitemap_crawl_results' exists and contains your .md files.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    files_to_process = [f for f in os.listdir(markdown_directory) if f.endswith('.md')]
    print(f"Found {len(files_to_process)} markdown files to process in '{markdown_directory}'.")

    all_chunks = []
    all_metadatas = []

    # --- Step 1: Read, Clean, and Chunk all files ---
    for filename in files_to_process:
        filepath = os.path.join(markdown_directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use the new, more robust cleaning function
            cleaned_content = clean_markdown_content(content)
            
            # Extract title from the already cleaned content
            title = extract_title_from_markdown(cleaned_content)
            
            chunks = text_splitter.split_text(cleaned_content)
            
            for i, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                all_metadatas.append({
                    'source_file': filename,
                    'title': title,
                    'chunk_index': i
                })
            
            print(f" - Cleaned and split '{filename}' (Title: '{title}') into {len(chunks)} chunks.")

        except Exception as e:
            print(f" - ERROR processing file {filename}: {e}")

    if not all_chunks:
        print("No chunks were generated from the files. Exiting.")
        return
        
    print(f"\nTotal chunks to ingest: {len(all_chunks)}")

    # --- Step 2: Ingest in Batches ---
    BATCH_SIZE = 50
    DELAY_BETWEEN_BATCHES = 2

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_metadatas = all_metadatas[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = -(-len(all_chunks) // BATCH_SIZE)

        print(f"\n--- Processing Batch {batch_num}/{total_batches} ---")

        try:
            vector_store.add_texts(texts=batch_chunks, metadatas=batch_metadatas)
            print(f"Successfully ingested Batch {batch_num} ({len(batch_chunks)} chunks).")

        except Exception as e:
            print(f"ERROR on Batch {batch_num}: {e}")
        
        if i + BATCH_SIZE < len(all_chunks):
            print(f"Waiting for {DELAY_BETWEEN_BATCHES} seconds before next batch...")
            time.sleep(DELAY_BETWEEN_BATCHES)

    print("\n--- Data ingestion process complete. ---")


if __name__ == "__main__":
    ingest_into_supabase()
    print("\nSupabase ingestion process has finished.")
