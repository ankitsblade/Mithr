import os
import json
from dotenv import load_dotenv

# Import the necessary components. We only need the LLM for this task.
from shared_components import llm

# FIX: A more robust and detailed prompt to guide the LLM in creating a high-quality summary.
SUMMARY_PROMPT_TEMPLATE = """
You are an expert summarizer. Your task is to create a high-quality, concise summary of the provided text from the Mahindra University website.
The summary's primary purpose is to help a routing agent decide if a user's query is relevant to this document.

**Instructions:**
1.  Read the entire document content to understand its main purpose.
2.  Identify the key topics, entities, and concepts discussed (e.g., "admissions process," "B.Tech in AI," "Dr. Yajulu Medury," "campus facilities").
3.  Generate a single, dense sentence that captures these key topics.
4.  The summary should be descriptive and specific.

**Bad Summary Example:** "This document contains information about the university." (Too generic)
**Good Summary Example:** "Provides a detailed overview of the B.Tech in Artificial Intelligence program, including curriculum, faculty, and admission criteria." (Specific and informative)

Document Content:
---
{document_content}
---

One-sentence summary:
"""

def generate_summaries():
    """
    Reads all markdown files from a directory, generates a summary for each,
    and saves the results to a JSON file.
    """
    markdown_directory = "sitemap_crawl_results"
    output_file = "document_index.json"
    
    if not os.path.isdir(markdown_directory):
        print(f"ERROR: Directory not found: '{markdown_directory}'")
        return

    files_to_process = [f for f in os.listdir(markdown_directory) if f.endswith('.md')]
    print(f"Found {len(files_to_process)} markdown files to process.")

    document_index = []

    for filename in files_to_process:
        filepath = os.path.join(markdown_directory, filename)
        print(f" - Processing '{filename}'...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # FIX: Read the entire file content to give the LLM full context.
                content = f.read()
            
            prompt = SUMMARY_PROMPT_TEMPLATE.format(document_content=content)
            summary = llm.invoke(prompt).content.strip()
            
            document_index.append({
                "filename": filename,
                "summary": summary
            })
            print(f"   Summary: {summary}")

        except Exception as e:
            print(f"   - ERROR processing file {filename}: {e}")

    # Save the index to a JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(document_index, f, indent=2)
        print(f"\nSuccessfully created document index at '{output_file}'")
    except Exception as e:
        print(f"\nERROR saving index file: {e}")


if __name__ == "__main__":
    generate_summaries()

