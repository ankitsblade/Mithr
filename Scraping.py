import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from crawl4ai import AsyncWebCrawler
import os
import re

async def fetch_xml(session, url):
    """Asynchronously fetches and parses an XML file from a URL."""
    print(f"Fetching sitemap: {url}")
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            text = await response.text()
            return ET.fromstring(text)
    except aiohttp.ClientError as e:
        print(f"Error fetching {url}: {e}")
        return None
    except ET.ParseError as e:
        print(f"Error parsing XML from {url}: {e}")
        return None

def sanitize_filename(filename):
    """Removes invalid characters from a filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

async def main():
    """
    Crawls a website by first parsing its sitemap index, then crawling
    the URLs listed in each individual sitemap and saving the content
    to separate Markdown files.
    """
    sitemap_index_url = "https://www.mahindrauniversity.edu.in/sitemap_index.xml"
    output_dir = "sitemap_crawl_results"
    
    # --- Define a set of sitemaps to exclude ---
    sitemaps_to_exclude = {
        "https://www.mahindrauniversity.edu.in/post-sitemap.xml"
    }

    # --- Create output directory if it doesn't exist ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # --- XML Namespace, often found in sitemaps ---
    # This is needed to correctly parse the XML tags
    namespace = {'s': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    async with aiohttp.ClientSession() as session:
        # 1. Fetch and parse the main sitemap index
        index_root = await fetch_xml(session, sitemap_index_url)
        if index_root is None:
            print("Could not fetch or parse the sitemap index. Exiting.")
            return

        # 2. Extract individual sitemap URLs from the index
        sitemap_urls = [
            loc.text for loc in index_root.findall('s:sitemap/s:loc', namespace)
        ]
        
        if not sitemap_urls:
            print("No individual sitemaps found in the index.")
            return

        print(f"Found {len(sitemap_urls)} sitemaps to process.")

        # 3. Process each individual sitemap
        async with AsyncWebCrawler() as crawler:
            for sitemap_url in sitemap_urls:
                # --- Check if the sitemap should be excluded ---
                if sitemap_url in sitemaps_to_exclude:
                    print(f"Skipping excluded sitemap: {sitemap_url}\n")
                    continue

                sitemap_root = await fetch_xml(session, sitemap_url)
                if sitemap_root is None:
                    continue

                # Extract all page URLs from the individual sitemap
                page_urls = [
                    loc.text for loc in sitemap_root.findall('s:url/s:loc', namespace)
                ]

                if not page_urls:
                    print(f"No URLs found in {sitemap_url}. Skipping.")
                    continue
                
                print(f"Found {len(page_urls)} URLs in {sitemap_url}. Starting crawl...")
                
                all_markdown_content = []
                
                # Crawl each page in the current sitemap
                for i, page_url in enumerate(page_urls):
                    print(f"  ({i+1}/{len(page_urls)}) Crawling: {page_url}")
                    try:
                        result = await crawler.arun(url=page_url)
                        if result and result.markdown:
                            # Add a header to separate content from different pages
                            all_markdown_content.append(f"# Content from: {page_url}\n\n{result.markdown}")
                    except Exception as e:
                        print(f"    Failed to crawl {page_url}: {e}")

                # 4. Save the combined content to a file
                # Generate a clean filename from the sitemap URL
                sitemap_filename = sanitize_filename(sitemap_url.split('/')[-1])
                output_filename = os.path.join(output_dir, f"{sitemap_filename}.md")

                if all_markdown_content:
                    with open(output_filename, "w", encoding="utf-8") as f:
                        # Join all content with a clear separator
                        f.write("\n\n---\n\n".join(all_markdown_content))
                    print(f"✅ Successfully saved results to {output_filename}\n")
                else:
                    print(f"⚠️ No content scraped from sitemap: {sitemap_url}\n")


if __name__ == "__main__":
    # To run the async main function
    asyncio.run(main())
