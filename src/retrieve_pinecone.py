import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

def initialize_clients(index_name: str):
    """Initialize clients with specified index name."""
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv('OPENAI_API_KEY'),
        model="text-embedding-3-large"
    )
    return index, embeddings

def search(
    query: str, 
    index, 
    embeddings, 
    selected_file: Optional[str] = None, 
    selected_category: Optional[str] = None,
    top_k: int = 5
) -> List[Dict]:
    """Search the specified index with the query.
    
    Args:
        query: The search query
        index: Pinecone index
        embeddings: OpenAI embeddings
        selected_file: A specific file to search in (optional)
        selected_category: A specific category to search in (optional)
        top_k: Number of results to return
    """
    # Get vector embedding for the query
    vector = embeddings.embed_documents([query])[0]
    
  # Build filter based on parameters
    filter_dict = {}
    
    if selected_file and selected_file != "All Documents":
        filter_dict["filename"] = selected_file
        print(f"Filtering for file: {selected_file}")
    
    if selected_category and selected_category != "All Categories":
        filter_dict["product_category"] = selected_category
        print(f"Filtering for category: {selected_category}")
    
    # If no filters were added, set filter_dict to None
    if not filter_dict:
        filter_dict = None
        
    # For debugging
    print(f"Filter: {filter_dict}")
    
    # Search with vector
    results = index.query(
        vector=vector,
        filter=filter_dict,
        top_k=top_k,
        include_metadata=True
    )
    
    # For debugging - show what was actually found
    print(f"Found {len(results['matches'])} matches")
    if len(results['matches']) > 0:
        print(f"First match filename: {results['matches'][0]['metadata'].get('filename', 'No filename')}")
        print(f"First match category: {results['matches'][0]['metadata'].get('product_category', 'No category')}")

    # Format results
    return [
        {
            'text': match['metadata']['text'],
            'source': match['metadata']['filename'],
            'score': match['score'],
            'page': match['metadata'].get('page_number', 1)
        }
        for match in results['matches']
    ]

if __name__ == "__main__":
    # Example usage
    index_name = "dungs-poc-basic-chunking-all-documents"
    query = "fachkraft zielgruppe gas "
    selected_category = "RepNews"
    filename = "2024-03-21_RepNews-2024-005_Partly-Phase-Out-DMA-Series_1R0_1.pdf"

    index, embeddings = initialize_clients(index_name)
    results = search(query, index, embeddings, filename, selected_category)

    print(f"\nResults for query: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}, Page: {result['page']}")
        print(f"Text: {result['text'][:200]}...")
        print("-" * 80)