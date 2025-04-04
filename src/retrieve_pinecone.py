import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict
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

def search(query: str, index, embeddings, selected_file=None, top_k=5) -> List[Dict]:
    """Search the specified index with the query."""
    # Get vector embedding for the query
    vector = embeddings.embed_documents([query])[0]

    # Add file filter if a specific file is selected
    filter_dict = {"filename": selected_file} if selected_file else None

    # Search with vector
    results = index.query(
        vector=vector,
        filter=filter_dict,
        top_k=top_k,
        include_metadata=True
    )

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

    index, embeddings = initialize_clients(index_name)
    results = search(query, index, embeddings)

    print(f"\nResults for query: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}, Page: {result['page']}")
        print(f"Text: {result['text'][:200]}...")
        print("-" * 80)
