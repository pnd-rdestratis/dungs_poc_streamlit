import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from collections import Counter

def initialize_clients():
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index('dungs-poc-hybrid-2')
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return index, embeddings

def get_sparse_vector(text: str):
    # Simple keyword-based sparse vector
    words = text.lower().split()
    word_counts = Counter(words)

    return {
        'indices': list(range(len(word_counts))),
        'values': [float(count) for count in word_counts.values()]
    }

def search(query: str, index, embeddings, selected_file=None, top_k=5, alpha=0.5):
    # Get dense vector (semantic)
    vector = embeddings.embed_documents([query])[0]

    # Add file filter only if a specific file is selected
    filter_dict = {"filename": selected_file} if selected_file else None

    # Search with both vectors
    results = index.query(
        vector=vector,
        filter=filter_dict,  # Will be None for "All Documents"
        top_k=top_k,
        include_metadata=True,
        alpha=alpha
    )

    return [
        {
            'text': match['metadata']['text'],
            'source': match['metadata']['filename'],
            'score': match['score'],
            'page': match['metadata'].get('page_number', 1)
        }
        for match in results['matches']
    ]
