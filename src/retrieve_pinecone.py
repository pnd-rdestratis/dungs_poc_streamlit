from openai import OpenAI
import os
from pinecone import Pinecone
from collections import defaultdict

# Initialize clients
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Connect to the existing index
index_name = 'dungs-poc-2'
index = pc.Index(index_name)

# Define an example question
question = "MBE Datenblatt: Welchen Druckverlust / Arbeitsbereich hat ein MBC bei XXX m³/h"

# Generate an embedding for the question
response = client.embeddings.create(
    input=question,
    model='text-embedding-3-large'
)
query_embedding = response.data[0].embedding

# Query Pinecone with hybrid search
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    hybrid_config={
        "text_query": question,
        "alpha": 0.3,
        "metric_type": "dotproduct"
    }
)

# Collect all IDs to fetch and organize by pages
pages_to_fetch = defaultdict(list)
id_relationships = {}

# Store initial results
for match in results['matches']:
    match_id = match['id']
    metadata = match['metadata']
    filename = metadata.get('filename')
    page_number = metadata.get('page_number')

    # Use text_as_html if it exists as a key, otherwise use text
    content = metadata['text_as_html'] if 'text_as_html' in metadata else metadata.get('text', 'No text found')

    # Store original match info
    id_relationships[match_id] = {
        'score': match['score'],
        'text_score': match.get('text_score', 'N/A'),
        'text': content,
        'filename': filename,
        'page_number': page_number
    }

# Fetch all chunks for each relevant page
for match in results['matches']:
    filename = match['metadata'].get('filename')
    page_number = match['metadata'].get('page_number')

    if filename and page_number is not None:
        # Query for all chunks on this page
        page_results = index.query(
            vector=query_embedding,
            filter={
                "filename": filename,
                "page_number": page_number
            },
            top_k=100,
            include_metadata=True,
            hybrid_config={
                "text_query": question,
                "alpha": 0.7,
                "metric_type": "dotproduct"
            }
        )

        pages_to_fetch[(filename, page_number)] = page_results['matches']

# Print results organized by matched chunks
print("\nSearch Results:")
print("=" * 100)

for match_id, info in id_relationships.items():
    print(f"\n🎯 MATCHED CHUNK:")
    print(f"Vector Score: {info['score']:.4f}")
    print(f"Text Score: {info['text_score']}")
    print(f"ID: {match_id}")
    print(f"Text: {info['text']}")
    print(f"File: {info['filename']}, Page: {info['page_number']}")

    # Print all chunks from the same page
    if info['filename'] and info['page_number'] is not None:
        page_chunks = pages_to_fetch[(info['filename'], info['page_number'])]
        print(f"\n📄 OTHER CHUNKS ON THE SAME PAGE ({len(page_chunks)} total chunks):")
        for chunk in page_chunks:
            if chunk['id'] != match_id:  # Skip the matched chunk
                content = chunk['metadata']['text_as_html'] if 'text_as_html' in chunk['metadata'] else chunk['metadata'].get('text', 'No text found')
                print(f"\nChunk ID: {chunk['id']}")
                print(f"Vector Score: {chunk.get('score', 'N/A'):.4f}")
                print(f"Text Score: {chunk.get('text_score', 'N/A')}")
                print(f"Text: {content}")

    print("=" * 100)
