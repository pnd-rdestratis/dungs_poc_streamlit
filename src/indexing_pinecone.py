from openai import OpenAI
import os
from pinecone import Pinecone

# Initialize clients
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = 'dungs-poc'

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric='cosine'
    )

index = pc.Index(index_name)

data_chunks = [
    {
        "element_id": "5a3653e9fd0e6c63020a4ae472f31e81",
        "text": "Dast ist ein TESTTTT",
        "metadata": {
            "type": "Footer",
            "filetype": "application/pdf",
            "languages": ["eng"],
            "page_number": 15,
            "child_ids": ["123","233"],
            "filename": "Diff. pressure switch - Product handbook.pdf"
        }
    }
]

for chunk in data_chunks:
    response = client.embeddings.create(
        input=chunk['text'],
        model='text-embedding-3-large'
    )
    embedding = response.data[0].embedding
    metadata = chunk['metadata'].copy()
    metadata['text'] = chunk['text']  # Include original text

    index.upsert([
        {
            'id': chunk['element_id'],
            'values': embedding,
            'metadata': metadata
        }
    ])

print("Data has been successfully upserted into Pinecone.")
