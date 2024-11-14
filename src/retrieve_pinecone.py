from openai import OpenAI
import os
from pinecone import Pinecone

# Initialize clients
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Connect to the existing index
index_name = 'dungs-poc'
index = pc.Index(index_name)

# Define an example question
question = "What Test?"

# Generate an embedding for the question
response = client.embeddings.create(
    input=question,
    model='text-embedding-3-large'
)
query_embedding = response.data[0].embedding

# Query Pinecone
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

# Print results including the original text
print("\nSearch Results:")
print("-" * 50)
for match in results['matches']:
    print(f"Score: {match['score']:.4f}")
    print(f"ID: {match['id']}")
    print("Text:", match['metadata'].get('text', 'No text found'))
    print("Metadata:")
    for key, value in match['metadata'].items():
        if key != 'text':  # Skip printing text here again
            print(f"  {key}: {value}")
    print("-" * 50)
