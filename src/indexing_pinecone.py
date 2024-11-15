from openai import OpenAI
import os
from pinecone import Pinecone
import json
from pathlib import Path
import time
from typing import List, Dict, Any

# Initialize clients
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
except Exception as e:
    raise Exception(f"Failed to initialize clients: {str(e)}")

def check_existing_ids(index, ids: List[str]) -> set:
    """Check which IDs already exist in the index."""
    try:
        response = index.fetch(ids=ids)
        return set(response.vectors.keys())
    except Exception as e:
        print(f"Error checking existing IDs: {str(e)}")
        return set()

def create_batches(data: List[Dict], batch_size: int = 100):
    """Create batches of the specified size from the data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def process_batch(batch: List[Dict], client: OpenAI, existing_ids: set) -> List[Dict]:
    """Process a batch of chunks_settings_1 to get their embeddings, skipping existing IDs."""
    try:
        # Filter out existing IDs
        new_chunks = [chunk for chunk in batch if chunk.get('element_id') not in existing_ids]

        if not new_chunks:
            return []

        # Get all texts from the new chunks_settings_1
        texts = [chunk['text'] for chunk in new_chunks if chunk.get('text')]

        if not texts:
            return []

        # Get embeddings for all texts in batch
        response = client.embeddings.create(
            input=texts,
            model='text-embedding-3-large'
        )

        # Prepare vectors for upsert
        vectors = []
        for i, chunk in enumerate(new_chunks):
            if chunk.get('text'):
                metadata = chunk.get('metadata', {}).copy()
                metadata['text'] = chunk['text']
                metadata['type'] = chunk.get('type')

                vectors.append({
                    'id': chunk['element_id'],
                    'values': response.data[i].embedding,
                    'metadata': metadata
                })

        return vectors
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return []

def main():
    index_name = 'dungs-poc-2'
    chunks_dir = Path("../chunks_settings_2")
    batch_size = 100

    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=3072,
                metric='cosine'
            )
        index = pc.Index(index_name)
    except Exception as e:
        raise Exception(f"Failed to create/access Pinecone index: {str(e)}")

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found at {chunks_dir.absolute()}")

    total_files_processed = 0
    total_chunks_processed = 0
    total_chunks_skipped = 0
    start_time = time.time()

    for json_file in chunks_dir.glob("*.json"):
        try:
            print(f"\nProcessing file: {json_file.name}")

            # Read JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data_chunks = json.load(f)

            print(f"Found {len(data_chunks)} chunks_settings_1 in file")
            chunks_processed = 0
            chunks_skipped = 0

            # Process in batches
            for batch in create_batches(data_chunks, batch_size):
                # Get IDs from current batch
                batch_ids = [chunk['element_id'] for chunk in batch if chunk.get('element_id')]

                # Check which IDs already exist
                existing_ids = check_existing_ids(index, batch_ids)
                chunks_skipped += len(existing_ids)

                print(f"Processing batch of {len(batch)} chunks_settings_1 ({len(existing_ids)} already exist)...")

                # Process only new chunks_settings_1
                vectors = process_batch(batch, client, existing_ids)

                if vectors:
                    # Upsert batch to Pinecone
                    index.upsert(vectors=vectors)
                    chunks_processed += len(vectors)
                    print(f"Upserted {len(vectors)} new vectors")

                # Small delay between batches
                time.sleep(0.1)

            total_chunks_processed += chunks_processed
            total_chunks_skipped += chunks_skipped
            total_files_processed += 1
            print(f"Completed file {json_file.name}: {chunks_processed} chunks_settings_1 processed, {chunks_skipped} chunks_settings_1 skipped")

        except Exception as e:
            print(f"Error processing file {json_file.name}: {str(e)}")
            continue

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"\nIndexing complete!")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total chunks_settings_1 processed: {total_chunks_processed}")
    print(f"Total chunks_settings_1 skipped (already existed): {total_chunks_skipped}")
    print(f"Total processing time: {processing_time:.2f} seconds")
    if total_chunks_processed > 0:
        print(f"Average time per processed chunk: {(processing_time/total_chunks_processed):.2f} seconds")

if __name__ == "__main__":
    main()
