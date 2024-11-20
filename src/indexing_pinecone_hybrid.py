import os
from pinecone import Pinecone
from transformers import XLMRobertaTokenizer
from collections import Counter
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import json
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from time import sleep

def prepare_chunk_text(chunk: Dict) -> str:
    """Prepare chunk text with filename but weighted less prominently"""
    filename = chunk.get('metadata', {}).get('filename', '').replace('.pdf', '').replace('_', ' ')
    original_text = chunk['text']
    return f"Document {filename}: {original_text}"

def build_dict(input_batch):
    """Transform input IDs into dictionaries with BM25-style weighting"""
    sparse_emb = []

    # Calculate IDF across all documents
    total_docs = len(input_batch)
    doc_freq = Counter()
    for token_ids in input_batch:
        doc_freq.update(set(token_ids))

    # BM25 parameters
    k1 = 1.5
    b = 0.75
    avg_doc_length = sum(len(doc) for doc in input_batch) / len(input_batch)

    for token_ids in input_batch:
        term_freq = Counter(token_ids)
        indices = []
        values = []
        doc_length = len(token_ids)

        for token_id, tf in term_freq.items():
            # Skip special tokens
            if token_id in {0, 1, 2}:
                continue

            # Calculate BM25 score
            idf = max(0.0, np.log((total_docs - doc_freq[token_id] + 0.5) /
                                (doc_freq[token_id] + 0.5)))
            bm25_score = idf * ((tf * (k1 + 1)) /
                              (tf + k1 * (1 - b + b * doc_length / avg_doc_length)))

            indices.append(token_id)
            values.append(float(bm25_score))

        sparse_emb.append({'indices': indices, 'values': values})

    return sparse_emb

def generate_sparse_vectors(texts: List[str], tokenizer) -> List[Dict]:
    """Generate sparse vectors using XLM-RoBERTa tokenizer and BM25 weighting"""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,
        add_special_tokens=False
    )['input_ids']

    return build_dict(inputs)

def load_and_prepare_data() -> List[Dict]:
    chunks_dir = Path("../chunks_settings_2")
    all_chunks = []

    print("\nLoading chunks from files...")
    for json_file in chunks_dir.glob("*.json"):
        print(f"Reading {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)

    print(f"Total chunks loaded: {len(all_chunks)}")
    return all_chunks

def upsert_with_retry(index, vectors: List[Dict], max_retries: int = 3):
    """Upsert vectors with retry logic and dynamic chunk sizing"""
    chunk_size = 50  # Start with small chunks
    start_idx = 0

    while start_idx < len(vectors):
        end_idx = min(start_idx + chunk_size, len(vectors))
        batch = vectors[start_idx:end_idx]

        for attempt in range(max_retries):
            try:
                index.upsert(vectors=batch)
                print(f"Successfully upserted {len(batch)} vectors")
                start_idx = end_idx
                break
            except Exception as e:
                if "Request size" in str(e) and chunk_size > 1:
                    chunk_size = max(1, chunk_size // 2)
                    print(f"Reducing batch size to {chunk_size}")
                    break
                elif attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
                    sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise Exception(f"Failed to upsert after {max_retries} retries: {str(e)}")

def process_batch(batch: List[Dict], embeddings, tokenizer) -> List[Dict]:
    """Process a single batch of chunks"""
    # Prepare texts with filenames
    texts = [prepare_chunk_text(chunk) for chunk in batch]

    # Generate dense vectors
    dense_vectors = embeddings.embed_documents(texts)

    # Generate sparse vectors
    sparse_vectors = generate_sparse_vectors(texts, tokenizer)

    # Prepare vectors for upsert
    vectors = []
    for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
        vectors.append({
            'id': chunk['element_id'],
            'values': dense_vec,
            'sparse_values': sparse_vec,
            'metadata': {
                'filename': chunk.get('metadata', {}).get('filename'),
                'page_number': chunk.get('metadata', {}).get('page_number'),
                'text': chunk['text'],
                'text_with_filename': prepare_chunk_text(chunk)
            }
        })

    return vectors

def main():
    print("\nInitializing components...")

    # Initialize clients
    pc = Pinecone(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment="gcp-starter"
    )

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        show_progress_bar=True
    )

    # Initialize tokenizer for sparse vectors
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    # Load data
    chunks = load_and_prepare_data()
    total_chunks = len(chunks)

    # Setup Pinecone index
    index_name = 'dungs-poc-hybrid-2'
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="dotproduct",
            spec={
                "pod": {
                    "environment": "gcp-starter",
                    "pod_type": "starter"
                }
            }
        )

    index = pc.Index(index_name)

    # Process in smaller batches
    batch_size = 50  # Smaller batch size

    print(f"\nProcessing {total_chunks} chunks in batches of {batch_size}...")

    for i in tqdm(range(0, total_chunks, batch_size), desc="Processing batches"):
        i_end = min(i + batch_size, total_chunks)
        batch = chunks[i:i_end]

        try:
            print(f"\nProcessing batch {i//batch_size + 1}...")
            vectors = process_batch(batch, embeddings, tokenizer)

            print("Upserting vectors...")
            upsert_with_retry(index, vectors)

            print(f"Batch {i//batch_size + 1} completed:")
            print(f"- Processed chunks {i+1} to {i_end} of {total_chunks}")

        except Exception as e:
            print(f"\nError processing batch starting at {i}:")
            print(f"Error details: {str(e)}")
            continue

    # Show final stats
    try:
        stats = index.describe_index_stats()
        print("\nIndexing complete!")
        print(f"Total vectors in index: {stats['total_vector_count']}")
        print("Index statistics:")
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"\nError getting index stats: {str(e)}")

if __name__ == "__main__":
    main()
