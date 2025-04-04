import json
import os
from pathlib import Path
import re
from collections import defaultdict

# Import the functions from the existing code
from utils.enrich_chunks import add_product_info, fix_encoding_issues

def create_filter_indexes():
    """
    Create filter indexes for product_category, product_id, and product_name
    using the same logic as in the enrichment process.
    """
    # Find all PDF files in the documents directory
    base_dir = Path(__file__).parent.parent  # Get parent directory (repo root)
    documents_dir = base_dir / "documents"

    # Initialize sets to store unique values
    categories = set()
    product_ids = set()
    product_names = set()

    # Check if documents directory exists
    if not documents_dir.exists():
        print(f"Warning: Documents directory not found at {documents_dir}")
        print("Falling back to using chunk metadata...")
        return create_filter_indexes_from_chunks()

    # Process all PDF files
    pdf_files = list(documents_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")

    for file_path in pdf_files:
        try:
            # Create a dummy element to use with add_product_info
            dummy_element = [{
                'metadata': {}
            }]

            # Use the existing add_product_info function to extract metadata
            enriched_element = add_product_info(dummy_element, file_path)

            # Extract the metadata from the enriched element
            metadata = enriched_element[0]['metadata']

            # Add to sets if values are not empty
            if metadata.get('product_category'):
                categories.add(metadata['product_category'])
            if metadata.get('product_id'):
                product_ids.add(metadata['product_id'])
            if metadata.get('product_name'):
                product_names.add(metadata['product_name'])

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Convert sets to sorted lists
    categories_list = sorted(list(categories))
    product_ids_list = sorted(list(product_ids))
    product_names_list = sorted(list(product_names))

    print(f"Found {len(categories_list)} unique product categories")
    print(f"Found {len(product_ids_list)} unique product IDs")
    print(f"Found {len(product_names_list)} unique product names")

    # Create result dictionary
    result = {
        "product_categories": categories_list,
        "product_ids": product_ids_list,
        "product_names": product_names_list
    }

    # Save to JSON file
    output_path = base_dir / "src" / "filter_index.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Filter index saved to {output_path}")
    return result

def create_filter_indexes_from_chunks():
    """
    Alternative method to create filter indexes by reading existing chunk files
    if the documents directory is not available.
    """
    base_dir = Path(__file__).parent.parent
    chunks_dir = base_dir / "src" / "chunks" / "unstructured" / "basic_chunking" / "enriched_chunks"

    if not chunks_dir.exists():
        print(f"Error: Chunks directory not found at {chunks_dir}")
        return {}, {}, {}

    # Initialize sets to store unique values
    categories = set()
    product_ids = set()
    product_names = set()

    # Process all JSON chunk files
    json_files = list(chunks_dir.glob("*.json"))
    print(f"Found {len(json_files)} chunk files to process")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            for chunk in chunks:
                metadata = chunk.get('metadata', {})

                if metadata.get('product_category'):
                    categories.add(metadata['product_category'])
                if metadata.get('product_id'):
                    product_ids.add(metadata['product_id'])
                if metadata.get('product_name'):
                    product_names.add(metadata['product_name'])

        except Exception as e:
            print(f"Error processing chunk file {file_path}: {str(e)}")

    # Convert sets to sorted lists
    categories_list = sorted(list(categories))
    product_ids_list = sorted(list(product_ids))
    product_names_list = sorted(list(product_names))

    print(f"Found {len(categories_list)} unique product categories from chunks")
    print(f"Found {len(product_ids_list)} unique product IDs from chunks")
    print(f"Found {len(product_names_list)} unique product names from chunks")

    # Create result dictionary
    result = {
        "product_categories": categories_list,
        "product_ids": product_ids_list,
        "product_names": product_names_list
    }

    # Save to JSON file
    output_path = base_dir / "src" / "filter_index.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Filter index saved to {output_path}")
    return result

if __name__ == "__main__":
    create_filter_indexes()
