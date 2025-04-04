import os
import json

def extract_unique_product_values(chunks_directory):
    """
    Extract unique values for product categories, IDs, and names from enriched chunks.
    
    Args:
        chunks_directory: Path to the directory containing the enriched chunks
        
    Returns:
        tuple: Lists of unique product categories, IDs, and names
    """
    # Initialize sets to store unique values
    unique_categories = set()
    unique_ids = set()
    unique_names = set()
    
    # Process each JSON file in the directory
    for filename in os.listdir(chunks_directory):
        if filename.endswith('.json'):
            file_path = os.path.join(chunks_directory, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    chunk_data = json.load(f)
                    
                    # Process each item in the list
                    for item in chunk_data:
                        # Check if metadata exists and extract values from there
                        if 'metadata' in item:
                            metadata = item['metadata']
                            
                            if 'product_category' in metadata:
                                unique_categories.add(metadata['product_category'])
                            
                            if 'product_id' in metadata:
                                unique_ids.add(metadata['product_id'])
                            
                            if 'product_name' in metadata:
                                unique_names.add(metadata['product_name'])
                        
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON in {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Convert sets to sorted lists
    unique_categories_list = sorted(list(unique_categories))
    unique_ids_list = sorted(list(unique_ids))
    unique_names_list = sorted(list(unique_names))
    
    # Print basic statistics
    print(f"Found {len(unique_categories_list)} unique product categories")
    print(f"Found {len(unique_ids_list)} unique product IDs")
    print(f"Found {len(unique_names_list)} unique product names")
    
    return unique_categories_list, unique_ids_list, unique_names_list

def save_to_files(categories, ids, names, output_dir):
    """
    Save the unique values to separate text files.
    
    Args:
        categories: List of unique product categories
        ids: List of unique product IDs
        names: List of unique product names
        output_dir: Directory to save the output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save categories
    with open(os.path.join(output_dir, 'unique_categories.txt'), 'w', encoding='utf-8') as f:
        for category in categories:
            f.write(f"{category}\n")
            
    # Save IDs
    with open(os.path.join(output_dir, 'unique_product_ids.txt'), 'w', encoding='utf-8') as f:
        for prod_id in ids:
            f.write(f"{prod_id}\n")
            
    # Save names
    with open(os.path.join(output_dir, 'unique_product_names.txt'), 'w', encoding='utf-8') as f:
        for name in names:
            f.write(f"{name}\n")
    
    print(f"Unique values saved to files in {output_dir}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    CHUNKS_DIR = os.path.join(BASE_DIR, "chunks", "unstructured", "basic_chunking", "enriched_chunks")
    OUTPUT_DIR = os.path.join(BASE_DIR, "search_index_data")

    # Extract unique values
    categories, ids, names = extract_unique_product_values(CHUNKS_DIR)
    
    # Save to files
    save_to_files(categories, ids, names, OUTPUT_DIR)
    
    # Print the first few entries of each list
    if categories:
        print("\nSample categories:", categories[:5])
    if ids:
        print("Sample product IDs:", ids[:5])
    if names:
        print("Sample product names:", names[:5])