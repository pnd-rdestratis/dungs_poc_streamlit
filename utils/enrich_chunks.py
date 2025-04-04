import json
import os
from pathlib import Path
import re

def add_prefix_to_text(elements):
    for element in elements:
        filename = element['metadata']['filename'].replace('.pdf', '')
        element['text'] = f"Dieser Text ist im Dokument: {fix_encoding_issues(filename)} zu finden. {element['text']}"
    return elements

def fix_encoding_issues(text):
    """
    Fix common encoding issues in text found in PDF filenames.
    
    Args:
        text (str): The text with encoding issues
        
    Returns:
        str: The text with encoding issues fixed
    """
    # Step 1: Fix standard encoding replacement patterns
    replacements = {
        'â_„': ' ',      # This is likely a space or slash in many cases
        'â€"': '–',     # En dash
        'â€"': '—',     # Em dash
        "â_„": "/",     # Dash
        'â€™': ''',     # Right single quotation mark
        'â€˜': ''',     # Left single quotation mark
        'â€œ': '"',     # Left double quotation mark
        'â€': '"',      # Right double quotation mark
        'Ã¤': 'ä',      # German umlaut a
        'Ã¶': 'ö',      # German umlaut o
        'Ã¼': 'ü',      # German umlaut u
        'ÃŸ': 'ß',      # German sharp s
        'Â°': '°',      # Degree symbol
        'Ãœ': 'Ü',      # German umlaut U
    }
    
    # Apply replacements
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Clean up any double spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove file extension if present
    text = re.sub(r'\.pdf$', '', text, flags=re.IGNORECASE)

    # Remove underscore at the beginning of the name if present
    text = re.sub(r'^_', '', text)
    
    return text.strip()

def add_product_info(elements, file_path):
    """
    -product_category this is the name of the folder that the pdf is in, although some have a number like this in the name: (3). This number should be removed from the product category info.
    -product_id: if the pdf name starts with a value, take that value (until ther is a non-digit token) 
    -product_name: this is the name after the product id value.  
    """
    # Get product category from parent folder name (removing any numbering in parentheses)
    product_category = file_path.parent.name
    # Remove numbering pattern like "(3)" if present
    product_category = re.sub(r'\s*\(\d+\)', '', product_category).strip()
    
    # Get the filename without extension
    filename = file_path.stem
    
    # Extract product ID (digits at the beginning)
    if product_category == "RepNews":
        product_id_match = None
    else:
        product_id_match = re.match(r'^\d+', filename)
    product_id = product_id_match.group(0) if product_id_match else ""
    
    # Extract product name (everything after the product ID)
    product_name = filename
    if product_id:
        # Remove product ID from the beginning of the filename
        product_name = filename[len(product_id):].strip()
    
    # Clean up potential encoding issues in product name
    product_name = fix_encoding_issues(product_name)
    
    # Add product info to each element
    for element in elements:
        element['metadata']['product_category'] = product_category
        element['metadata']['product_id'] = product_id
        element['metadata']['product_name'] = product_name
        
        # Also add to text for better searchability
        product_info = f"Produktkategorie: {product_category}, "
        if product_id:
            product_info += f"Produkt ID: {product_id}, "
        if product_name:
            product_info += f"Produktname: {product_name}. "
            
        element['text'] = product_info + element['text']
    
    return elements

def process_all_files():
    # Define input and output directories using absolute paths
    base_dir = Path(__file__).parent.parent  # Get the parent directory of the current script's parent
    input_dir = base_dir / 'src' / 'chunks' / 'unstructured' / 'basic_chunking'
    output_dir = input_dir / 'enriched_chunks'

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
        
    # Process all JSON files recursively, excluding the enriched_chunks folder
    for file_path in input_dir.glob('**/*.json'):
        # Skip files in the enriched_chunks directory
        if 'enriched_chunks' in file_path.parts:
            continue
            
        try:
            # Read input file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Modify the data - first add product info, then prefix
            modified_data = add_product_info(data, file_path)
            modified_data = add_prefix_to_text(modified_data)

            # Generate a unique filename (using stem to avoid duplicates)
            # Add parent directory names to make filename unique
            parent_parts = file_path.parent.relative_to(input_dir).parts if file_path.parent != input_dir else []
            unique_name = "_".join([*parent_parts, file_path.stem]) + ".json"
            output_path = output_dir / unique_name
            
            # Write to output file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(modified_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
if __name__ == "__main__":
    process_all_files()
