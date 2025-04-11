import json
import os
import re
from pathlib import Path

def extract_filename_info(filename):
    """
    Extract information from filenames following the pattern:
    6-stellige Artikel Nr. _ Kategorie _ Typ _ (evtl. Zusätze für Baureihe oder Variante) _ BMA / DB _ Sprache
    
    Args:
        filename (str): The filename to parse
        
    Returns:
        dict: Dictionary containing extracted information
    """
    # Remove file extension
    filename = filename.replace('.pdf.json', '').replace('.pdf', '')
    
    # Initialize result dictionary
    result = {
        'original_filename': filename,
        'product_id': '',
        'product_name': '',
        'product_category': '',
        'document_type': ''
    }
    
    # Special case for RepNews files
    if filename.startswith('RepNews'):
        # Example: RepNews_2024-009_New-SAV_1R0
        # or: RepNews 2015-02_MB Filter 405-407
        repnews_pattern = r'(RepNews[_\s-]*)(.+)'
        match = re.match(repnews_pattern, filename)
        if match:
            result['product_id'] = 'RepNews'
            content_part = match.group(2) or ''
            result['product_name'] = content_part.strip()
            result['product_category'] = 'RepNews'
            result['document_type'] = 'RepNews'
        return result
    
    # Special case for 999999 files (special documents)
    if filename.startswith('999999'):
        parts = filename.split('_')
        result['product_id'] = '999999'
        if len(parts) > 1:
            # Find document type part
            doc_type_index = -1
            for i, part in enumerate(parts):
                if 'BMA' in part or 'DB' in part:
                    doc_type_index = i
                    result['document_type'] = 'Betriebs- und Montageanleitung' if 'BMA' in part else 'Datenblatt'
                    break
            
            # Extract product name (everything between ID and document type)
            if doc_type_index > 1:  # If we found a document type and it's not right after the ID
                result['product_name'] = '_'.join(parts[1:doc_type_index])
                # Extract product category (first part of the product name)
                if parts[1:doc_type_index]:
                    # For example, from "Kugelhahn_KH", extract "Kugelhahn" as category
                    first_part = parts[1]
                    result['product_category'] = first_part.split('_')[0] if '_' in first_part else first_part
            else:
                # If no document type found or it's right after the ID, take everything after ID
                result['product_name'] = '_'.join(parts[1:])
                # Extract product category (first part of the product name)
                if parts[1:]:
                    # For example, from "Kugelhahn_KH", extract "Kugelhahn" as category
                    first_part = parts[1]
                    result['product_category'] = first_part.split('_')[0] if '_' in first_part else first_part
        return result
    
    # Regular case: Try to match the standard pattern
    # First, extract the product ID (6-digit number at the beginning)
    id_match = re.match(r'^(\d{6})', filename)
    if id_match:
        result['product_id'] = id_match.group(1)
        
        # Remove the product ID from the filename
        remaining = filename[len(result['product_id']):].strip('_')
        
        # Split the remaining parts by underscore
        parts = remaining.split('_')
        
        # Extract document type (BMA or DB)
        doc_type_found = False
        doc_type_index = -1
        for i, part in enumerate(parts):
            if 'BMA' in part:
                result['document_type'] = 'Betriebs- und Montageanleitung'
                doc_type_found = True
                doc_type_index = i
                break
            elif 'DB' in part:
                result['document_type'] = 'Datenblatt'
                doc_type_found = True
                doc_type_index = i
                break
        
        # Extract product name (everything before document type)
        if doc_type_found and doc_type_index > 0:
            result['product_name'] = '_'.join(parts[:doc_type_index])
            # Extract product category (first part of the product name)
            if parts:
                # For example, from "Kugelhahn_KH", extract "Kugelhahn" as category
                result['product_category'] = parts[0].split('_')[0] if '_' in parts[0] else parts[0]
        elif not doc_type_found:
            # If no document type found, take all parts as product name
            result['product_name'] = '_'.join(parts)
            # Extract product category (first part of the product name)
            if parts:
                # For example, from "Kugelhahn_KH", extract "Kugelhahn" as category
                result['product_category'] = parts[0].split('_')[0] if '_' in parts[0] else parts[0]
    
    return result

def enrich_chunks_with_filename_info():
    """
    Enrich chunks with information extracted from filenames.
    """
    # Define the path to the chunks directory
    chunks_dir = Path('/Users/riccardodestratis/PycharmProjects/dungs_poc/documents-one/chunks')
    
    print(f"Looking for files in: {chunks_dir}")
    
    # Check if the directory exists
    if not chunks_dir.exists():
        print(f"Directory not found: {chunks_dir}")
        return
    
    # Get all JSON files in the directory
    json_files = list(chunks_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files in the directory")
    
    # Process each file
    for file_path in json_files:
        filename = file_path.name
        print(f"Processing: {filename}")
        
        # Extract information from filename
        extracted_info = extract_filename_info(filename)
        
        try:
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Enrich each chunk with the extracted information
            for chunk in data:
                # Add extracted information to metadata
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}
                
                chunk['metadata']['product_id'] = extracted_info['product_id']
                chunk['metadata']['product_name'] = extracted_info['product_name']
                chunk['metadata']['product_category'] = extracted_info['product_category']
                chunk['metadata']['document_type'] = extracted_info['document_type']
                
                # Remove old prefix if it exists
                text = chunk['text']
                old_prefix_pattern = rf"Produkt ID: {extracted_info['product_id']}, Produktname: {extracted_info['product_name']}, Dokumenttyp: [^\.]+\. "
                text = re.sub(old_prefix_pattern, "", text)
                
                # Also remove any existing new format prefix
                new_prefix_pattern = rf"Dieser Text ist für Produt: {extracted_info['product_name']}. Produkt ID: {extracted_info['product_id']}. Der Dokumentyp is [^\.]+\. "
                text = re.sub(new_prefix_pattern, "", text)
                
                # Add to text in the requested format
                product_info = f"Dieser Text ist für Produt: {extracted_info['product_name']}. "
                product_info += f"Produkt ID: {extracted_info['product_id']}. "
                product_info += f"Der Dokumentyp is {extracted_info['document_type']}. "
                
                chunk['text'] = product_info + text
            
            # Write the enriched data back to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    print("Enrichment completed.")

if __name__ == "__main__":
    enrich_chunks_with_filename_info()
