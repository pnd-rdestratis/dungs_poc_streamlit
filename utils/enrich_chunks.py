import json
import os
from pathlib import Path

def add_prefix_to_text(elements):
    for element in elements:
        filename = element['metadata']['filename'].replace('.pdf', '')
        element['text'] = f"Dieser Text ist im Dokument: {filename} zu finden. {element['text']}"
    return elements

def process_all_files():
    # Define input and output directories
    input_dir = Path('../src/chunks/unstructured/basic_chunking')
    output_dir = input_dir / 'enriched_chunks'

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Process each JSON file in input directory
    for file_path in input_dir.glob('*.json'):
        try:
            # Read input file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Modify the data
            modified_data = add_prefix_to_text(data)

            # Write to output file in enriched_chunks folder
            output_path = output_dir / file_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(modified_data, f, indent=2, ensure_ascii=False)

            print(f"Processed: {file_path.name}")

        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

if __name__ == "__main__":
    process_all_files()
