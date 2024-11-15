"""
JSON Processing Script with parent-child relationship processing
"""

from typing import List, Dict, Any
import json
import os
from pathlib import Path

def clean_metadata(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes specified fields from metadata of each object.
    """
    for obj in data:
        if 'metadata' in obj:
            # Remove data_source completely
            if 'data_source' in obj['metadata']:
                del obj['metadata']['data_source']

            # Remove specific fields if they exist
            fields_to_remove = ['permissions_data', 'filesize_bytes', 'date_created',
                              'date_modified', 'date_processed']
            for field in fields_to_remove:
                if field in obj['metadata']:
                    del obj['metadata'][field]
    return data

def build_child_relationships(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds child_ids lists for each object based on parent_id relationships and
    places them in metadata right after page_number.
    """
    # First create a mapping of parent_ids to child element_ids
    parent_to_children: Dict[str, List[str]] = {}

    # Build parent-child relationships
    for obj in data:
        element_id = obj.get('element_id')
        parent_id = obj.get('metadata', {}).get('parent_id')

        if parent_id:
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            parent_to_children[parent_id].append(element_id)

    # Add child_ids to each object's metadata
    for obj in data:
        element_id = obj.get('element_id')
        if 'metadata' not in obj:
            obj['metadata'] = {}

        # Create new metadata dict with desired order
        new_metadata = {}
        for key, value in obj['metadata'].items():
            new_metadata[key] = value
            if key == 'page_number':  # Place child_ids after page_number
                new_metadata['child_ids'] = parent_to_children.get(element_id, [])

        # If there was no page_number, add child_ids at the end
        if 'page_number' not in new_metadata:
            new_metadata['child_ids'] = parent_to_children.get(element_id, [])

        obj['metadata'] = new_metadata

    return data

def remove_image_objects(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes all objects with type 'Image' from a list of dictionaries.
    """
    filtered_data = [obj for obj in data if obj.get('type') != 'Image']
    return filtered_data

def replace_umlauts(json_str: str) -> str:
    """
    Replaces German umlauts in a string with their ASCII equivalents.
    """
    replacements = {
        'ä': 'ae',
        'ü': 'ue',
        'ö': 'oe',
        'ß': 'ss'
    }

    for umlaut, replacement in replacements.items():
        json_str = json_str.replace(umlaut, replacement)
    return json_str

def remove_page_numbers_objects(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes all objects with type 'PageNumber' from a list of dictionaries.
    """
    filtered_data = [obj for obj in data if obj.get('type') != 'PageNumber']
    return filtered_data

def process_json_file(file_path: Path) -> None:
    """
    Processes a single JSON file by removing Image objects, PageNumber objects, replacing umlauts,
    adding child relationships, and cleaning metadata.
    """
    try:
        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Remove Image objects
        original_length = len(data)
        filtered_data = remove_image_objects(data)
        images_removed = original_length - len(filtered_data)

        # Remove PageNumber objects
        before_page_numbers = len(filtered_data)
        filtered_data = remove_page_numbers_objects(filtered_data)
        page_numbers_removed = before_page_numbers - len(filtered_data)

        # Build child relationships
        enhanced_data = build_child_relationships(filtered_data)

        # Clean metadata
        cleaned_data = clean_metadata(enhanced_data)

        # Replace umlauts
        json_str = json.dumps(cleaned_data, ensure_ascii=False)
        processed_str = replace_umlauts(json_str)
        final_data = json.loads(processed_str)

        # Write processed data back to file
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(final_data, file, ensure_ascii=False, indent=2)

        print(f"Processed {file_path.name}: removed {images_removed} Image objects, {page_numbers_removed} PageNumber objects, cleaned metadata")

    except json.JSONDecodeError as e:
        print(f"JSON Error in {file_path}: {str(e)}")
    except IOError as e:
        print(f"IO Error with {file_path}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error with {file_path}: {str(e)}")

def main() -> None:
    """
    Main function that walks through the directory and processes JSON files.
    """
    folder_path = Path("../chunks_settings_2")

    if not folder_path.exists():
        print(f"Error: Directory {folder_path} does not exist")
        return

    total_files = 0
    json_files = 0

    print(f"Starting processing in: {folder_path.absolute()}")

    # Process all JSON files in directory and subdirectories
    for root, _, files in os.walk(folder_path):
        total_files += len(files)
        for file in files:
            if file.endswith('.json'):
                json_files += 1
                file_path = Path(root) / file
                process_json_file(file_path)

    print("\nProcessing complete!")
    print(f"Total files scanned: {total_files}")
    print(f"JSON files processed: {json_files}")

if __name__ == "__main__":
    main()
