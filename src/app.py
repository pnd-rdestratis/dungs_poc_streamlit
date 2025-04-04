import streamlit as st
from retrieve_pinecone import initialize_clients, search
import os
import json
from pathlib import Path
from streamlit_pdf_viewer import pdf_viewer
from llm_processing import analyze_content_with_llm
from dotenv import load_dotenv

# from utils.enrich_chunks import fix_encoding_issues

USE_STREAMLIT_SECRETS = False

# Environment variable setup
if USE_STREAMLIT_SECRETS:
    # Use Streamlit secrets
    os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']
    os.environ['AZURE_OPENAI_API_KEY'] = st.secrets['AZURE_OPENAI_API_KEY']
else:
    # Load environment variables from .env file
    load_dotenv()
    os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')

    if not os.environ['PINECONE_API_KEY'] or not os.environ['AZURE_OPENAI_API_KEY']:
        raise EnvironmentError(
            "Required environment variables PINECONE_API_KEY and AZURE_OPENAI_API_KEY must be set in .env file")

# Base path for documents
DOCS_PATH = Path(__file__).parent.parent / "documents"

# Path to product index JSON file
PRODUCT_INDEX_PATH = Path(__file__).parent.parent / "search_index_data" / "product_index.json"

# Page config
st.set_page_config(page_title="Dungs Search", layout="wide")

# Index options
INDEXES = {
    "Chunking by Title": "dungs-poc-by-title-chunking",
    "Chunking by Page": "dungs-poc-by-page-chunking",
    "Basic Chunking 500 Tokens": "dungs-poc-basic-chunking",
    "Basic Chunking 1000 Tokens": "dungs-poc-basic-chunking-1000",
    "Basic Chunking 500 Tokens Enriched": "dungs-poc-basic-chunking-500-enriched",
    "Basic Chunking All Documents": "dungs-poc-basic-chunking-all-documents"
}

# Set fixed index to use
SELECTED_INDEX = "Basic Chunking All Documents"


def load_product_index():
    """Load product index from JSON file."""
    try:
        with open(PRODUCT_INDEX_PATH, 'r', encoding='utf-8') as f:
            product_data = json.load(f)
        return product_data.get('products', [])
    except Exception as e:
        st.error(f"Error loading product index: {str(e)}")
        return []


def get_category_to_products_mapping(products):
    """Create a mapping from categories to their products."""
    category_to_products = {}

    # Add "All Categories" as first option
    category_to_products["All Categories"] = ["All Documents"]

    # Group products by category
    for product in products:
        category = product.get('product_category', '')
        filename = product.get('filename', '')

        if not category or not filename:
            continue

        if category not in category_to_products:
            category_to_products[category] = []

        # Add the product to its category list if not already there
        if filename not in category_to_products[category]:
            category_to_products[category].append(filename)

        # Also add to All Categories
        if filename not in category_to_products["All Categories"]:
            category_to_products["All Categories"].append(filename)

    return category_to_products


def get_all_pdfs(docs_path: Path) -> dict:
    """Scan the documents directory and all subdirectories for PDF files.
    Returns a dictionary of categories/folders and their PDF files."""
    categories = {}
    categories["All Categories"] = []  # Initialize with empty list

    # Add the "All Documents" option first
    categories["All Categories"].append("All Documents")

    # Walk through all directories and subdirectories
    for root, _, files in os.walk(docs_path):
        pdf_files = [file for file in files if file.lower().endswith('.pdf')]
        if not pdf_files:
            continue

        rel_root = os.path.relpath(root, docs_path)
        category = "Root" if rel_root == "." else rel_root

        if category not in categories:
            categories[category] = []

        for file in pdf_files:
            file_path = os.path.join(rel_root, file) if rel_root != "." else file
            categories[category].append(file_path)
            # Also add to All Categories
            if file_path not in categories["All Categories"]:
                categories["All Categories"].append(file_path)

    return categories


def get_available_pdfs_with_product_index():
    """Get available PDFs with product index information."""
    # Load product index
    products = load_product_index()

    # Create category to products mapping
    category_to_products = get_category_to_products_mapping(products)

    return category_to_products


def display_single_pdf_source(filename: str, page: int, key_prefix: str, counter: int):
    """Display a single PDF source with subfolder support."""
    # Try direct path first
    pdf_path = DOCS_PATH / filename

    # If file doesn't exist at direct path, search subfolders
    if not pdf_path.exists():
        for root, _, files in os.walk(DOCS_PATH):
            if os.path.basename(filename) in files:
                pdf_path = Path(os.path.join(root, os.path.basename(filename)))
                break

    if pdf_path.exists():
        st.markdown(f"**{filename} (Page {page})**")
        pdf_viewer(
            pdf_path,
            width=800,
            height=800,
            pages_to_render=[page],
            render_text=True,
            key=f"{key_prefix}_{filename.replace('/', '_')}_{page}_{counter}"
        )
    else:
        st.error(f"File not found: {filename}")


def extract_citation(text: str) -> list:
    """Extract filename and page number from citation format [filename, Page/Seite X]."""
    import re
    pattern = r'\[(.*?),\s*(?:Page|Seite)\s*(\d+)\]'
    matches = re.findall(pattern, text)

    seen = set()
    unique_citations = []

    for filename, page in matches:
        citation = (filename, int(page))
        if citation not in seen:
            seen.add(citation)
            unique_citations.append(citation)

    return unique_citations


def transform_filenames(files_list):
    """Transform filenames by removing .pdf extension for display while keeping original mapping."""
    display_to_file = {}
    display_names = []

    for filename in files_list:
        if filename == "All Documents" or filename == "All Files in Category":
            # Keep special options as is
            display_to_file[filename] = filename
            display_names.append(filename)
        else:
            display_name = filename.replace('.pdf', '')
            display_to_file[display_name] = filename
            display_names.append(display_name)

    return display_names, display_to_file


def main():
    st.markdown(
        r"""
        <style>
            .stAppDeployButton {display:none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main title
    st.title("📚 Supportcenter Assistant")
    st.write("I am your AI Supportcenter Assistant, ask me anything about the Test Documents!")

    # Get product categories and documents from product index
    category_to_products = get_available_pdfs_with_product_index()

    # Extract all category names and sort them
    category_names = list(category_to_products.keys())
    if "All Categories" in category_names:
        # Ensure "All Categories" is the first item
        category_names.remove("All Categories")
        category_names = ["All Categories"] + sorted(category_names)

    # Sidebar content
    st.sidebar.image("src/logo.png", width=150)

    with st.sidebar:
        # Category selection
        selected_category = st.sidebar.selectbox("Select Product Category:", category_names)

        # Get files for the selected category
        if selected_category in category_to_products:
            category_files = category_to_products[selected_category]
        else:
            # Fallback if category not found
            category_files = ["All Documents"]

        # Add an "All Files in Category" option at the top of the document selection
        if selected_category != "All Categories":
            display_names, display_to_file = transform_filenames(["All Files in Category"] + category_files)
        else:
            display_names, display_to_file = transform_filenames(category_files)

        # Document selection (filtered by preceding category selection)
        selected_display_name = st.sidebar.selectbox("Select document:", display_names)

        # Determine filter mode
        if selected_display_name == "All Files in Category":
            # Filter by category only
            selected_file = None
            filter_mode = "category"
        elif selected_display_name == "All Documents":
            # Don't filter
            selected_file = None
            filter_mode = "all"
        else:
            # Get the original filename from the display name
            selected_file = display_to_file[selected_display_name]

            # Determine if we're filtering by category or specific file
            if selected_category != "All Categories" and selected_file != "All Documents":
                filter_mode = "file"
            else:
                filter_mode = "category"

        # Search settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            # Note about the fixed index being used
            st.markdown("*💡 Using 'Basic Chunking All Documents' for comprehensive search across all documents*")
            top_k = st.slider("Number of results", 1, 20, 5)
            show_pdfs = st.toggle('📄 Show PDF Sources', value=True)
            use_llm = st.toggle('🤖 Get LLM Response', value=True)

    # Initialize index with the fixed index
    try:
        index, embeddings = initialize_clients(INDEXES[SELECTED_INDEX])
    except Exception as e:
        st.error(f"Error initializing: {str(e)}")
        st.stop()

    # Search input
    query = st.text_input("Enter your search query:")

    if query:
        try:
            # Determine what to search based on the filter mode
            search_file = None
            search_category = None

            if filter_mode == "file":
                # Search in a specific file
                search_file = selected_file
            elif filter_mode == "category" and selected_category != "All Categories":
                # Search in all files of a specific category
                search_category = selected_category

            # Call the search function with the appropriate filter
            results = search(
                query=query,
                index=index,
                embeddings=embeddings,
                selected_file=search_file,
                selected_category=search_category,
                top_k=top_k
            )

            if results:
                if use_llm:
                    with st.spinner('Durchsuche Dokumente...'):
                        response_stream = analyze_content_with_llm(
                            query,
                            results,
                            DOCS_PATH,
                            stream=True
                        )

                        # Create two columns for LLM response and sources
                        llm_col1, llm_col2 = st.columns([1, 1])

                        with llm_col1:
                            st.markdown("#### LLM Response")
                            # Display streaming response
                            full_response = ""
                            message_placeholder = st.empty()

                            # Updated streaming handling
                            for chunk in response_stream:
                                if chunk.content is not None:
                                    full_response += chunk.content
                                    message_placeholder.markdown(full_response + "|")

                            message_placeholder.markdown(full_response)

                        # Display LLM sources in right column
                        with llm_col2:
                            if show_pdfs:
                                citations = extract_citation(full_response)
                                if citations:
                                    st.markdown("#### LLM Response Sources")
                                    for idx, (filename, page) in enumerate(citations):
                                        display_single_pdf_source(filename, page, "llm_response", idx)

                        # Separator
                        st.markdown("---")

                # RAG Sources Section
                st.markdown("### RAG Sources")
                for i, r in enumerate(results):
                    page_num = int(float(r['page']))
                    with st.expander(f"Score: {r['score']:.6f} | {r['source']} | Page: {page_num}", expanded=False):
                        if show_pdfs:
                            # Try direct path first
                            pdf_path = DOCS_PATH / r['source']

                            # If file doesn't exist at direct path, search subfolders
                            if not pdf_path.exists():
                                for root, _, files in os.walk(DOCS_PATH):
                                    if os.path.basename(r['source']) in files:
                                        pdf_path = Path(os.path.join(root, os.path.basename(r['source'])))
                                        break

                            if pdf_path.exists():
                                pdf_viewer(
                                    pdf_path,
                                    width=800,
                                    height=800,
                                    pages_to_render=[page_num],
                                    render_text=True,
                                    key=f"search_pdf_{i}_{page_num}"
                                )
                            else:
                                st.error(f"File not found: {r['source']}")

                        st.markdown("**Chunk Content:**")
                        st.write(r['text'])
            else:
                st.warning("No results found")

        except Exception as e:
            st.error(f"Search error: {str(e)}")


if __name__ == "__main__":
    main()
