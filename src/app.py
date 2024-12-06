import streamlit as st
from retrieve_hybrid import initialize_clients, search
import os
from pathlib import Path
from streamlit_pdf_viewer import pdf_viewer
from llm_processing import analyze_content_with_llm

# Set environment variables from Streamlit secrets
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Base path for documents
DOCS_PATH = Path(__file__).parent.parent / "documents"

# Page config
st.set_page_config(page_title="Dungs Search", layout="wide")

# Index options
INDEXES = {
    "Chunking by Title": "dungs-poc-by-title-chunking",
    "Chunking by Page": "dungs-poc-by-page-chunking",
    "Basic Chunking 500 Tokens": "dungs-poc-basic-chunking",
    "Basic Chunking 1000 Tokens": "dungs-poc-basic-chunking-1000",
    "Basic Chunking 500 Tokens Enriched": "dungs-poc-basic-chunking-500-enriched"
}

# Available documents
FILES = [
    "All Documents",
    "FRM_Anleitung-engl.pdf",
    "FRS_Anleitung.pdf",
    "MBE-DMS_Anleitung.pdf",
    "MBE-PS-GW_Anleitung.pdf",
    "MBE-PS_Anleitung.pdf",
    "MBE_Anleitung.pdf",
    "MBE_Datenblatt.pdf",
    "MPA41_Handbuch.pdf"
]

def display_single_pdf_source(filename: str, page: int, key_prefix: str, counter: int):
    """Display a single PDF source."""
    pdf_path = DOCS_PATH / filename
    if pdf_path.exists():
        st.markdown(f"**{filename} (Page {page})**")
        pdf_viewer(
            pdf_path,
            width=800,
            height=800,
            pages_to_render=[page],
            render_text=True,
            key=f"{key_prefix}_{filename}_{page}_{counter}"
        )


def extract_citation(text: str) -> list:
    """Extract filename and page number from citation format [filename, Page/Seite X]."""
    import re
    pattern = r'\[(.*?),\s*(?:Page|Seite)\s*(\d+)\]'
    matches = re.findall(pattern, text)

    # Use a set to track seen citations while preserving order
    seen = set()
    unique_citations = []

    for filename, page in matches:
        citation = (filename, int(page))
        # Only add if we haven't seen this exact combination before
        if citation not in seen:
            seen.add(citation)
            unique_citations.append(citation)

    return unique_citations

def transform_filenames():
    """Transform filenames by removing .pdf extension for display while keeping original mapping."""
    display_to_file = {"All Documents": "All Documents"}
    display_names = ["All Documents"]

    for filename in FILES[1:]:  # Skip "All Documents"
        display_name = filename.replace('.pdf', '')
        display_to_file[display_name] = filename
        display_names.append(display_name)

    return display_names, display_to_file

def main():
    st.title("📚 Supportcenter Assistant")

    # Sidebar content
    st.sidebar.image("src/logo.png", width=150)

    with st.sidebar:
        # Transform filenames for display
        display_names, display_to_file = transform_filenames()

        # Document selection
        selected_display_name = st.selectbox("Select document:", display_names)
        selected_file = None if selected_display_name == "All Documents" else display_to_file[selected_display_name]

        # Search settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.markdown("*💡 'Basic Chunking 500 Tokens Enriched' provides best accuracy across all documents*")
            selected_index_name = st.radio(
                "Select Vector Index:",
                options=list(INDEXES.keys()),
                index=list(INDEXES.keys()).index("Basic Chunking 500 Tokens Enriched")
            )
            top_k = st.slider("Number of results", 1, 20, 5)
            show_pdfs = st.toggle('📄 Show PDF Sources', value=True)
            use_llm = st.toggle('🤖 Get LLM Response', value=True)

    # Initialize index
    try:
        index, embeddings = initialize_clients(INDEXES[selected_index_name])
    except Exception as e:
        st.error(f"Error initializing: {str(e)}")
        st.stop()

    # Search input
    query = st.text_input("Enter your search query:")

    if query:
        try:
            results = search(
                query=query,
                index=index,
                embeddings=embeddings,
                selected_file=selected_file,
                top_k=top_k
            )

            if results:
                # LLM Response Section with two columns
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

                            for chunk in response_stream:
                                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[
                                    0].delta.content is not None:
                                    full_response += chunk.choices[0].delta.content
                                    message_placeholder.markdown(full_response + "▌")

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

                # RAG Sources Section (full width, below columns)
                st.markdown("### RAG Sources")
                for i, r in enumerate(results):
                    page_num = int(float(r['page']))
                    with st.expander(f"Score: {r['score']:.6f} | {r['source']} | Page: {page_num}", expanded=False):
                        if show_pdfs:
                            pdf_path = DOCS_PATH / r['source']
                            if pdf_path.exists():
                                pdf_viewer(
                                    pdf_path,
                                    width=800,
                                    height=800,
                                    pages_to_render=[page_num],
                                    render_text=True,
                                    key=f"search_pdf_{i}_{page_num}"
                                )

                        st.markdown("**Chunk Content:**")
                        st.write(r['text'])
            else:
                st.warning("No results found")

        except Exception as e:
            st.error(f"Search error: {str(e)}")

if __name__ == "__main__":
    main()
