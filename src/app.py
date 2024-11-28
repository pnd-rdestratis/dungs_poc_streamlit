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
DOCS_PATH = Path("/Users/riccardodestratis/PycharmProjects/dungs_poc/documents")

# Page config
st.set_page_config(page_title="Dungs Search", layout="wide")
st.sidebar.image("src/logo.png", width=150)

# Index options
INDEXES = {
    "Chunking by Title": "dungs-poc-by-title-chunking",
    "Chunking by Page": "dungs-poc-by-page-chunking",
    "Basic Chunking 500 Tokens": "dungs-poc-basic-chunking",
    "Basic Chunking 1000 Tokens": "dungs-poc-basic-chunking-1000"
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
            key=f"{key_prefix}_{filename}_{page}_{counter}"  # Added counter to make key unique
        )

def extract_citation(text: str) -> list:
    """Extract filename and page number from citation format [filename, Page/Seite X]."""
    citations = []
    import re
    # Handle both English "Page" and German "Seite"
    pattern = r'\[(.*?),\s*(?:Page|Seite)\s*(\d+)\]'
    matches = re.findall(pattern, text)
    return [(filename, int(page)) for filename, page in matches]

def main():
    st.title("📚 Documentation RAG PoC")

    # Select index
    selected_index_name = st.sidebar.radio(
        "Select Vector Index:",
        options=list(INDEXES.keys())
    )

    # Initialize index
    try:
        index, embeddings = initialize_clients(INDEXES[selected_index_name])
    except Exception as e:
        st.error(f"Error initializing: {str(e)}")
        st.stop()

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        # Document selection
        selected_file = st.selectbox("Select document:", FILES)
        selected_file = None if selected_file == "All Documents" else selected_file

        # Search settings
        with st.expander("Search Settings", expanded=True):
            top_k = st.slider("Number of results", 1, 20, 5)
            show_pdfs = st.toggle('📄 Show PDF Sources', value=False)
            use_llm = st.toggle('🤖 Get LLM Response', value=False)

    with col2:
        # Search form
        with st.form(key='search_form'):
            query = st.text_input("Enter your search query:")
            submit_button = st.form_submit_button("Search")

        if submit_button and query:
            try:
                results = search(
                    query=query,
                    index=index,
                    embeddings=embeddings,
                    selected_file=selected_file,
                    top_k=top_k
                )

                if results:
                    # LLM Response Section
                    if use_llm:
                        with st.spinner('Durchsuche Dokumente...'):
                            response_stream = analyze_content_with_llm(
                                query,
                                results,
                                DOCS_PATH,
                                stream=True
                            )

                            # Display streaming response
                            full_response = ""
                            message_placeholder = st.empty()

                            for chunk in response_stream:
                                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[
                                    0].delta.content is not None:
                                    full_response += chunk.choices[0].delta.content
                                    message_placeholder.markdown(full_response + "▌")

                            message_placeholder.markdown(full_response)

                            # Display LLM response sources
                            if show_pdfs:
                                citations = extract_citation(full_response)
                                if citations:
                                    st.markdown("### Response Sources")
                                    for idx, (filename, page) in enumerate(citations):
                                        display_single_pdf_source(filename, page, "llm_response", idx)

                            st.markdown("---")

                    # Search Results Section
                    st.markdown("### RAG Sources")
                    for i, r in enumerate(results):
                        page_num = int(float(r['page']))
                        with st.expander(f"Score: {r['score']:.6f} | {r['source']} | Page: {page_num}", expanded=False):
                            st.markdown("**Text:**")
                            st.write(r['text'])

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
                else:
                    st.warning("No results found")

            except Exception as e:
                st.error(f"Search error: {str(e)}")

if __name__ == "__main__":
    main()
