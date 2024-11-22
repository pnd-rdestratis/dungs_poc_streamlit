import streamlit as st
from retrieve_hybrid import initialize_clients, search
import os
from pathlib import Path
from streamlit_pdf_viewer import pdf_viewer
from get_pdf_content import analyze_content_with_llm

# Set environment variables from Streamlit secrets
os.environ['PINECONE_API_KEY'] = st.secrets['PINECONE_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Base path for documents
DOCS_PATH = Path("/Users/riccardodestratis/PycharmProjects/dungs_poc/documents")

st.set_page_config(page_title="Dungs Search", layout="wide")
st.sidebar.image("src/logo.png",width=150)

# Index options
INDEXES = {
    "Chunking by Title": "dungs-poc-by-title-chunking",
    "Chunking by Page": "dungs-poc-by-page-chunking",
    "Basic Chunking 500 Tokens": "dungs-poc-basic-chunking",
    "Basic Chunking 1000 Tokens" : "dungs-poc-basic-chunking-1000"
}

# Select index first
selected_index_name = st.sidebar.radio(
    "Select Vector Index:",
    options=list(INDEXES.keys())
)

# Initialize with selected index
try:
    index, embeddings = initialize_clients(INDEXES[selected_index_name])
except Exception as e:
    st.error(f"Error initializing: {str(e)}")
    st.stop()

st.title("📚 Documentation RAG PoC")

# Two columns
col1, col2 = st.columns([1, 2])

with col1:
    # File selection with "All Documents" option
    files = [
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
    selected_file = st.selectbox("Select document:", files)
    selected_file = None if selected_file == "All Documents" else selected_file

    # Search settings
    with st.expander("Search Settings", expanded=True):
        top_k = st.slider("Number of results", 1, 20, 5)
        show_pdfs = st.toggle('📄 Show PDF Sources', value=False)
        use_llm = st.toggle('🤖 Get LLM Response', value=False)

with col2:
    # Use a form to control when the search is triggered
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

            if results:  # First check if we have results
                if use_llm:  # Then check if LLM is enabled
                    with st.spinner('Analyzing content with LLM...'):
                        with st.chat_message("assistant"):
                            response_stream = analyze_content_with_llm(
                                query,
                                results,
                                DOCS_PATH,
                                stream=True
                            )
                            full_response = ""
                            message_placeholder = st.empty()

                            # Process the streaming response
                            for chunk in response_stream:
                                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                                    full_response += chunk.choices[0].delta.content
                                    # Update the message placeholder with the accumulated response
                                    message_placeholder.markdown(full_response + "▌")

                            # Final update without the cursor
                            message_placeholder.markdown(full_response)
                        st.markdown("---")

                # Show regular search results (moved outside the use_llm condition)
                for i, r in enumerate(results):
                    page_num = int(float(r['page']))

                    with st.expander(
                            f"Score: {r['score']:.6f} | {r['source']} | Page: {page_num}",
                            expanded=False
                    ):
                        st.markdown("**Text:**")
                        st.write(r['text'])

                        if show_pdfs:
                            pdf_path = DOCS_PATH / r['source']
                            if pdf_path.exists():
                                st.markdown(f"**Source: {r['source']} (Page {page_num})**")
                                pdf_viewer(
                                    pdf_path,
                                    width=700,
                                    pages_to_render=[page_num],
                                    render_text=True,
                                    key=f"pdf_viewer_{i}_{page_num}"
                                )

            else:  # No results found condition moved here
                st.warning("No results found")

        except Exception as e:
            st.error(f"Search error: {str(e)}")
