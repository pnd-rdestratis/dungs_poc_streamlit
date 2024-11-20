import streamlit as st
from retrieve_hybrid import initialize_clients, search

st.set_page_config(page_title="Dungs Search", layout="wide")

# Initialize
try:
    index, embeddings = initialize_clients()
except Exception as e:
    st.error(f"Error initializing: {str(e)}")
    st.stop()

st.title("📚 Dungs Documentation Search")

# Two columns
col1, col2 = st.columns([1, 2])

with col1:
    # File selection with "All Documents" option
    files = [
        "All Documents",  # Added this option
        "FRM_Anleitung-engl.pdf",
        "FRS_Anleitung.pdf",
        "MBE-DMS_Anleitung.pdf",
        "MBE-PS-GW_Anleitung.pdf",
        "MBE_Anleitung.pdf",
        "MBE_Datenblatt.pdf",
        "MPA41_Handbuch.pdf"
    ]
    selected_file = st.selectbox("Select document:", files)

    # Convert selection to None if "All Documents" is selected
    selected_file = None if selected_file == "All Documents" else selected_file

    # Search settings
    with st.expander("Search Settings", expanded=True):
        alpha = st.slider(
            "Search Type",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="0 = Keyword Search, 1 = Semantic Search"
        )
        top_k = st.slider("Number of results", 1, 20, 5)

with col2:
    query = st.text_input("Enter your search query:")

    if query:
        try:
            results = search(
                query=query,
                index=index,
                embeddings=embeddings,
                selected_file=selected_file,  # This will be None for "All Documents"
                top_k=top_k,
                alpha=alpha
            )

            if results:
                st.success(f"Found {len(results)} results")

                for r in results:
                    with st.expander(
                        f"Score: {r['score']:.6f} | {r['source']} | Page: {r['page']}",
                        expanded=True
                    ):
                        st.markdown("**Text:**")
                        st.write(r['text'])

            else:
                st.warning("No results found")

        except Exception as e:
            st.error(f"Search error: {str(e)}")
