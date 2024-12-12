import streamlit as st
from retrieve_hybrid import initialize_clients, search
import os
from pathlib import Path
from streamlit_pdf_viewer import pdf_viewer
from llm_processing import analyze_content_with_llm
from dotenv import load_dotenv
from streamlit_oauth import OAuth2Component
import requests

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

    AUTHORIZE_URL = os.environ.get('AUTHORIZE_URL')
    TOKEN_URL = os.environ.get('TOKEN_URL')
    REFRESH_TOKEN_URL = os.environ.get('REFRESH_TOKEN_URL')
    REVOKE_TOKEN_URL = os.environ.get('REVOKE_TOKEN_URL')
    CLIENT_ID = os.environ.get('CLIENT_ID')
    CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
    REDIRECT_URI = os.environ.get('REDIRECT_URI')
    SCOPE = "User.ReadBasic.All"
    if not os.environ['PINECONE_API_KEY'] or not os.environ['AZURE_OPENAI_API_KEY']:
        raise EnvironmentError(
            "Required environment variables PINECONE_API_KEY and AZURE_OPENAI_API_KEY must be set in .env file")

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


def get_user_info(access_token):
    """Get user info from Microsoft Graph API"""
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    response = requests.get('https://graph.microsoft.com/v1.0/me', headers=headers)
    if response.status_code == 200:
        return response.json()
    return None


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

    seen = set()
    unique_citations = []

    for filename, page in matches:
        citation = (filename, int(page))
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


def render_login():
    """Render the login page"""
    st.markdown(
        """
        <style>
            .stAppDeployButton {display:none;}
            .main {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .stButton > button {
                background-color: #0078D4 !important;
                color: white !important;
                border: none !important;
                padding: 0.5rem 1rem !important;
                font-size: 1rem !important;
                border-radius: 4px !important;
                width: 200px !important;
            }
            div[data-testid="stVerticalBlock"] {
                gap: 0px !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Use columns for horizontal centering
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Welcome to</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Supportcenter AI Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; margin-bottom: 2rem; color: #666;'>Please login to continue</p>",
                    unsafe_allow_html=True)

        # Center the button
        col4, col5, col6 = st.columns([1, 2, 1])
        with col5:
            oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, REFRESH_TOKEN_URL,
                                     REVOKE_TOKEN_URL)
            result = oauth2.authorize_button("Login with Microsoft", REDIRECT_URI, SCOPE, extras_params={"prompt" : "select_account"})

            if result and 'token' in result:
                st.session_state.token = result.get('token')
                user_info = get_user_info(st.session_state.token['access_token'])
                user_email = user_info.get("userPrincipalName")
                st.write(user_info)
                if user_email in os.environ.get("AUTHORIZED_USERS",""):
                    st.session_state.user_info = user_info
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.write("User not allowed")



def render_app():
    """Render the main application"""

    st.markdown(
        r"""
        <style>
            .stAppDeployButton {display:none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Header with user info and avatar
    header_col1, header_col2, header_col3 = st.columns([1, 2, 1])

    with header_col3:
        if 'user_info' in st.session_state:
            user = st.session_state.user_info
            display_name = user.get('displayName', 'User')
            st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; align-items: center; gap: 10px;">
                    <img src="https://api.dicebear.com/6.x/initials/svg?seed={display_name}" 
                         style="width: 40px; height: 40px; border-radius: 50%;" 
                         alt="avatar">
                </div>
            """, unsafe_allow_html=True)

    # Main title with personalized greeting
    if 'user_info' in st.session_state:
        # Extract first name from displayName instead of using givenName
        display_name = st.session_state.user_info.get('displayName', 'there')
        first_name = display_name.split()[0] if display_name else 'there'
        st.title("📚 Supportcenter Assistant")
        st.write(f"Hi {first_name}")
        st.write("I am your AI Supportcenter Assistant, ask me anything about the Test Documents!")
    else:
        st.title("📚 Supportcenter Assistant")

    # Sidebar content
    st.sidebar.image("logo.png", width=150)

    # Add logout button
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

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


def main():
    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding-top: 2rem;
        }
        .stButton>button {
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Render login or main app based on authentication state
    if not st.session_state.authenticated:
        render_login()
    else:
        render_app()


if __name__ == "__main__":
    main()
