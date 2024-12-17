import fitz
from PIL import Image
from langchain_openai import AzureChatOpenAI
import io
from pathlib import Path
from typing import Tuple, List, Set, Dict
import base64
from openai import OpenAI
import os
import shutil
def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_unique_pages(results: List[Dict]) -> List[Tuple[str, int]]:
    """Get unique combinations of filenames and page numbers."""
    unique_pages = set()
    for r in results:
        unique_pages.add((r['source'], int(float(r['page']))))
    return list(unique_pages)



def analyze_content_with_llm(query: str, results: List[Dict], docs_path: Path, stream=True) -> str:
    """Analyze content using GPT-4o with both chunks and PDF pages."""
    llm = AzureChatOpenAI(
        openai_api_version="2024-08-01-preview",
        azure_deployment="gpt-4o",
        azure_endpoint="https://azure-openai-dungs.openai.azure.com",
        streaming=True,
        temperature=0.1
    )

    # Clear output directory at the start of each search
    output_dir = "output"
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(exist_ok=True)

    # Get unique pages and process them
    unique_pages = get_unique_pages(results)
    page_contents = []

    for filename, page_num in unique_pages:
        text, image_path = process_pdf_page(
            str(docs_path / filename),
            page_num,
            "output"
        )
        page_contents.append({
            "filename": filename,
            "page": page_num,
            "text": text,
            "image": encode_image(image_path)
        })

    # Prepare the prompt
    prompt = f"""
    You are a support Center Assistant for Dungs, a manufactrer of Combustion Control products . Please answer the following question based on the provided content:

    Question: {query}

    ### Information Provided:
    1. Chunks from the vector search.
    2. Full page content and images extracted from the PDFs.

    ### Response Guidelines:
    - Structure your response clearly, using bullet points or list elements where appropriate.
    - Always answer in the same language as the question. For example, if the question is in German but the documents are in English, respond in German.
    - Include inline citations always in this specifc format [Filename, Page X] for every reference (regardless of the language). This is essential for creating clickable links in the interface.
    - Only use content that is relevant to answer the question. If certain information does not contribute to the response, omit it.
    - Don't mention stuff like: The provided information says... only mention that if you cannot find the relevant information required to answer the question
    ### Handling Special Cases:
    1. If you cannot answer the question based on the provided content:  
       Inform the user that the content is insufficient to provide an answer.  

    2. If the document name in the retrieved chunks does not match the document name specified in the question:  
       Notify the user with the following message (translated into the query's language):  
       "Search over all documents was not successful. Please try again by selecting the specific document in the sidebar."  

    3. If the query specifies a document, but this document is not mentioned in the chunks:  
       Inform the user that the chunks are likely not relevant and suggest selecting the correct document in the sidebar.
       Or that he they shuold specify the question more precisely.

    ### Provided Data:
    Potentially relevant chunks from vector search:
    """

    # Add chunks information
    for r in results:
        prompt += f"\nFrom {r['source']}, Page {int(float(r['page']))}: {r['text']}\n"

    try:
        return llm.stream(prompt)
    except Exception as e:
        print(f"Azure OpenAI Error: {str(e)}")
        return f"An error occurred: {str(e)}"


def process_pdf_page(pdf_path: str, page_number: int, output_dir: str = "output") -> Tuple[str, str]:
    """Extract text and create image from a specific PDF page."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get filename without extension for output naming
    pdf_name = Path(pdf_path).stem

    try:
        # Open PDF
        doc = fitz.open(pdf_path)

        # Convert to 0-based page number
        page_idx = page_number - 1

        if page_idx < 0 or page_idx >= len(doc):
            raise ValueError(f"Invalid page number. PDF has {len(doc)} pages.")

        # Get the page
        page = doc[page_idx]

        # Extract text
        text = page.get_text()

        # Get page as image with higher resolution
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        # Save image
        image_path = f"{output_dir}/{pdf_name}_page_{page_number}.png"
        img.save(image_path, "PNG", dpi=(300, 300))

        doc.close()

        return text, image_path

    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")
