import fitz
from PIL import Image
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
    # Clear output directory at the start of each search
    output_dir = "output"
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(exist_ok=True)

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # Get unique pages
    unique_pages = get_unique_pages(results)

    # Process each unique page
    page_contents = []
    images_and_texts = []

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
    prompt = f"""Please answer the following question based on the provided content:

Question: {query}

I will provide you with:
1. Relevant chunks from the vector search
2. Full page content and images from the PDFs

Please structure your response in this format:


When possible try to structure the respone nicely, using bullet points ore lists eletments.

Always respond in the same language as the question. So if a question is asked in German but the documents are
in English, you still need to respond in German.
Provide your answer with inline citations using [Filename, Page X] format (no matter which language).
This formatting is crucial for creating clickable links in the interface.
Please consider that potentially not all content is relevant to respond to the question.

If you cannot answer the question based on the provided content tell the user.
If the document name in the retrieved chunks does not match the document name specified in the question
Tell the user something like this:

Search over all documents was not successful, please try again by selecting the specific document in the Sidebar.
Also this should always be in the language of the user's query

Potentially Relevant chunks from vector search:
"""

    # Add chunks information
    for r in results:
        prompt += f"\nFrom {r['source']}, Page {int(float(r['page']))}: {r['text']}\n"

    # Create messages with text and images
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    # Add images and their full page content
    for page in page_contents:
        messages[0]["content"].extend([
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{page['image']}"
                }
            },
            {
                "type": "text",
                "text": f"\nFull page content from {page['filename']}, Page {page['page']}:\n{page['text']}"
            }
        ])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
        stream=stream
    )

    if stream:
        # Return the streaming response generator
        return response
    else:
        # Return complete response for non-streaming case
        return response.choices[0].message.content


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
