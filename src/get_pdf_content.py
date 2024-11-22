import fitz
from PIL import Image
import io
from pathlib import Path
from typing import Tuple, List, Set, Dict
import base64
from openai import OpenAI
import os

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
    """Analyze content using GPT-4 Vision with both chunks and PDF pages."""
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

Please cite your sources using the format (Filename, Page X) when providing information.

Relevant chunks from vector search:
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

if __name__ == "__main__":
    # Your specific PDF file and page
    pdf_path = "/Users/riccardodestratis/PycharmProjects/dungs_poc/documents/MPA41_Handbuch.pdf"
    page_number = 5
    output_dir = "output"

    try:
        # Process PDF page
        text, image_path = process_pdf_page(pdf_path, page_number, output_dir)
        print(f"\nProcessed PDF page {page_number}")
        print(f"Image saved to: {image_path}")

        # Analyze content with GPT-4 Vision
        print("\nAnalyzing content with GPT-4 Vision...")
        analysis = analyze_page_content(image_path, text)

        print("\nAnalysis Results:")
        print("-" * 80)
        print(analysis)
        print("-" * 80)

    except Exception as e:
        print(f"Error: {str(e)}")
