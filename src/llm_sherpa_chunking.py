import os
from datetime import datetime
import json
from pathlib import Path
from llmsherpa.readers import LayoutPDFReader
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, server_url="http://localhost:5010/api/parseDocument?renderFormat=all"):
        self.reader = LayoutPDFReader(server_url)

    def process_document(self, pdf_path):
        """Process a single PDF document using llmsherpa's layout understanding"""
        try:
            # Read the PDF
            doc = self.reader.read_pdf(str(pdf_path))

            # Extract structured content
            structured_content = {
                "metadata": {
                    "filename": Path(pdf_path).name,
                    "processed_date": datetime.now().isoformat()
                },
                "sections": self._process_sections(doc),
                "chunks": self._process_chunks(doc),
                "tables": self._process_tables(doc)
            }

            return True, structured_content
        except Exception as e:
            return False, f"Error processing document: {str(e)}"

    def _process_sections(self, doc):
        """Extract sections with their hierarchy"""
        sections = []
        for section in doc.sections():
            section_data = {
                "title": section.title,
                "level": section.level,
                "content": section.to_text(include_children=True, recurse=True),
                "bbox": section.bbox if hasattr(section, 'bbox') else None,
                "page_num": section.page_idx + 1 if hasattr(section, 'page_idx') and section.page_idx >= 0 else None
            }
            sections.append(section_data)
        return sections

    def _process_chunks(self, doc):
        """Extract chunks with context"""
        chunks = []
        for chunk in doc.chunks():
            chunk_data = {
                "text": chunk.to_context_text(),
                "bbox": chunk.bbox if hasattr(chunk, 'bbox') else None,
                "page_num": chunk.page_idx + 1 if hasattr(chunk, 'page_idx') and chunk.page_idx >= 0 else None,
                "parent_context": chunk.parent_text()
            }
            chunks.append(chunk_data)
        return chunks

    def _process_tables(self, doc):
        """Extract tables with their structure"""
        tables = []
        for table in doc.tables():
            table_data = {
                "text": table.to_text(),
                "html": table.to_html(),
                "bbox": table.bbox if hasattr(table, 'bbox') else None,
                "page_num": table.page_idx + 1 if hasattr(table, 'page_idx') and table.page_idx >= 0 else None
            }
            tables.append(table_data)
        return tables

def process_documents():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"chunks/llmsherpa/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    processor = PDFProcessor()

    # Process all PDFs in the documents directory
    docs_dir = Path("../documents")
    for pdf_path in tqdm(list(docs_dir.glob("*.pdf")), desc="Processing PDFs"):
        logger.info(f"Processing: {pdf_path}")

        success, result = processor.process_document(pdf_path)

        if success:
            # Save processed content
            output_file = output_dir / f"{pdf_path.stem}_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully processed: {pdf_path}")
        else:
            logger.error(f"Failed to process {pdf_path}: {result}")

    logger.info("Processing complete!")

if __name__ == "__main__":
    process_documents()
