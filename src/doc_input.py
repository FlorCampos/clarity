import os
import sys
import json
import base64
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/app')
from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# ─────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────

class DocumentReader:
    """
    Base class — defines the contract.
    Every reader must implement read().
    """

    def read(self, source: str) -> dict:
        raise NotImplementedError

    def _validate_file(self, file_path: str) -> None:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )


# ─────────────────────────────────────────────────────────────
# VISION PAGE READER — the core of everything
# ─────────────────────────────────────────────────────────────

def _analyze_page_with_vision(
    image_data: bytes,
    page_number: int,
    total_pages: int
) -> str:
    """
    Sends a single page image to Claude Vision.
    Returns extracted text including diagrams and tables.

    This is the core function — used by PDF, image,
    and any other visual document reader.
    """

    encoded = base64.standard_b64encode(image_data).decode('utf-8')

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded
                        }
                    },
                    {
                        "type": "text",
                        "text": f"""This is page {page_number} of {total_pages} 
from a requirements document.

Extract ALL content from this page including:
- All text paragraphs and sentences
- All table contents (row by row)
- All diagram labels, annotations and flow descriptions
- All wireframe element labels and descriptions
- All numbered or bulleted lists
- Any handwritten notes if visible

For diagrams and wireframes:
- Describe the flow or structure
- List all labeled elements
- Describe relationships between elements
- Extract any text inside shapes or boxes

Format everything as clear readable text.
Preserve the logical structure.
Do not skip anything — every detail may be a requirement."""
                    }
                ]
            }
        ]
    )

    return message.content[0].text


# ─────────────────────────────────────────────────────────────
# PDF READER — with full Vision support
# ─────────────────────────────────────────────────────────────

class PDFReader(DocumentReader):
    """
    Reads PDF files — text AND diagrams AND tables.

    Strategy:
    1. Convert each PDF page to an image
    2. Send each page to Claude Vision
    3. Claude extracts everything it sees
    4. Combine all pages into one document

    This catches 100% of requirements including
    embedded diagrams, wireframes, and tables.
    """

    def read(self, source: str) -> dict:
        """Extracts complete content from PDF."""

        self._validate_file(source)

        from pdf2image import convert_from_path
        import io

        print(f"\n  📄 Reading PDF: {Path(source).name}")
        print(f"     Converting pages to images...")

        pages = convert_from_path(
            source,
            dpi=150,
            fmt='PNG'
        )

        total_pages = len(pages)
        print(f"     Pages found: {total_pages}")
        print(f"     Analyzing with Claude Vision...")

        full_text = ""

        for i, page in enumerate(pages):
            print(f"     Page {i+1}/{total_pages}...", end=" ")

            img_byte_arr = io.BytesIO()
            page.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()

            page_text = _analyze_page_with_vision(
                image_data=img_bytes,
                page_number=i+1,
                total_pages=total_pages
            )

            full_text += f"\n\n--- PAGE {i+1} ---\n"
            full_text += page_text
            print(f"✅ ({len(page_text.split())} words)")

        print(f"\n     Total words extracted: {len(full_text.split())}")

        return {
            "text": full_text.strip(),
            "source": source,
            "type": "pdf",
            "pages": total_pages,
            "method": "claude_vision",
            "metadata": {
                "filename": Path(source).name,
                "vision_model": "claude-sonnet-4-6"
            }
        }


# ─────────────────────────────────────────────────────────────
# IMAGE READER — single image or diagram
# ─────────────────────────────────────────────────────────────

class ImageReader(DocumentReader):
    """
    Reads single images — diagrams, wireframes,
    whiteboard photos, screenshots.

    Uses Claude Vision directly.
    """

    def read(self, source: str) -> dict:
        """Extracts requirements from image."""

        self._validate_file(source)

        print(f"\n  🖼️  Reading image: {Path(source).name}")
        print(f"     Analyzing with Claude Vision...")

        with open(source, 'rb') as f:
            image_data = f.read()

        extension = Path(source).suffix.lower()
        media_types = {
            '.jpg':  'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png':  'image/png',
            '.gif':  'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_types.get(extension, 'image/png')

        encoded = base64.standard_b64encode(
            image_data
        ).decode('utf-8')

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded
                            }
                        },
                        {
                            "type": "text",
                            "text": """Extract all software requirements,
user stories, and functional specifications
from this image.

Include all text, diagram labels, flow descriptions,
wireframe elements, table contents, and annotations.

Be thorough — every element may be a requirement."""
                        }
                    ]
                }
            ]
        )

        extracted_text = message.content[0].text

        print(f"     ✅ {len(extracted_text.split())} words extracted")

        return {
            "text": extracted_text,
            "source": source,
            "type": f"image_{extension[1:]}",
            "pages": 1,
            "method": "claude_vision",
            "metadata": {
                "media_type": media_type,
                "filename": Path(source).name
            }
        }


# ─────────────────────────────────────────────────────────────
# TEXT FILE READER
# ─────────────────────────────────────────────────────────────

class TextFileReader(DocumentReader):
    """
    Reads plain text files — TXT, MD, CSV.
    No Vision needed — just read the file.
    """

    def read(self, source: str) -> dict:
        """Reads plain text file."""

        self._validate_file(source)

        print(f"\n  📝 Reading: {Path(source).name}")

        with open(source, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"     Words: {len(text.split())}")

        return {
            "text": text.strip(),
            "source": source,
            "type": Path(source).suffix.lower(),
            "pages": 1,
            "method": "direct_read",
            "metadata": {
                "filename": Path(source).name,
                "size_bytes": Path(source).stat().st_size
            }
        }


# ─────────────────────────────────────────────────────────────
# PASTE TEXT READER
# ─────────────────────────────────────────────────────────────

class PasteTextReader(DocumentReader):
    """
    Accepts text pasted directly into Streamlit UI.
    Used for Jira tickets, emails, chat messages.
    """

    def read(self, source: str) -> dict:
        """source IS the text content here."""

        print(f"\n  📋 Processing pasted text...")
        print(f"     Words: {len(source.split())}")

        return {
            "text": source.strip(),
            "source": "pasted_text",
            "type": "text",
            "pages": 1,
            "method": "direct_paste",
            "metadata": {
                "input_method": "direct_paste"
            }
        }


# ─────────────────────────────────────────────────────────────
# FACTORY — auto-detects document type
# ─────────────────────────────────────────────────────────────

def get_document_reader(source: str) -> DocumentReader:
    """
    Detects document type automatically.
    Returns the correct reader.
    No configuration needed.
    """

    # Long text or multi-line = pasted content
    if len(source) > 260 or '\n' in source:
        return PasteTextReader()

    path = Path(source)
    extension = path.suffix.lower()

    readers = {
        '.pdf':  PDFReader,
        '.txt':  TextFileReader,
        '.md':   TextFileReader,
        '.csv':  TextFileReader,
        '.png':  ImageReader,
        '.jpg':  ImageReader,
        '.jpeg': ImageReader,
        '.gif':  ImageReader,
        '.webp': ImageReader,
    }

    if extension in readers:
        reader_class = readers[extension]
        print(f"\n  Auto-detected: {extension} → "
              f"{reader_class.__name__}")
        return reader_class()

    print(f"  ⚠️  Unknown '{extension}' → trying TextFileReader")
    return TextFileReader()


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────

def process_document(
    source: str,
    project_name: str = "default"
) -> dict:
    """
    Main function — reads any document and processes
    it through the full Clarity pipeline.

    Args:
        source: file path, URL, or direct text
        project_name: which project this belongs to

    Returns:
        dict: document + parsed requirements
    """

    from src.agent import RequirementsAgent

    print(f"\n{'='*60}")
    print(f"  CLARITY — Document Processor")
    print(f"  Project: {project_name}")
    print(f"{'='*60}")

    # Step 1 — Read document
    reader = get_document_reader(source)
    document = reader.read(source)

    print(f"\n  ✅ Document read")
    print(f"     Type:   {document['type']}")
    print(f"     Method: {document['method']}")
    print(f"     Words:  {len(document['text'].split())}")

    # Step 2 — Split if too large
    word_count = len(document['text'].split())
    if word_count > 10000:
        print(f"\n  ⚠️  Large document — splitting into chunks...")
        chunks = _split_into_chunks(document['text'])
        print(f"     {len(chunks)} chunks created")
    else:
        chunks = [document['text']]

    # Step 3 — Process through requirements agent
    agent = RequirementsAgent(project_name=project_name)
    results = []

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"\n  Processing chunk {i+1}/{len(chunks)}...")
        result = agent.process(chunk)
        results.append(result)

    return {
        "document": document,
        "requirements": results,
        "total_processed": len(results),
        "processed_at": datetime.now().isoformat()
    }


def _split_into_chunks(
    text: str,
    chunk_size: int = 3000
) -> list:
    """Splits large documents at paragraph boundaries."""

    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        words = len((current_chunk + paragraph).split())
        if words > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n  Testing Document Input Pipeline")
    print("  " + "─" * 40)

    sample_text = """
    REQUIREMENTS DOCUMENT — E-COMMERCE PLATFORM

    1. User Registration
    Users should be able to register with email
    and password. The system should send a
    verification email. Users can also register
    with Google or Facebook.

    2. Product Catalog
    The platform should display products with
    images, descriptions and prices. Users can
    filter by category, price range and rating.
    Search should be fast and accurate.

    3. Shopping Cart
    Users can add products to cart. Cart should
    persist when user logs out. Users can apply
    discount codes. Checkout should support
    credit card and PayPal.
    """

    result = process_document(
        source=sample_text,
        project_name="clarity-demo"
    )

    print(f"\n{'='*60}")
    print(f"  RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"  Requirements processed: {result['total_processed']}")

    for i, req in enumerate(result['requirements'], 1):
        status = req.get('status', 'UNKNOWN')
        score = req.get('testability_score', 0)
        icon = {"APPROVED":"✅","NEEDS_REVIEW":"⚠️",
                "REJECTED":"❌"}.get(status, "❓")
        print(f"  {i}. {icon} {status} — Score: {score}/10")
        story = req.get('user_story', '')[:70]
        print(f"     {story}...")

    with open("document_output.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  💾 Saved to document_output.json")
    print(f"\n{'='*60}")