from pathlib import Path
import pymupdf
import re
import json


DOCUMENTS_DIR = Path("data/documents")


def clean_text(text):
    """
    Clean extracted PDF text.
    """

    # Replace multiple spaces with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def extract_pages_from_pdf(pdf_path):
    """
    Extract text from every page of a PDF.

    Returns a list containing:
    - document name
    - page number
    - page text
    """

    pages = []

    pdf = pymupdf.open(pdf_path)

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text()

        text = clean_text(text)

        if text:
            pages.append(
                {
                    "document": pdf_path.name,
                    "page": page_number,
                    "text": text,
                }
            )

    pdf.close()

    return pages


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.

    chunk_size:
        Maximum approximate number of words per chunk.

    overlap:
        Number of words shared between neighboring chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks_from_pages(pages):
    """
    Convert extracted pages into smaller chunks
    while preserving document and page metadata.
    """

    all_chunks = []

    chunk_id = 0

    for page in pages:

        chunks = chunk_text(
            page["text"],
            chunk_size=500,
            overlap=100,
        )

        for chunk in chunks:

            chunk_id += 1

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document": page["document"],
                    "page": page["page"],
                    "text": chunk,
                }
            )

    return all_chunks


def load_all_documents():
    """
    Load all PDFs from data/documents/.
    """

    all_pages = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}")

        pages = extract_pages_from_pdf(pdf_path)

        all_pages.extend(pages)

    return all_pages


if __name__ == "__main__":

    print("\n===================================")
    print("ENTERPRISE RAG - DOCUMENT INGESTION")
    print("===================================\n")

    pages = load_all_documents()

    print(f"\nTotal pages extracted: {len(pages)}")

    chunks = create_chunks_from_pages(pages)

    print(f"Total chunks created: {len(chunks)}")

    # Save chunks for the retrieval pipeline
    output_path = Path("data/chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nChunks saved to: {output_path}")

    print("\n===================================")
    print("SAMPLE CHUNKS")
    print("===================================\n")

    for chunk in chunks[:5]:

        print(f"Chunk ID : {chunk['chunk_id']}")
        print(f"Document : {chunk['document']}")
        print(f"Page     : {chunk['page']}")
        print(f"Text     : {chunk['text'][:300]}")

        print("\n" + "-" * 70 + "\n")