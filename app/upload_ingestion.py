from pathlib import Path
import re
import json

import pymupdf
from docx import Document


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):
    """
    Clean extracted document text.
    """

    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# PDF Extraction
# --------------------------------------------------

def extract_pdf(file_path):
    """
    Extract text from every page of a PDF.

    Returns:
        [
            {
                "document": "...",
                "page": 1,
                "text": "..."
            }
        ]
    """

    pages = []

    pdf = pymupdf.open(file_path)

    for page_number, page in enumerate(pdf, start=1):

        text = page.get_text()
        text = clean_text(text)

        if text:
            pages.append(
                {
                    "document": Path(file_path).name,
                    "page": page_number,
                    "text": text,
                }
            )

    pdf.close()

    return pages


# --------------------------------------------------
# DOCX Extraction
# --------------------------------------------------

def extract_docx(file_path):
    """
    Extract text from a DOCX document.

    DOCX documents do not have PDF-style page numbers
    in a simple extraction workflow, so we use page 1
    as the source identifier.
    """

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)
    full_text = clean_text(full_text)

    if not full_text:
        return []

    return [
        {
            "document": Path(file_path).name,
            "page": 1,
            "text": full_text,
        }
    ]


# --------------------------------------------------
# TXT Extraction
# --------------------------------------------------

def extract_txt(file_path):
    """
    Extract text from a TXT file.
    """

    # Try UTF-8 first
    try:

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

    except UnicodeDecodeError:

        # Fallback for Windows-encoded files
        with open(file_path, "r", encoding="cp1252") as file:
            text = file.read()

    text = clean_text(text)

    if not text:
        return []

    return [
        {
            "document": Path(file_path).name,
            "page": 1,
            "text": text,
        }
    ]


# --------------------------------------------------
# Universal Document Extraction
# --------------------------------------------------

def extract_document(file_path):
    """
    Automatically select the correct extractor
    based on the file extension.
    """

    file_path = Path(file_path)

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        return extract_pdf(file_path)

    elif extension == ".docx":

        return extract_docx(file_path)

    elif extension == ".txt":

        return extract_txt(file_path)

    else:

        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Supported formats: PDF, DOCX, TXT."
        )


# --------------------------------------------------
# Chunking
# --------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# Create Chunks
# --------------------------------------------------

def create_chunks_from_pages(pages):
    """
    Convert extracted pages into chunks while
    preserving document and page metadata.
    """

    all_chunks = []

    for page in pages:

        chunks = chunk_text(
            page["text"],
            chunk_size=500,
            overlap=100,
        )

        for chunk in chunks:

            all_chunks.append(
                {
                    "document": page["document"],
                    "page": page["page"],
                    "text": chunk,
                }
            )

    # Assign IDs after all chunks have been created
    for chunk_id, chunk in enumerate(all_chunks, start=1):

        chunk["chunk_id"] = chunk_id

    return all_chunks


# --------------------------------------------------
# Process Uploaded Files
# --------------------------------------------------

def process_uploaded_files(file_paths):
    """
    Process multiple uploaded documents.

    Supported:
        PDF
        DOCX
        TXT

    Returns:
        List of chunks ready for BM25 and FAISS indexing.
    """

    all_pages = []

    for file_path in file_paths:

        file_path = Path(file_path)

        print(f"Processing: {file_path.name}")

        pages = extract_document(file_path)

        print(
            f"  Extracted {len(pages)} page/source sections."
        )

        all_pages.extend(pages)

    chunks = create_chunks_from_pages(all_pages)

    print(f"Created {len(chunks)} chunks.")

    return chunks


# --------------------------------------------------
# Save Chunks
# --------------------------------------------------

def save_chunks(chunks, output_path):
    """
    Save chunks to JSON.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"Chunks saved to: {output_path}")


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    print("\n===================================")
    print("UPLOAD DOCUMENT INGESTION TEST")
    print("===================================\n")

    documents_dir = Path("data/documents")

    files = list(documents_dir.glob("*.pdf"))

    if not files:

        print("No documents found.")

    else:

        chunks = process_uploaded_files(files)

        print(
            f"\nTotal chunks created: {len(chunks)}"
        )

        print("\nSample chunk:")

        print("-" * 70)

        if chunks:

            sample = chunks[0]

            print(
                f"Document : {sample['document']}"
            )

            print(
                f"Page     : {sample['page']}"
            )

            print(
                f"Chunk ID : {sample['chunk_id']}"
            )

            print(
                f"Text     : {sample['text'][:300]}"
            )

        print("-" * 70)