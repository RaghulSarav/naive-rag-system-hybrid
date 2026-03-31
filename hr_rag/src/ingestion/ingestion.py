import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.db import get_vector_store

load_dotenv()


def ingest_pdf(file_path: str, collection_name: str = "hr_support_desk") -> None:
    """
    Ingest a PDF file into the PGVector store.

    Steps:
      1. Load PDF pages via PyPDFLoader
      2. Enrich each page's metadata (source, page, document_name, category)
      3. Split pages into overlapping chunks
      4. Compute embeddings and store in PGVector
    """
    print(f"[Ingestion] Loading PDF: {file_path}")

    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"[Ingestion] Loaded {len(docs)} page(s).")

    # 2. Metadata enrichment — attach document_name and page info to each chunk
    document_name = os.path.basename(file_path)
    for doc in docs:
        doc.metadata.update(
            {
                "source": file_path,
                "document_name": document_name,
                "document_extension": "pdf",
                "page": doc.metadata.get("page", 0),
                "category": collection_name,
                "last_updated": os.path.getmtime(file_path),
            }
        )

    # 3. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"[Ingestion] Created {len(chunks)} chunk(s).")

    # 4+5. Compute embeddings and store via PGVector
    vector_store = get_vector_store(collection_name=collection_name)
    vector_store.add_documents(chunks)
    print("[Ingestion] Ingestion completed successfully!")


if __name__ == "__main__":
    # Run directly: $env:PYTHONPATH="."; python src/ingestion/ingestion.py
    ingest_pdf("data/HR_Support_Desk_KnowledgeBase.pdf")
