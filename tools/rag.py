import chromadb
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings
from models import ManualSearchResult

# ChromaDB client — stores embeddings locally in chroma_db/ folder
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="bearing_manual")

# Ollama embeddings model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

MANUAL_PATH = "manuals/bearing_manual.pdf"


def load_pdf_chunks(filepath: str, chunk_size: int = 500) -> list:
    """
    Load PDF and split into chunks.
    
    We split by characters because the LLM has a context limit —
    we can't feed the whole PDF at once. Smaller chunks = more
    precise retrieval.
    """
    reader = PdfReader(filepath)
    chunks = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        
        # Split page into chunks of chunk_size characters
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "page": page_num + 1
                })
    
    return chunks


def ingest_manual():
    """
    Load the SKF PDF into ChromaDB.
    Only needs to run once — ChromaDB persists to disk.
    """
    # Check if already ingested
    if collection.count() > 0:
        print(f"Manual already ingested — {collection.count()} chunks in ChromaDB.")
        return
    
    print("Loading PDF...")
    chunks = load_pdf_chunks(MANUAL_PATH)
    print(f"Found {len(chunks)} chunks. Embedding now (this takes a few minutes)...")
    
    # Embed and store each chunk
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk["text"])
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"page": chunk["page"]}]
        )
        if i % 50 == 0:
            print(f"  Embedded {i}/{len(chunks)} chunks...")
    
    print(f"Done — {len(chunks)} chunks stored in ChromaDB.")


def search_manuals(query: str, n_results: int = 3) -> ManualSearchResult:
    """
    Search the SKF manual for relevant sections.
    
    Takes a plain-English query, embeds it, finds the most
    similar chunks in ChromaDB, returns the best match.
    """
    # Embed the query
    query_embedding = embeddings.embed_query(query)
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    if not results["documents"][0]:
        return ManualSearchResult(
            relevant_section="No relevant section found.",
            page_number=None,
            confidence=0.0
        )
    
    # Take the best match
    best_doc = results["documents"][0][0]
    best_meta = results["metadatas"][0][0]
    best_distance = results["distances"][0][0]
    
    # Convert distance to confidence (lower distance = higher confidence)
    confidence = round(1 - best_distance, 4)
    
    return ManualSearchResult(
        relevant_section=best_doc,
        page_number=best_meta.get("page"),
        confidence=confidence
    )


if __name__ == "__main__":
    print("Ingesting SKF bearing manual into ChromaDB...")
    ingest_manual()
    
    print("\nTesting search...")
    result = search_manuals("bearing vibration fault detection")
    print(f"\nPage       : {result.page_number}")
    print(f"Confidence : {result.confidence}")
    print(f"Section    : {result.relevant_section[:300]}...")