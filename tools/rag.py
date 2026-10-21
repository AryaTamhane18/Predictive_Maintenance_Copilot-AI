import os
import re
import sys
import json
import math
import hashlib
import collections
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import requests

try:
    import pdfplumber
    HAVE_PLUMBER = True
except ImportError:                       # tables are skipped without it
    HAVE_PLUMBER = False
    from pypdf import PdfReader

# ── Config ───────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR    = os.path.join(ROOT, "manuals")
CHROMA_DIR  = os.path.join(ROOT, "chroma_db")
REGISTRY    = os.path.join(CHROMA_DIR, "ingest_registry.json")
TABLE_STORE = os.path.join(CHROMA_DIR, "tables.json")
COLLECTION  = "documents"

EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL  = "http://localhost:11434/api/embeddings"
VISION_URL  = "http://localhost:11434/api/generate"
VISION_MODEL = "llava"

CHUNK_SIZE    = 900
CHUNK_OVERLAP = 150
MIN_CHUNK     = 80

FURNITURE_RATIO = 0.05     # a line on 5% of pages is running furniture
TABLE_MIN_ROWS  = 4        # fewer rows than this is not worth treating as a table
TABLE_MIN_COLS  = 3
TABLE_DENSITY   = 5        # lines with 3+ numbers before a page is inspected
ROWS_PER_CHUNK  = 6        # table rows grouped per chunk

RRF_K = 60

TABLE_SETTINGS = {
    "vertical_strategy":   "text",   # these tables are aligned by
    "horizontal_strategy": "text",   # whitespace, not ruled lines
    "text_tolerance":      2
}

NUM_RE     = re.compile(r"^[+\-–]?\d+([.,]\d+)?$")
HEADING_RE = re.compile(r"^(\d+(?:\.\d+){1,3})\s+([A-Z][^.]{3,80})$")
FIGURE_RE  = re.compile(r"^(fig\.?|figure|diagram|table|chart)\s*(\d+)",
                        re.IGNORECASE)


# ── Embeddings ───────────────────────────────────────────────────────

def get_embedding(text: str) -> list:
    """Embed one piece of text with the local Ollama model."""
    r = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text},
                      timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]


def describe_image(png_bytes: bytes, context: str = "") -> str:
    """
    Ask the local vision model what a figure shows, so a page that is
    mostly a drawing becomes searchable text instead of nothing.
    """
    import base64
    prompt = (
        "This is a figure from a technical maintenance manual. "
        "Describe in two or three sentences what it shows: the "
        "components, the measurement or procedure, and any labels you "
        "can read. Be factual; do not speculate."
    )
    if context:
        prompt += f"\n\nNearby caption text: {context[:300]}"

    r = requests.post(VISION_URL, json={
        "model":  VISION_MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(png_bytes).decode()],
        "stream": False,
        "options": {"temperature": 0}
    }, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "").strip()


# ── Page structure detection ─────────────────────────────────────────

def detect_furniture(pages: list) -> tuple:
    """
    Find repeated header and footer lines by counting how often each
    line appears in that position. Frequency works on any document;
    assuming "the first line is the chapter" does not.
    """
    n = sum(1 for p in pages if p)
    if n == 0:
        return set(), set()

    first, last = collections.Counter(), collections.Counter()
    for lines in pages:
        if not lines:
            continue
        if not lines[0].isdigit():
            first[lines[0]] += 1
        if len(lines) > 1 and not lines[-1].isdigit():
            last[lines[-1]] += 1

    threshold = max(2, int(n * FURNITURE_RATIO))
    return ({t for t, c in first.items() if c >= threshold},
            {t for t, c in last.items()  if c >= threshold})


def detect_page_offset(pages: list) -> int:
    """
    Difference between the PDF page index and the number printed on
    the page. Returns None when pages do not agree, so a citation
    never shows a page number that was guessed.
    """
    offsets = []
    for i, lines in enumerate(pages):
        if lines and lines[-1].isdigit():
            printed = int(lines[-1])
            if 0 < printed <= len(pages):
                offsets.append((i + 1) - printed)

    if len(offsets) < 5:
        return None
    best, count = collections.Counter(offsets).most_common(1)[0]
    return best if count >= len(offsets) * 0.6 else None


def find_heading(lines: list) -> str:
    """Spot a numbered heading such as '4.2.1 Torque specification'."""
    for line in lines[:6]:
        m = HEADING_RE.match(line)
        if m:
            return f"{m.group(1)} {m.group(2)}".strip()
    return ""


def is_number(cell) -> bool:
    cell = (cell or "").strip().replace(" ", "")
    return bool(cell) and bool(NUM_RE.match(cell))


# ── Tables ───────────────────────────────────────────────────────────

def looks_tabular(lines: list) -> bool:
    """
    Cheap test for whether a page is worth inspecting for tables:
    several lines each carrying three or more numbers. Running the
    full table finder on every page of a long document is slow, and
    on prose pages it finds nothing anyway.
    """
    dense = sum(
        1 for l in lines
        if len([w for w in l.split() if NUM_RE.match(w.replace(" ", ""))]) >= 3
    )
    return dense >= TABLE_DENSITY


def extract_tables(page, lines: list) -> list:
    """
    Pull data tables out of a page.

    Rows are kept exactly as printed and never split, which is the
    whole point: a row is one record and cutting it destroys the link
    between a component ID and its values.

    The printed header block above the table is kept verbatim as
    context. Mapping each value to its own column name is NOT done:
    these tables are whitespace-aligned with multi-line headers that
    span columns, and a wrong column name would be worse than none.
    """
    tables = []
    try:
        found = page.find_tables(table_settings=TABLE_SETTINGS)
    except Exception:
        return tables

    for t_index, table in enumerate(found):
        try:
            raw = table.extract()
        except Exception:
            continue

        rows = [r for r in raw if any((c or "").strip() for c in r)]
        if len(rows) < TABLE_MIN_ROWS or len(rows[0]) < TABLE_MIN_COLS:
            continue

        # a data row carries several filled cells, at least one numeric
        data = [r for r in rows
                if sum(1 for c in r if (c or "").strip()) >= 3
                and any(is_number(c) for c in r)]
        if len(data) < 3:
            continue

        # the header block is whatever text sits above the table
        try:
            top = table.bbox[1]
            header_lines = [l for l in lines
                            if l.strip() and not l.strip().isdigit()][:6]
        except Exception:
            header_lines = lines[:4]

        tables.append({
            "index":   t_index,
            "context": [l for l in header_lines][:5],
            "rows":    [[(c or "").strip() for c in r] for r in data],
            "n_cols":  len(rows[0])
        })

    return tables


def table_chunks(table: dict, meta_base: dict) -> list:
    """
    Turn one table into retrievable pieces.

    Two kinds, because two kinds of question get asked:

      a summary chunk  — answers "what does this table cover?"
      row-group chunks — answer "what is the value for part X?"

    Each row-group repeats the header context, so a chunk retrieved in
    isolation still says what its numbers mean.
    """
    context = "\n".join(table["context"]).strip()
    chunks  = []

    preview = "\n".join(" | ".join(r) for r in table["rows"][:4])
    chunks.append({
        "text": (f"TABLE on {meta_base['source']} "
                 f"page {meta_base['printed_page']}\n"
                 f"{context}\n\n"
                 f"{len(table['rows'])} rows, {table['n_cols']} columns. "
                 f"First rows:\n{preview}"),
        "kind": "table_summary"
    })

    for start in range(0, len(table["rows"]), ROWS_PER_CHUNK):
        group = table["rows"][start:start + ROWS_PER_CHUNK]
        body  = "\n".join(" | ".join(r) for r in group)
        chunks.append({
            "text": (f"TABLE rows (as printed)\n{context}\n\n{body}"),
            "kind": "table_rows"
        })

    return chunks


# ── Figures ──────────────────────────────────────────────────────────

def figure_caption(lines: list) -> str:
    """Find a caption such as 'Diagram 2 — Estimation of ...'."""
    for i, line in enumerate(lines):
        if FIGURE_RE.match(line.strip()):
            tail = lines[i + 1] if i + 1 < len(lines) else ""
            return f"{line.strip()} {tail.strip()}".strip()
    return ""


def render_page_png(path: str, page_no: int, zoom: float = 1.6):
    """Render one page to PNG bytes, for vision captioning."""
    try:
        import fitz                       # PyMuPDF
    except ImportError:
        return None
    try:
        doc  = fitz.open(path)
        page = doc[page_no - 1]
        pix  = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        data = pix.tobytes("png")
        doc.close()
        return data
    except Exception:
        return None


# ── Chunking prose ───────────────────────────────────────────────────

def chunk_text(text: str,
               size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list:
    """Split prose into overlapping pieces, cutting at sentence ends."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text] if len(text) >= MIN_CHUNK else []

    chunks, start = [], 0
    while start < len(text):
        end   = min(start + size, len(text))
        piece = text[start:end]

        if end < len(text):
            cut = max(piece.rfind("\n"), piece.rfind(". "))
            if cut > size * 0.5:
                piece = piece[:cut + 1]
                end   = start + cut + 1

        piece = piece.strip()
        if len(piece) >= MIN_CHUNK:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


# ── Building chunks for one document ─────────────────────────────────

def build_chunks(path: str, caption_figures: bool = False,
                 max_figures: int = 25) -> tuple:
    """
    Parse one PDF into chunks plus a structured table store.

    Returns (chunks, tables) where tables is a list of records kept
    for exact lookup outside the vector search.
    """
    source = os.path.basename(path)
    doc_id = os.path.splitext(source)[0]

    if not HAVE_PLUMBER:
        return _build_chunks_textonly(path, source, doc_id), []

    chunks, table_store = [], []
    figures_done = 0

    with pdfplumber.open(path) as pdf:
        page_lines = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            page_lines.append([l.strip() for l in text.split("\n") if l.strip()])

        headers, footers = detect_furniture(page_lines)
        offset           = detect_page_offset(page_lines)

        n_tables = n_figs = 0
        section  = ""

        for i, page in enumerate(pdf.pages):
            pdf_page = i + 1
            lines    = page_lines[i]
            printed  = (pdf_page - offset) if offset is not None else -1

            body = [l for l in lines
                    if l not in headers and l not in footers
                    and not l.isdigit()]

            running = lines[0] if lines and lines[0] in headers else ""
            heading = find_heading(body)
            if heading:
                section = heading
            elif running:
                section = running

            meta_base = {
                "doc_id":       doc_id,
                "source":       source,
                "section":      section or "",
                "pdf_page":     pdf_page,
                "printed_page": printed
            }

            # ── tables ────────────────────────────────────────────
            page_tables = []
            if body and looks_tabular(body):
                page_tables = extract_tables(page, body)

            for table in page_tables:
                n_tables += 1
                table_id = f"{doc_id}_p{pdf_page}_t{table['index']}"
                table_store.append({
                    "table_id":     table_id,
                    "doc_id":       doc_id,
                    "source":       source,
                    "pdf_page":     pdf_page,
                    "printed_page": printed,
                    "section":      section,
                    "context":      table["context"],
                    "rows":         table["rows"]
                })
                for j, piece in enumerate(table_chunks(table, meta_base)):
                    chunks.append({
                        "id":   f"{table_id}_c{j}",
                        "text": piece["text"],
                        "meta": {**meta_base,
                                 "content_type": piece["kind"],
                                 "table_id":     table_id,
                                 "chunk":        j}
                    })

            # ── prose ─────────────────────────────────────────────
            # On a table page the numbers are already captured above,
            # so only the non-numeric lines go to prose. This stops
            # loose digits polluting the text index.
            if page_tables:
                prose_lines = [l for l in body
                               if len([w for w in l.split()
                                       if NUM_RE.match(w.replace(" ", ""))]) < 3]
            else:
                prose_lines = body

            prose = "\n".join(prose_lines).strip()
            for j, piece in enumerate(chunk_text(prose)):
                chunks.append({
                    "id":   f"{doc_id}_p{pdf_page}_c{j}",
                    "text": piece,
                    "meta": {**meta_base, "content_type": "prose", "chunk": j}
                })

            # ── figures ───────────────────────────────────────────
            # A page that is mostly a drawing yields almost no text.
            # Previously those pages were dropped silently; now they
            # are recorded, and optionally described by the vision
            # model so they can be found.
            if len(prose) < MIN_CHUNK and not page_tables:
                caption = figure_caption(lines)
                described = ""
                if caption_figures and figures_done < max_figures:
                    png = render_page_png(path, pdf_page)
                    if png:
                        try:
                            described = describe_image(png, caption)
                            figures_done += 1
                        except Exception:
                            described = ""

                if caption or described:
                    n_figs += 1
                    text = (f"FIGURE on {source} page "
                            f"{printed if printed > 0 else pdf_page}\n"
                            f"{caption}\n{described}").strip()
                    chunks.append({
                        "id":   f"{doc_id}_p{pdf_page}_fig",
                        "text": text,
                        "meta": {**meta_base,
                                 "content_type": "figure",
                                 "chunk":        0}
                    })

        print(f"    page offset: {offset if offset is not None else 'none'} | "
              f"furniture lines: {len(headers) + len(footers)}")
        print(f"    {n_tables} tables, {n_figs} figures, "
              f"{sum(1 for c in chunks if c['meta']['content_type'] == 'prose')}"
              f" prose chunks")

    for c in chunks:
        c["meta"]["ingested_at"] = datetime.now().isoformat(timespec="seconds")

    return chunks, table_store


def _build_chunks_textonly(path, source, doc_id):
    """Fallback when pdfplumber is not installed: prose only."""
    reader = PdfReader(path)
    pages  = [[l.strip() for l in (p.extract_text() or "").split("\n") if l.strip()]
              for p in reader.pages]
    headers, footers = detect_furniture(pages)
    offset = detect_page_offset(pages)

    out, section = [], ""
    for i, lines in enumerate(pages):
        pdf_page = i + 1
        body = [l for l in lines if l not in headers and l not in footers
                and not l.isdigit()]
        if not body:
            continue
        heading = find_heading(body)
        if heading:
            section = heading
        elif lines and lines[0] in headers:
            section = lines[0]

        for j, piece in enumerate(chunk_text("\n".join(body))):
            out.append({
                "id": f"{doc_id}_p{pdf_page}_c{j}",
                "text": piece,
                "meta": {"doc_id": doc_id, "source": source,
                         "section": section or "", "pdf_page": pdf_page,
                         "printed_page": (pdf_page - offset) if offset else -1,
                         "content_type": "prose", "chunk": j}
            })
    return out


# ── Registry and table store ─────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Collection ───────────────────────────────────────────────────────

def _collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name     = COLLECTION,
        metadata = {"hnsw:space": "cosine"}
    )


def ingest_documents(force: bool = False,
                     caption_figures: bool = False) -> None:
    """Ingest every PDF in the documents folder, skipping unchanged ones."""
    collection = _collection()
    registry   = {} if force else _load_json(REGISTRY, {})
    all_tables = [] if force else _load_json(TABLE_STORE, [])

    files = sorted(f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf"))
    if not files:
        print(f"No PDFs found in {DOCS_DIR}")
        return
    if not HAVE_PLUMBER:
        print("pdfplumber not installed — tables will not be extracted.\n"
              "  pip install pdfplumber\n")

    print(f"Found {len(files)} document(s)\n")

    for name in files:
        path   = os.path.join(DOCS_DIR, name)
        digest = _file_hash(path)
        doc_id = os.path.splitext(name)[0]

        if registry.get(doc_id, {}).get("hash") == digest and not force:
            print(f"  {name}: unchanged, skipping")
            continue

        print(f"  {name}: parsing ...")
        chunks, tables = build_chunks(path, caption_figures=caption_figures)
        total = len(chunks)
        if total == 0:
            print("    no extractable text — is it a scan? skipping")
            continue

        try:
            collection.delete(where={"doc_id": doc_id})
        except Exception:
            pass
        all_tables = [t for t in all_tables if t["doc_id"] != doc_id] + tables

        print(f"    {total} chunks — embedding ...")
        batch = 50
        for i in range(0, total, batch):
            group = chunks[i:i + batch]
            collection.add(
                ids        = [c["id"] for c in group],
                embeddings = [get_embedding(c["text"]) for c in group],
                documents  = [c["text"] for c in group],
                metadatas  = [c["meta"] for c in group]
            )
            print(f"      {min(i + batch, total)}/{total}")

        registry[doc_id] = {"hash": digest, "source": name,
                            "chunks": total, "tables": len(tables),
                            "ingested": datetime.now().isoformat(timespec="seconds")}
        _save_json(REGISTRY, registry)
        _save_json(TABLE_STORE, all_tables)
        print("    done\n")

    print(f"Collection ready — {collection.count()} chunks, "
          f"{len(all_tables)} tables stored")


# ── Exact table lookup, outside the vector search ────────────────────

def lookup_table_rows(term: str, limit: int = 10) -> list:
    """
    Find table rows containing an exact string, such as a bearing
    designation. This does not involve embeddings or a language model
    at all: a value question is answered from the stored record.
    """
    tables = _load_json(TABLE_STORE, [])
    needle = term.strip().lower()
    out    = []

    for table in tables:
        for row in table["rows"]:
            if any(needle in (cell or "").lower() for cell in row):
                out.append({
                    "row":          row,
                    "context":      table["context"],
                    "source":       table["source"],
                    "printed_page": table["printed_page"],
                    "pdf_page":     table["pdf_page"],
                    "section":      table["section"],
                    "table_id":     table["table_id"]
                })
                if len(out) >= limit:
                    return out
    return out


# ── Keyword index (BM25) ─────────────────────────────────────────────

_bm25_cache = None
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/.]*")


def tokenize(text: str) -> list:
    """Keep digits, hyphens and slashes so 6205-2RS stays one term."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _build_bm25():
    global _bm25_cache
    data  = _collection().get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]

    tokenized = [tokenize(d) for d in docs]
    df = collections.Counter()
    for toks in tokenized:
        df.update(set(toks))

    n       = len(tokenized)
    avg_len = (sum(len(t) for t in tokenized) / n) if n else 0
    idf     = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    _bm25_cache = {"docs": docs, "metas": metas,
                   "tf": [collections.Counter(t) for t in tokenized],
                   "len": [len(t) for t in tokenized],
                   "avg_len": avg_len, "idf": idf, "n": n}
    return _bm25_cache


def bm25_search(query: str, top_k: int = 10) -> list:
    idx = _bm25_cache or _build_bm25()
    if idx["n"] == 0:
        return []

    k1, b  = 1.5, 0.75
    scores = collections.defaultdict(float)
    for term in tokenize(query):
        if term not in idx["idf"]:
            continue
        weight = idx["idf"][term]
        for i, tf in enumerate(idx["tf"]):
            f = tf.get(term, 0)
            if not f:
                continue
            norm = 1 - b + b * (idx["len"][i] / idx["avg_len"])
            scores[i] += weight * (f * (k1 + 1)) / (f + k1 * norm)

    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


# ── Search ───────────────────────────────────────────────────────────

def _citation(meta: dict) -> str:
    """Build a citation an engineer can look up, from metadata alone."""
    parts = [meta.get("source", "document")]
    if meta.get("section"):
        parts.append(meta["section"])

    kind = meta.get("content_type", "prose")
    label = {"table_rows": "table", "table_summary": "table",
             "figure": "figure"}.get(kind)
    printed = meta.get("printed_page", -1)
    page = (f"page {printed} (PDF page {meta.get('pdf_page')})"
            if printed and printed > 0 else f"PDF page {meta.get('pdf_page')}")
    parts.append(f"{label} on {page}" if label else page)
    return " — ".join(parts)


def _key(meta: dict) -> str:
    return (f"{meta.get('doc_id')}|{meta.get('pdf_page')}|"
            f"{meta.get('content_type')}|{meta.get('chunk')}")


def search_manuals(query: str,
                   top_k: int = 3,
                   mode: str = "hybrid",
                   doc_id: str = None,
                   content_type: str = None) -> dict:
    """
    Search the document library.

    mode: "hybrid" (default), "vector" or "keyword". Hybrid matters
    because semantic search is good at meaning and poor at exact
    strings such as component IDs.

    content_type: restrict to "prose", "table_rows" or "figure".
    """
    collection = _collection()
    if collection.count() == 0:
        return {"found": False, "query": query, "mode": mode,
                "excerpt": "No documents ingested. Run ingest_documents().",
                "citation": "none", "source": "none", "similarity": 0.0,
                "results": [], "all_excerpts": []}

    clauses = []
    if doc_id:
        clauses.append({"doc_id": doc_id})
    if content_type:
        clauses.append({"content_type": content_type})
    where = clauses[0] if len(clauses) == 1 else (
        {"$and": clauses} if clauses else None)

    pool   = max(top_k * 4, 10)
    ranked = {}

    if mode in ("hybrid", "vector"):
        res = collection.query(
            query_embeddings = [get_embedding(query)],
            n_results        = min(pool, collection.count()),
            where            = where
        )
        for rank, (doc, meta, dist) in enumerate(zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0])):
            ranked[_key(meta)] = {"text": doc, "meta": meta,
                                  "vector_rank": rank + 1, "keyword_rank": None,
                                  "similarity": round(1 - dist, 4)}

    if mode in ("hybrid", "keyword"):
        idx = _bm25_cache or _build_bm25()
        for rank, (i, _s) in enumerate(bm25_search(query, pool)):
            meta = idx["metas"][i]
            if doc_id and meta.get("doc_id") != doc_id:
                continue
            if content_type and meta.get("content_type") != content_type:
                continue
            k = _key(meta)
            if k in ranked:
                ranked[k]["keyword_rank"] = rank + 1
            else:
                ranked[k] = {"text": idx["docs"][i], "meta": meta,
                             "vector_rank": None, "keyword_rank": rank + 1,
                             "similarity": None}

    if not ranked:
        return {"found": False, "query": query, "mode": mode,
                "excerpt": "No relevant content found.", "citation": "none",
                "source": "none", "similarity": 0.0,
                "results": [], "all_excerpts": []}

    def fused(entry):
        score = 0.0
        if entry["vector_rank"]:
            score += 1 / (RRF_K + entry["vector_rank"])
        if entry["keyword_rank"]:
            score += 1 / (RRF_K + entry["keyword_rank"])
        return score

    best = sorted(ranked.values(), key=fused, reverse=True)[:top_k]

    hits = []
    for entry in best:
        meta = entry["meta"]
        hits.append({
            "text":         entry["text"],
            "citation":     _citation(meta),
            "content_type": meta.get("content_type", "prose"),
            "doc_id":       meta.get("doc_id"),
            "source":       meta.get("source"),
            "section":      meta.get("section", ""),
            "pdf_page":     meta.get("pdf_page"),
            "printed_page": meta.get("printed_page"),
            "chunk":        meta.get("chunk", -1),
            "similarity":   entry["similarity"],
            "vector_rank":  entry["vector_rank"],
            "keyword_rank": entry["keyword_rank"],
            "fusion_score": round(fused(entry), 5)
        })

    return {"found": True, "query": query, "mode": mode,
            "excerpt": hits[0]["text"], "citation": hits[0]["citation"],
            "source": hits[0]["source"], "similarity": hits[0]["similarity"],
            "results": hits, "all_excerpts": [h["text"] for h in hits]}


def format_for_prompt(result: dict) -> str:
    """
    Lay the passages out for an agent's prompt, each labelled with its
    citation, so the model can only refer to sources that exist.
    """
    if not result.get("found"):
        return "No relevant manual section found."
    parts = []
    for i, hit in enumerate(result["results"], 1):
        parts.append(f"[Source {i}] {hit['citation']}\n{hit['text']}")
    return "\n\n".join(parts)


def load_manuals_to_chromadb(force: bool = False) -> None:
    ingest_documents(force=force)


# ── Run ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 68)
    print("  INGESTION")
    print("=" * 68)
    ingest_documents(force="--force" in sys.argv,
                     caption_figures="--figures" in sys.argv)

    print("\n" + "=" * 68)
    print("  TEST SEARCHES")
    print("=" * 68)
    for q in ["What causes abnormal vibration in a bearing?",
              "radial internal clearance values",
              "1210 EKTN9"]:
        print(f"\nQuery: {q}")
        for hit in search_manuals(q).get("results", []):
            sim = f"{hit['similarity']:.3f}" if hit["similarity"] else "  -  "
            print(f"  [{hit['content_type']:<13}] sim {sim} "
                  f"v{hit['vector_rank']} k{hit['keyword_rank']}")
            print(f"     {hit['citation']}")
            print(f"     {hit['text'][:100]}...")

    print("\n" + "=" * 68)
    print("  EXACT TABLE LOOKUP (no embeddings, no LLM)")
    print("=" * 68)
    for row in lookup_table_rows("1210 EKTN9", limit=2):
        print(f"  {row['source']} page {row['printed_page']}")
        print(f"  row: {' | '.join(row['row'])}")
