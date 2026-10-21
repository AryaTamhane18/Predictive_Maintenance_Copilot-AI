# eval_rag.py
# Measures the three things that matter for this project:
#
#   1. Does the search find the right passage?
#      Compared across semantic, keyword and hybrid retrieval.
#
#   2. Does it find exact identifiers — bearing designations and
#      part numbers? This is where semantic search is weakest and
#      keyword search earns its place.
#
#   3. Are the citations real? Every cited page is opened in the
#      source PDF and checked against the retrieved text. A citation
#      that cannot be verified is worse than no citation.
#
# Questions are generated from the documents themselves, so no
# ground truth has to be written by hand and this works the same
# on one document or a hundred.
#
# Usage:
#   python eval_rag.py              # 20 questions
#   python eval_rag.py --n 40
#   python eval_rag.py --regenerate # rebuild the question set

import os
import re
import sys
import json
import random
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
from pypdf import PdfReader
from tools.rag import _collection, search_manuals, DOCS_DIR

CHAT_URL   = "http://localhost:11434/api/generate"
CHAT_MODEL = "mistral"
CACHE      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "eval_questions.json")
SEED       = 42


def ask_llm(prompt: str) -> str:
    r = requests.post(CHAT_URL, json={"model": CHAT_MODEL, "prompt": prompt,
                                      "stream": False,
                                      "options": {"temperature": 0}},
                      timeout=300)
    r.raise_for_status()
    return r.json().get("response", "").strip()


# ── 1. Build questions from the documents ────────────────────────────

VAGUE = ("this passage", "the passage", "this document", "the text",
         "this section", "according to the")


def make_questions(n: int) -> list:
    """
    Take a passage, ask the model to write a question only that
    passage answers. The passage is then the known correct answer,
    so the ground truth comes for free.
    """
    data  = _collection().get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]
    if not docs:
        print("No chunks found. Run tools/rag.py first.")
        sys.exit(1)

    # only prose passages make good questions; a table of numbers does not
    pool = [i for i, d in enumerate(docs)
            if len(d) > 400 and metas[i].get("content_type") == "prose"]
    random.Random(SEED).shuffle(pool)

    cases = []
    print(f"Generating {n} questions ...")
    for i in pool:
        if len(cases) >= n:
            break
        passage = docs[i]
        try:
            q = ask_llm(
                "Below is a passage from a bearing maintenance manual.\n"
                "Write ONE specific question that this passage answers.\n"
                "Write it as a maintenance engineer would ask it, naming "
                "the actual subject.\n"
                "Never refer to 'this passage' or 'the text'.\n"
                "Output only the question.\n\n"
                f"PASSAGE:\n{passage[:1200]}\n\nQUESTION:"
            ).split("\n")[0].strip().strip('"')
        except Exception as exc:
            print(f"  model error: {exc}")
            continue

        if len(q) < 25 or not q.endswith("?"):
            continue
        if any(v in q.lower() for v in VAGUE):
            continue

        cases.append({"question": q,
                      "doc_id":   metas[i].get("doc_id"),
                      "pdf_page": metas[i].get("pdf_page"),
                      "chunk":    metas[i].get("chunk")})
        print(f"  [{len(cases):>2}/{n}] {q[:66]}")

    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print()
    return cases


def load_questions(n: int, regenerate: bool) -> list:
    if regenerate or not os.path.exists(CACHE):
        return make_questions(n)
    with open(CACHE, encoding="utf-8") as f:
        cases = json.load(f)
    if len(cases) < n:
        return make_questions(n)
    print(f"Using {n} cached questions (--regenerate to rebuild)\n")
    return cases[:n]


# ── 2. Retrieval score ───────────────────────────────────────────────

def rank_of_correct(case: dict, hits: list) -> int:
    """Position of the correct passage, 1-based. 0 means not found."""
    for i, h in enumerate(hits, 1):
        if (h["doc_id"] == case["doc_id"]
                and h["pdf_page"] == case["pdf_page"]
                and h.get("chunk") == case["chunk"]):
            return i
    return 0


def score_retrieval(cases: list, mode: str, k: int = 5) -> dict:
    top1 = top3 = topk = same_page = 0
    mrr  = 0.0

    for case in cases:
        hits = search_manuals(case["question"], top_k=k,
                              mode=mode).get("results", [])
        rank = rank_of_correct(case, hits)
        if rank:
            mrr  += 1 / rank
            top1 += rank == 1
            top3 += rank <= 3
            topk += 1
        if any(h["doc_id"] == case["doc_id"]
               and h["pdf_page"] == case["pdf_page"] for h in hits):
            same_page += 1

    n = len(cases) or 1
    return {"mode": mode, "top1": top1 / n, "top3": top3 / n,
            "topk": topk / n, "page": same_page / n, "mrr": mrr / n}


# ── 3. Exact identifiers ─────────────────────────────────────────────

ID_RE = re.compile(r"\b(?=[A-Za-z]*\d)[A-Z0-9][A-Z0-9\-/]{4,14}\b")


def identifier_cases(limit: int = 15) -> list:
    """
    Collect bearing designations and part numbers that appear in only
    one passage, so there is exactly one correct answer.
    """
    data  = _collection().get(include=["documents", "metadatas"])
    docs, metas = data["documents"], data["metadatas"]

    seen = {}
    for i, doc in enumerate(docs):
        for token in set(ID_RE.findall(doc)):
            seen.setdefault(token, []).append(i)

    unique = [(t, v[0]) for t, v in seen.items() if len(v) == 1]
    unique.sort()
    random.Random(SEED).shuffle(unique)

    return [{"question": t,
             "doc_id":   metas[i].get("doc_id"),
             "pdf_page": metas[i].get("pdf_page"),
             "chunk":    metas[i].get("chunk")}
            for t, i in unique[:limit]]


# ── 4. Citation check ────────────────────────────────────────────────

def words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]{4,}", text.lower()))


def check_citations(cases: list, sample: int = 12) -> dict:
    """
    Open each cited page in the real PDF and check that the retrieved
    text is actually on it. This is what separates a citation from a
    plausible-looking reference.
    """
    readers = {}
    checked = valid = 0
    bad     = []

    for case in cases[:sample]:
        for hit in search_manuals(case["question"], top_k=2).get("results", []):
            path = os.path.join(DOCS_DIR, hit["source"])
            if not os.path.exists(path):
                continue
            if path not in readers:
                readers[path] = PdfReader(path)

            try:
                page_text = readers[path].pages[hit["pdf_page"] - 1].extract_text() or ""
            except Exception:
                continue

            checked += 1
            retrieved = words(hit["text"])
            onpage    = words(page_text)
            if not retrieved:
                continue
            overlap = len(retrieved & onpage) / len(retrieved)

            if overlap > 0.6:
                valid += 1
            else:
                bad.append((hit["citation"], round(overlap, 2)))

    return {"checked": checked, "valid": valid,
            "rate": valid / max(checked, 1), "bad": bad[:3]}


# ── Report ───────────────────────────────────────────────────────────

def table(rows: list):
    print(f"  {'mode':<9} {'top-1':>7} {'top-3':>7} {'top-5':>7} "
          f"{'page':>7} {'MRR':>7}")
    print("  " + "-" * 50)
    for r in rows:
        print(f"  {r['mode']:<9} {r['top1']:>6.0%} {r['top3']:>7.0%} "
              f"{r['topk']:>7.0%} {r['page']:>7.0%} {r['mrr']:>7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print(f"  RAG EVALUATION — {_collection().count()} chunks indexed")
    print("=" * 62 + "\n")

    cases = load_questions(args.n, args.regenerate)

    print("-" * 62)
    print("  1. RETRIEVAL — is the correct passage returned?")
    print("-" * 62)
    table([score_retrieval(cases, m) for m in ("vector", "keyword", "hybrid")])
    print("\n  top-n = correct passage within the first n results")
    print("  page  = a passage from the correct page was returned")
    print("  MRR   = 1/rank of the correct passage, averaged\n")

    print("-" * 62)
    print("  2. EXACT IDENTIFIERS — bearing designations, part numbers")
    print("-" * 62)
    ids = identifier_cases()
    if ids:
        print(f"  {len(ids)} identifiers, e.g. "
              f"{', '.join(c['question'] for c in ids[:4])}\n")
        table([score_retrieval(ids, m) for m in ("vector", "keyword", "hybrid")])
    else:
        print("  none found")
    print()

    print("-" * 62)
    print("  3. CITATIONS — does the cited page really contain the text?")
    print("-" * 62)
    cit = check_citations(cases)
    print(f"  Citations checked : {cit['checked']}")
    print(f"  Verified on page  : {cit['valid']}")
    print(f"  Valid             : {cit['rate']:.0%}")
    for citation, overlap in cit["bad"]:
        print(f"    mismatch ({overlap}): {citation[:70]}")
    print()

    print("=" * 62)
    print("  Questions are written by the model from passages it was")
    print("  shown, so their wording echoes the source. Read the")
    print("  comparison between modes rather than any single number.")
    print("=" * 62)


if __name__ == "__main__":
    main()
