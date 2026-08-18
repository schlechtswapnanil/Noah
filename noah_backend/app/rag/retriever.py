"""RAG Retriever module for Noah.

Grounds responses in:
1. Local PDF documents in the `rag/` folder (architecture guides, intent taxonomy, API roadmaps).
2. Live and cached content from the PayTo FAQ website (https://payto.one/FAQ).
"""

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import List

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAG_DIRS = [
    BASE_DIR / "rag",
    BASE_DIR / "app" / "rag",
    BASE_DIR / "docs",
    BASE_DIR.parent,
]
CACHE_FILE = BASE_DIR / "rag" / "web_faq_cache.json"
PAYTO_FAQ_URL = "https://payto.one/FAQ"

_CHUNKS_CACHE: List[str] = []
_CHUNKS_LOADED: bool = False


def fetch_web_faq() -> List[str]:
    """Fetch FAQ statements from https://payto.one/FAQ or fallback to cache."""
    cached_items: List[str] = []
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached_items = json.load(f)
        except Exception:
            pass

    try:
        req = urllib.request.Request(
            PAYTO_FAQ_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NoahAI/1.0"}
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            html_doc = resp.read().decode("utf-8")
            
        scripts = re.findall(r'src="([^"]+index[^"]+\.js)"', html_doc)
        faq_texts = []
        for s in scripts:
            js_url = "https://payto.one" + s if s.startswith("/") else s
            with urllib.request.urlopen(
                urllib.request.Request(js_url, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=4
            ) as js_resp:
                js_content = js_resp.read().decode("utf-8")
                matches = re.findall(r'["\']([^"\']{25,600})["\']', js_content)
                relevant = [
                    m.strip() for m in matches
                    if any(k in m.lower() for k in [
                        "payto", "loyalty", "points", "wallet", "merchant",
                        "faq", "barcode", "refund", "card", "privacy", "deal", "discount"
                    ]) and not m.startswith("http") and not m.startswith("/")
                ]
                faq_texts.extend(relevant)

        if faq_texts:
            # Deduplicate while preserving order
            seen = set()
            unique_faqs = []
            for item in faq_texts:
                if item not in seen:
                    seen.add(item)
                    unique_faqs.append(item)

            # Update cache file
            try:
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(unique_faqs, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
            return unique_faqs
    except Exception as e:
        logger.warning(f"Could not fetch live FAQ from {PAYTO_FAQ_URL} ({e}); using cache.")

    return cached_items


def _load_all_chunks() -> List[str]:
    """Load chunks from PDFs and the web FAQ."""
    chunks: List[str] = []

    # 1. Load from PDFs in rag directories
    seen_files = set()
    for rag_dir in RAG_DIRS:
        if not rag_dir.exists():
            continue
        for pdf_path in rag_dir.glob("*.pdf"):
            if pdf_path.name in seen_files:
                continue
            seen_files.add(pdf_path.name)
            try:
                reader = PdfReader(str(pdf_path))
                for page_idx, page in enumerate(reader.pages):
                    text = (page.extract_text() or "").strip()
                    if text:
                        # Split by paragraphs or fixed window
                        for i in range(0, len(text), 900):
                            chunk = text[i:i + 1100].strip()
                            if len(chunk) > 30:
                                chunks.append(f"[{pdf_path.stem} p.{page_idx+1}]: {chunk}")
            except Exception as e:
                logger.debug(f"Skipping PDF {pdf_path}: {e}")
                continue

    # 2. Load Web FAQ items
    web_faqs = fetch_web_faq()
    for item in web_faqs:
        chunks.append(f"[PayTo FAQ Web]: {item}")

    return chunks


def get_chunks(reload: bool = False) -> List[str]:
    """Get all cached chunks."""
    global _CHUNKS_CACHE, _CHUNKS_LOADED
    if not _CHUNKS_LOADED or reload:
        _CHUNKS_CACHE = _load_all_chunks()
        _CHUNKS_LOADED = True
    return _CHUNKS_CACHE


def retrieve(query: str, limit: int = 3, minimum_score: float = 0.06) -> List[str]:
    """Retrieve the most relevant context chunks for a query."""
    chunks = get_chunks()
    if not chunks:
        return []

    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform([query, *chunks])
        scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [chunks[index] for index, score in ranked[:limit] if score >= minimum_score]
    except Exception as e:
        logger.exception(f"RAG retrieval error: {e}")
        return []
