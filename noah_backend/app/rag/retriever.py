"""Retrieval over PayTo's *user-facing* documentation.

Two problems this replaces:

* the corpus was every PDF it could find, including ``Noah - Week 1 Plan.pdf``,
  ``APIs and Subscriptions.pdf`` and the intent-taxonomy spec - internal
  engineering documents that were retrievable and quotable to end users.  Only
  documents on `USER_FACING_DOCUMENTS` are indexed now, and anything else found
  in those folders is skipped with a log line.
* every query re-fitted a TF-IDF vectoriser over the whole corpus.  The index is
  now built once and reused.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAG_DIRS = [BASE_DIR / "app" / "rag", BASE_DIR / "rag", BASE_DIR / "docs"]

# Allowlist, not a glob: a document reaches end users only if it is written for
# them.  Add new customer-facing PDFs here; internal specs, roadmaps and API
# cost sheets must stay out.
USER_FACING_DOCUMENTS = {
    "Privacy_Policy.pdf",
    "Terms_And_Conditions.pdf",
    "PayTo_User_Guide.pdf",
}

CACHE_FILE = BASE_DIR / "rag" / "web_faq_cache.json"
PAYTO_FAQ_URL = "https://payto.one/FAQ"

# A chunk has to be a sentence or two before it can answer anything; the FAQ
# scrape yields a lot of bare interface labels ("Offers & loyalty programs").
MINIMUM_CHUNK_CHARACTERS = 60
DEFAULT_MINIMUM_SCORE = 0.12

_CHUNKS_CACHE: List[str] = []
_CHUNKS_LOADED = False
_VECTORIZER: Optional[TfidfVectorizer] = None
_MATRIX = None


def fetch_web_faq() -> List[str]:
    """Fetch FAQ statements from payto.one, falling back to the local cache."""
    cached: List[str] = []
    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Could not read the FAQ cache", exc_info=True)

    try:
        request = urllib.request.Request(
            PAYTO_FAQ_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NoahAI/1.0"},
        )
        with urllib.request.urlopen(request, timeout=4) as response:
            html_document = response.read().decode("utf-8")

        scripts = re.findall(r'src="([^"]+index[^"]+\.js)"', html_document)
        collected: List[str] = []
        for script in scripts:
            url = "https://payto.one" + script if script.startswith("/") else script
            with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=4,
            ) as script_response:
                content = script_response.read().decode("utf-8")
            collected.extend(
                match.strip()
                for match in re.findall(r'["\']([^"\']{25,600})["\']', content)
                if any(keyword in match.lower() for keyword in (
                    "payto", "loyalty", "points", "wallet", "merchant", "faq",
                    "barcode", "refund", "card", "privacy", "deal", "discount"))
                and not match.startswith(("http", "/"))
            )

        if collected:
            unique = list(dict.fromkeys(collected))
            try:
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                CACHE_FILE.write_text(json.dumps(unique, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
            except Exception:
                logger.debug("Could not write the FAQ cache", exc_info=True)
            return unique
    except Exception as error:
        logger.warning("Could not fetch live FAQ from %s (%s); using cache.",
                       PAYTO_FAQ_URL, error)

    return cached


def _load_all_chunks() -> List[str]:
    chunks: List[str] = []
    seen_files = set()

    for directory in RAG_DIRS:
        if not directory.exists():
            continue
        for pdf_path in sorted(directory.glob("*.pdf")):
            if pdf_path.name in seen_files:
                continue
            seen_files.add(pdf_path.name)
            if pdf_path.name not in USER_FACING_DOCUMENTS:
                logger.info("Skipping %s: not on the user-facing allowlist.", pdf_path.name)
                continue
            try:
                reader = PdfReader(str(pdf_path))
            except Exception:
                logger.warning("Could not read %s", pdf_path, exc_info=True)
                continue
            for page_index, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                for start in range(0, len(text), 900):
                    chunk = text[start:start + 1100].strip()
                    if len(chunk) > MINIMUM_CHUNK_CHARACTERS:
                        chunks.append(f"[{pdf_path.stem} p.{page_index + 1}]: {chunk}")

    for item in fetch_web_faq():
        if len(item.strip()) > MINIMUM_CHUNK_CHARACTERS:
            chunks.append(f"[PayTo FAQ Web]: {item.strip()}")

    return chunks


def get_chunks(reload: bool = False) -> List[str]:
    """Return the indexed corpus, loading and vectorising it once."""
    global _CHUNKS_CACHE, _CHUNKS_LOADED, _VECTORIZER, _MATRIX
    if _CHUNKS_LOADED and not reload:
        return _CHUNKS_CACHE

    _CHUNKS_CACHE = _load_all_chunks()
    _CHUNKS_LOADED = True
    _VECTORIZER, _MATRIX = None, None
    if _CHUNKS_CACHE:
        try:
            _VECTORIZER = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            _MATRIX = _VECTORIZER.fit_transform(_CHUNKS_CACHE)
        except Exception:
            logger.exception("Could not build the retrieval index")
            _VECTORIZER, _MATRIX = None, None
    logger.info("RAG corpus: %d chunks from %d user-facing documents + FAQ",
                len(_CHUNKS_CACHE), len(USER_FACING_DOCUMENTS))
    return _CHUNKS_CACHE


def retrieve(query: str, limit: int = 3,
             minimum_score: float = DEFAULT_MINIMUM_SCORE) -> List[str]:
    """Return the most relevant chunks, or nothing when none are relevant.

    Returning nothing is a useful answer: the response layer turns it into
    "I don't have anything on that", which beats letting the model improvise
    from an unrelated interface label.
    """
    chunks = get_chunks()
    if not chunks or _VECTORIZER is None or _MATRIX is None:
        return []
    try:
        scores = cosine_similarity(_VECTORIZER.transform([query]), _MATRIX).ravel()
    except Exception:
        logger.exception("RAG retrieval error")
        return []
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [chunks[index] for index, score in ranked[:limit] if score >= minimum_score]
