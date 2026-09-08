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

import numpy as np
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
    # Written for users, and the source for "what can you do?", "where do your
    # prices come from?", "do you remember what I said?".  Keep it in step with
    # actual behaviour - it is quoted verbatim.
    "Noah_Capabilities.md",
}

CACHE_FILE = BASE_DIR / "rag" / "web_faq_cache.json"
PAYTO_FAQ_URL = "https://payto.one/FAQ"

# A chunk has to be a sentence or two before it can answer anything; the FAQ
# scrape yields a lot of bare interface labels ("Offers & loyalty programs").
MINIMUM_CHUNK_CHARACTERS = 60
DEFAULT_MINIMUM_SCORE = 0.12
# A scoped search is already restricted to the document written to answer that
# kind of question, so the floor only has to reject genuine nonsense rather than
# arbitrate between 280 unrelated fragments.
SCOPED_MINIMUM_SCORE = 0.04

# Retrieval prior. The FAQ corpus is regex-scraped out of the site's JavaScript
# bundle, so it is mostly interface labels and legal boilerplate - high recall,
# low precision, and 250-odd fragments against a dozen curated sections, which
# lets it outrank the real answer on term overlap alone. Curated documents are
# written to answer questions, so they win a close call.
CURATED_WEIGHT = 1.0
SCRAPED_FAQ_WEIGHT = 0.75
WEB_FAQ_PREFIX = "[PayTo FAQ Web]"

# Retrieval scopes. "What can you do?" and "What is PayTo's refund policy?" are
# answered by different documents, and ranking them in one pool means tuning a
# weight until both happen to win - which is fragile in one language and broken
# in the other. The classifier already separates the two intents, so the caller
# names the scope and retrieval only searches that shelf.
CAPABILITY_DOCUMENT = "Noah_Capabilities"
SCOPE_CAPABILITIES = "capabilities"   # what Noah is, what it can and cannot do
SCOPE_PAYTO = "payto"                 # PayTo policy, privacy, features, FAQ

# scikit-learn ships an English stop list only. Without a German one, a query
# like "Woher kommen deine Preise?" matches any short German marketing string
# containing "deine", and short chunks win on cosine length normalisation.
GERMAN_STOP_WORDS = [
    "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am",
    "an", "ander", "andere", "anderem", "anderen", "anderer", "anderes", "auch",
    "auf", "aus", "bei", "beim", "bin", "bis", "bist", "da", "damit", "dann",
    "das", "dass", "dein", "deine", "deinem", "deinen", "deiner", "deines",
    "dem", "den", "denn", "der", "des", "dessen", "dich", "die", "dies",
    "diese", "diesem", "diesen", "dieser", "dieses", "dir", "doch", "dort",
    "du", "durch", "ein", "eine", "einem", "einen", "einer", "eines", "er",
    "es", "etwas", "euer", "eure", "für", "gegen", "gewesen", "hab", "habe",
    "haben", "hat", "hatte", "hatten", "hier", "hin", "ich", "ihm", "ihn",
    "ihnen", "ihr", "ihre", "im", "in", "indem", "ins", "ist", "ja", "jede",
    "jedem", "jeden", "jeder", "jedes", "jene", "jetzt", "kann", "kannst",
    "kein", "keine", "können", "könnt", "machen", "mal", "man", "manche",
    "mein", "meine", "meinem", "meinen", "meiner", "mich", "mir", "mit",
    "muss", "müssen", "nach", "nicht", "nichts", "noch", "nun", "nur", "ob",
    "oder", "ohne", "schon", "sehr", "sein", "seine", "selbst", "sich", "sie",
    "sind", "so", "soll", "sollen", "sondern", "sonst", "über", "um", "und",
    "uns", "unser", "unsere", "unter", "vom", "von", "vor", "war", "waren",
    "was", "weg", "weil", "weiter", "welche", "welchem", "welchen", "welcher",
    "welches", "wenn", "werde", "werden", "wie", "wieder", "will", "wir",
    "wird", "wirst", "wo", "wollen", "wollte", "würde", "würden", "zu", "zum",
    "zur", "zwar", "zwischen",
]

_CHUNKS_CACHE: List[str] = []
_CHUNKS_LOADED = False
_VECTORIZER: Optional[TfidfVectorizer] = None
_MATRIX = None
_WEIGHTS = None


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


def _markdown_sections(path: Path) -> List[str]:
    """Split Markdown into one chunk per paragraph, labelled by its heading.

    Paragraph rather than whole section, because these documents are written
    twice - once in English, once in German. A single mixed chunk dilutes both:
    a German query scores poorly against a chunk that is half English. Splitting
    on the blank line gives each language its own dense chunk while the heading
    prefix keeps the topic attached to both.
    """
    text = path.read_text(encoding="utf-8")
    chunks, heading, paragraph = [], None, []

    def flush():
        if not paragraph:
            return
        content = " ".join(" ".join(paragraph).split())
        paragraph.clear()
        if len(content) > MINIMUM_CHUNK_CHARACTERS:
            # The heading goes in the label, not the body. The label is part of
            # the indexed text so its terms still help matching, but keeping it
            # out of the body means the first sentence of a chunk is a real
            # sentence - the response layer quotes that directly.
            label = f"{path.stem}{' - ' + heading if heading else ''}"
            chunks.append(f"[{label}]: {content}")

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            flush()
            heading = stripped[3:].strip()
        elif stripped.startswith("# "):
            flush()
        elif not stripped:
            flush()
        else:
            paragraph.append(stripped)
    flush()
    return chunks


def _load_all_chunks() -> List[str]:
    chunks: List[str] = []
    seen_files = set()

    for directory in RAG_DIRS:
        if not directory.exists():
            continue
        for markdown_path in sorted(directory.glob("*.md")):
            if markdown_path.name in seen_files:
                continue
            seen_files.add(markdown_path.name)
            if markdown_path.name not in USER_FACING_DOCUMENTS:
                logger.info("Skipping %s: not on the user-facing allowlist.",
                            markdown_path.name)
                continue
            try:
                chunks.extend(_markdown_sections(markdown_path))
            except Exception:
                logger.warning("Could not read %s", markdown_path, exc_info=True)

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
    global _CHUNKS_CACHE, _CHUNKS_LOADED, _VECTORIZER, _MATRIX, _WEIGHTS
    if _CHUNKS_LOADED and not reload:
        return _CHUNKS_CACHE

    _CHUNKS_CACHE = _load_all_chunks()
    _CHUNKS_LOADED = True
    _VECTORIZER, _MATRIX, _WEIGHTS = None, None, None
    if _CHUNKS_CACHE:
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            from sklearn.pipeline import FeatureUnion
            stop_words = list(ENGLISH_STOP_WORDS) + GERMAN_STOP_WORDS
            # Word features carry topic; character n-grams carry morphology, which
            # German needs badly - "Sprachen"/"Sprache", "Preise"/"Preisen",
            # "überweisen"/"Überweisung" are the same question to a user and
            # different tokens to a word-level vectoriser.
            _VECTORIZER = FeatureUnion([
                ("word", TfidfVectorizer(stop_words=stop_words, sublinear_tf=True)),
                # min_df=1: the corpus is a few hundred short chunks, so a stem
                # that appears once ("merk-" in the memory section) is exactly
                # the discriminative signal, not noise to be pruned away.
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5),
                                         sublinear_tf=True, min_df=1)),
            ])
            _MATRIX = _VECTORIZER.fit_transform(_CHUNKS_CACHE)
            _WEIGHTS = np.array([
                SCRAPED_FAQ_WEIGHT if chunk.startswith(WEB_FAQ_PREFIX) else CURATED_WEIGHT
                for chunk in _CHUNKS_CACHE
            ])
        except Exception:
            logger.exception("Could not build the retrieval index")
            _VECTORIZER, _MATRIX, _WEIGHTS = None, None, None
    logger.info("RAG corpus: %d chunks from %d user-facing documents + FAQ",
                len(_CHUNKS_CACHE), len(USER_FACING_DOCUMENTS))
    return _CHUNKS_CACHE


def retrieve(query: str, limit: int = 3,
             minimum_score: Optional[float] = None,
             scope: Optional[str] = None) -> List[str]:
    """Return the most relevant chunks, or nothing when none are relevant.

    `scope` restricts the search: SCOPE_CAPABILITIES searches only the document
    describing Noah itself, SCOPE_PAYTO only PayTo's own documentation and FAQ.
    Passing None searches everything.

    Returning nothing is a useful answer: the response layer turns it into
    "I don't have anything on that", which beats letting the model improvise
    from an unrelated interface label.
    """
    if minimum_score is None:
        minimum_score = SCOPED_MINIMUM_SCORE if scope else DEFAULT_MINIMUM_SCORE

    chunks = get_chunks()
    if not chunks or _VECTORIZER is None or _MATRIX is None:
        return []
    try:
        scores = cosine_similarity(_VECTORIZER.transform([query]), _MATRIX).ravel()
        if _WEIGHTS is not None:
            scores = scores * _WEIGHTS
    except Exception:
        logger.exception("RAG retrieval error")
        return []

    if scope is not None:
        is_capability = np.array([CAPABILITY_DOCUMENT in c for c in chunks])
        keep = is_capability if scope == SCOPE_CAPABILITIES else ~is_capability
        scores = np.where(keep, scores, -1.0)

    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [chunks[index] for index, score in ranked[:limit] if score >= minimum_score]
