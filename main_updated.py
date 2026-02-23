#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a single structured database from two Word documents.

Input DOCX format assumption (flexible, best-effort):
Entries contain labeled fields like:
  Data:
  Ora:
  Autor:
  Link:
  Titlu:
  Text:
Repeated multiple times, sometimes preceded by numbering like "1)".
The script groups paragraphs into entries based on "Data:" occurrences
and captures multi-paragraph "Text:" until the next entry starts.

Outputs:
  - proteste.sqlite  (SQLite DB)
  - proteste.jsonl   (JSON Lines)
  - proteste.csv     (flat CSV, UTF-8 BOM for Excel)
  - proteste.xlsx    (native Excel)
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from docx import Document  # pip install python-docx
from openpyxl import Workbook  # pip install openpyxl

import numpy as np  # pip install numpy
from sklearn.feature_extraction.text import TfidfVectorizer  # pip install scikit-learn
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import unicodedata
import hdbscan
from typing import Iterable
from sentence_transformers import SentenceTransformer
import umap

# ----------------------------
# Parsing helpers
# ----------------------------

RO_MONTHS = {
    "ianuarie": 1, "februarie": 2, "martie": 3, "aprilie": 4, "mai": 5,
    "iunie": 6, "iulie": 7, "august": 8, "septembrie": 9, "octombrie": 10,
    "noiembrie": 11, "decembrie": 12,
}

RO_STOPWORDS_MIN = {
    "și", "şi", "să", "sa", "pana", "am", "și", "si", "sau", "în", "in", "la", "de", "din", "cu", "pe", "pentru", "prin",
    "că", "ca", "care", "ce", "este", "sunt", "a", "ai", "ale", "al", "un", "o",
    "unei", "unui", "lui", "ei", "lor", "mai", "doar", "fost", "fi", "se", "nu",
    "da", "au", "iar", "între", "intre", "despre", "după", "dupa",
    "aceasta", "acesta", "aceste", "acest", "acelor", "acele", "acei",
    "aici", "acolo", "unde", "când", "cand",
}


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def build_doc_text(title: Optional[str], text: str) -> str:
    t = normalize_ws(title or "")
    x = normalize_ws(text or "")
    if t and x:
        return f"{t}. {x}"
    return t or x


def try_parse_date_ro(date_raw: str) -> Optional[str]:
    """
    Best-effort parse Romanian-ish dates to ISO YYYY-MM-DD.
    """
    if not date_raw:
        return None

    s = normalize_ws(date_raw)
    s = s.split(",")[0].strip()

    m = re.match(r"^(\d{1,2})[./](\d{1,2})[./](\d{4})$", s)
    if m:
        d, mo, y = map(int, m.groups())
        try:
            return datetime(y, mo, d).date().isoformat()
        except ValueError:
            return None

    m = re.match(r"^(\d{1,2})\s+([A-Za-zăâîșţț]+)\s+(\d{4})$", s, flags=re.IGNORECASE)
    if m:
        d = int(m.group(1))
        mon_txt = m.group(2).lower().replace("ţ", "ț")
        y = int(m.group(3))
        mo = RO_MONTHS.get(mon_txt)
        if mo:
            try:
                return datetime(y, mo, d).date().isoformat()
            except ValueError:
                return None

    return None


LABEL_PATTERNS = {
    "date_raw": re.compile(r"^\s*Data\s*:\s*(.*)\s*$", re.IGNORECASE),
    "time_raw": re.compile(r"^\s*Ora\s*:\s*(.*)\s*$", re.IGNORECASE),
    "author": re.compile(r"^\s*Autor\s*:\s*(.*)\s*$", re.IGNORECASE),
    "link": re.compile(r"^\s*Link\s*:\s*(.*)\s*$", re.IGNORECASE),
    "title": re.compile(r"^\s*Titlu\s*:\s*(.*)\s*$", re.IGNORECASE),
    "text_label": re.compile(r"^\s*Text\s*:\s*(.*)\s*$", re.IGNORECASE),
}


def is_entry_separator(par: str) -> bool:
    return bool(re.match(r"^\s*\d+\s*\)\s*$", (par or "").strip()))


# Romanian letters included; \w also includes digits/underscore, so we avoid it.
_RO_WORD_RE = re.compile(r"[A-Za-zĂÂÎȘȚăâîșț]+", re.UNICODE)

_NLP_RO: Optional[Any] = None
_NLP_RO_KIND: str = "none"

def make_spacy_ro_lemmatizer():
    """
    Best-effort Romanian lemmatizer.

    Priority:
      1) spacy-stanza (Stanza) Romanian pipeline: best coverage/robustness for RO.
      2) spaCy model ro_core_news_sm if available.
      3) None (caller will use regex fallback tokenizer).
    """
    # 1) Prefer Stanza via spacy-stanza (most robust for Romanian)
    try:
        import spacy_stanza  # type: ignore
        # processors chosen for speed + lemmatization quality
        nlp = spacy_stanza.load_pipeline("ro", processors="tokenize,pos,lemma")
        return nlp, "stanza"
    except Exception:
        pass

    # 2) Fall back to spaCy model if installed
    try:
        import spacy  # type: ignore
        nlp = spacy.load("ro_core_news_sm", disable=["ner", "parser"])
        return nlp, "spacy"
    except Exception:
        pass

    # 3) Final fallback: regex-based tokenizer
    return None, "regex"


def spacy_ro_lemmas(text: str) -> List[str]:
    """
    Returns Romanian lemmas/tokens for TF-IDF.
    - Keeps alphabetic tokens, length >= 2
    - Drops stopwords when supported by the pipeline
    - Never crashes: falls back to a Romanian-aware regex tokenizer
    """
    global _NLP_RO, _NLP_RO_KIND

    if not text:
        return []

    if _NLP_RO is None:
        _NLP_RO, _NLP_RO_KIND = make_spacy_ro_lemmatizer()

    # Regex fallback (no lemmatization but safe + consistent)
    if _NLP_RO is None or _NLP_RO_KIND == "regex":
        return [w.lower() for w in _RO_WORD_RE.findall(text) if len(w) >= 2]

    doc = _NLP_RO(text)

    out: List[str] = []
    for tok in doc:
        # Keep alphabetic tokens
        if not getattr(tok, "is_alpha", False):
            continue

        # Drop stopwords if the pipeline provides them
        # (stanza-backed tokens sometimes don't have is_stop reliably)
        is_stop = getattr(tok, "is_stop", False)
        if is_stop:
            continue

        lemma = (getattr(tok, "lemma_", "") or tok.text).lower().strip()

        # Some pipelines return "_" or empty lemma in edge cases
        if not lemma or lemma == "_":
            lemma = tok.text.lower().strip()

        if len(lemma) >= 2:
            out.append(lemma)

    return out
@dataclass
class ProtestRecord:
    source_file: str
    source_name: Optional[str]
    date_raw: Optional[str]
    date_iso: Optional[str]
    time_raw: Optional[str]
    author: Optional[str]
    link: Optional[str]
    title: Optional[str]
    text: str

    # Separate tagging outputs (one column per method)
    tags_tfidf: str = ""
    tag_bert: str = ""
    tags_lda: str = ""
    tags_hdbscan: str = ""
    tags_umap_hdbscan: str = ""

    # Backward-compatible combined tags (JSON array string)
    tags: str = "[]"

    content_hash: str = ""



def read_docx_paragraphs(docx_path: Path) -> List[str]:
    """
    Reads only normal paragraphs.
    (User says this works well for their documents.)
    """
    doc = Document(str(docx_path))
    pars: List[str] = []
    for p in doc.paragraphs:
        t = p.text
        if t is None:
            continue
        t = t.strip("\u00a0").strip()
        if t.strip():
            pars.append(t)
    return pars


def parse_records_from_paragraphs(paragraphs: List[str], source_file: str) -> List[ProtestRecord]:
    records: List[ProtestRecord] = []

    cur: Dict[str, Optional[str]] = {
        "date_raw": None,
        "time_raw": None,
        "author": None,
        "link": None,
        "title": None,
    }
    in_text = False
    text_chunks: List[str] = []

    def flush_record():
        nonlocal cur, in_text, text_chunks, records
        text = normalize_ws("\n".join(text_chunks).strip())
        if not any(cur.values()) and not text:
            cur = {k: None for k in cur}
            in_text = False
            text_chunks = []
            return

        date_iso = try_parse_date_ro(cur.get("date_raw") or "")
        sig = normalize_ws(f"{cur.get('title') or ''}||{cur.get('link') or ''}||{text[:500]}")
        content_hash = hashlib.sha256(sig.encode("utf-8")).hexdigest()

        rec = ProtestRecord(
            source_file=source_file,
            source_name=None,
            date_raw=cur.get("date_raw"),
            date_iso=date_iso,
            time_raw=cur.get("time_raw"),
            author=cur.get("author"),
            link=cur.get("link"),
            title=cur.get("title"),
            text=text,
            tags_tfidf="",
            tag_bert="",
            tags_lda="",
            tags_hdbscan="",
            tags_umap_hdbscan="",
            tags="[]",
            content_hash=content_hash,
        )
        records.append(rec)

        cur = {k: None for k in cur}
        in_text = False
        text_chunks = []

    for par in paragraphs:
        if is_entry_separator(par):
            continue

        if LABEL_PATTERNS["date_raw"].match(par) and (any(cur.values()) or text_chunks):
            flush_record()

        matched = False
        for key, pat in LABEL_PATTERNS.items():
            m = pat.match(par)
            if not m:
                continue
            matched = True

            if key == "text_label":
                in_text = True
                remainder = (m.group(1) or "").strip()
                if remainder:
                    text_chunks.append(remainder)
            else:
                cur[key] = (m.group(1) or "").strip() or None
                if in_text:
                    in_text = False
            break

        if matched:
            continue

        if in_text:
            text_chunks.append(par)
        else:
            if any([cur.get("date_raw"), cur.get("title"), cur.get("link")]):
                text_chunks.append(par)

    flush_record()

    cleaned: List[ProtestRecord] = []
    for r in records:
        if (r.title or r.link or r.author or r.date_raw) and r.text:
            cleaned.append(r)
    return cleaned

def fold_diacritics(s: str) -> str:
    # naive but effective: turn "național" -> "national"
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )

def clean_topic_label(
    features: List[str],
    max_items: int = 8,
    prefer_bigrams: bool = True,
    fold_ro_diacritics: bool = True,
) -> str:
    """
    Deduplicate topic label terms so you don't get:
      'parc parc național ... parc national'
    Strategy:
      - optionally fold diacritics for duplicate detection
      - prefer longer n-grams (bigrams) over unigrams
      - don't include a unigram if it's already contained in a chosen bigram
      - deduplicate by (normalized) token set / string
    """
    def norm(x: str) -> str:
        x2 = x.lower().strip()
        return fold_diacritics(x2) if fold_ro_diacritics else x2

    # Prefer longer terms first (bigrams before unigrams)
    if prefer_bigrams:
        features = sorted(features, key=lambda x: (-len(x.split()), x))

    chosen: List[str] = []
    chosen_norm: set[str] = set()
    covered_tokens: set[str] = set()

    for feat in features:
        if len(chosen) >= max_items:
            break

        f_norm = norm(feat)
        if f_norm in chosen_norm:
            continue

        toks = f_norm.split()

        # If unigram is already part of a chosen bigram, skip
        if len(toks) == 1 and toks[0] in covered_tokens:
            continue

        # If bigram is essentially redundant because all its tokens are already covered, skip
        if len(toks) > 1 and all(t in covered_tokens for t in toks):
            continue

        chosen.append(feat)
        chosen_norm.add(f_norm)
        for t in toks:
            covered_tokens.add(t)

    return " ".join(chosen)


# ----------------------------
# TF-IDF topic tagging
# ----------------------------

def tag_topics_tfidf_kmeans_safe(
    docs: List[str],
    n_topics: int = 12,
    top_terms: int = 8,
    min_df: int = 2,
    max_df: float = 0.85,
    stopwords: Optional[set] = None,
    random_state: int = 42,
) -> Tuple[List[List[str]], List[str]]:
    """
    Robust TF-IDF + KMeans topic tagging.

    Returns:
      tags_per_doc: list of list-of-tags (one list per doc; typically 1 tag/doc)
      topic_labels: list of human-readable topic descriptors (top TF-IDF terms per cluster)
    """
    stopwords = stopwords or set()

    keep_idx = [i for i, d in enumerate(docs) if d and d.strip()]
    docs_clean = [docs[i] for i in keep_idx]
    if not docs_clean:
        return [[] for _ in docs], []

    # For small corpora, min_df=2 often wipes vocabulary.
    min_df_eff = 1 if len(docs_clean) < 30 else min_df

    # Keep Unicode letter tokens (incl. Romanian diacritics), length>=2, ignore numbers
    token_pattern = r"(?u)\b[^\W\d_]{2,}\b"

    vec = TfidfVectorizer(
        tokenizer=spacy_ro_lemmas,
        preprocessor=None,
        token_pattern=None,  # IMPORTANT when using custom tokenizer
        ngram_range=(1, 2),
        min_df=min_df_eff,
        max_df=max_df,
    )
    X = vec.fit_transform(docs_clean)

    # Fallback: remove stopwords + unigrams only if vocabulary is empty
    if X.shape[1] == 0:
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 1),
            min_df=1,
            max_df=0.95,
            token_pattern=token_pattern,
        )
        X = vec.fit_transform(docs_clean)

    if X.shape[1] == 0:
        return [[] for _ in docs], []

    k = min(n_topics, X.shape[0])
    if k < 2:
        tags_per_doc = [[] for _ in docs]
        for i in keep_idx:
            tags_per_doc[i] = ["topic_00: single_doc"]
        return tags_per_doc, ["single_doc"]

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    clusters = km.fit_predict(X)

    terms = np.array(vec.get_feature_names_out())
    topic_labels: List[str] = []
    for c in range(k):
        center = km.cluster_centers_[c]
        top_idx = center.argsort()[-top_terms:][::-1]
        top_feats = terms[top_idx].tolist()
        topic_labels.append(clean_topic_label(top_feats, max_items=8))

    tags_per_doc = [[] for _ in docs]
    for j, orig_i in enumerate(keep_idx):
        c = int(clusters[j])
        tags_per_doc[orig_i] = [f"topic_{c:02d}: {topic_labels[c]}"]

    return tags_per_doc, topic_labels

def tag_topics_lda_safe(
    docs: List[str],
    n_topics: int = 12,
    top_terms: int = 8,
    min_df: int = 2,
    max_df: float = 0.9,
    stopwords: Optional[set] = None,
    random_state: int = 42,
) -> Tuple[List[List[str]], List[str]]:
    """
    LDA topic tagging (bag-of-words).

    Returns:
      tags_per_doc: list of list-of-tags per doc (1 tag/doc: top topic)
      topic_labels: list of human-readable topic labels (top terms per topic)

    Robustness:
      - If corpus is small, min_df auto to 1
      - If vocabulary collapses, fallback to min_df=1 and unigrams only
    """
    stopwords = stopwords or set()

    keep_idx = [i for i, d in enumerate(docs) if d and d.strip()]
    docs_clean = [docs[i] for i in keep_idx]
    if not docs_clean:
        return [[] for _ in docs], []

    min_df_eff = 1 if len(docs_clean) < 30 else min_df

    token_pattern = r"(?u)\b[^\W\d_]{2,}\b"  # unicode letters, len>=2

    cv = CountVectorizer(
        lowercase=True,
        stop_words=list(stopwords) if stopwords else None,
        ngram_range=(1, 2),
        min_df=min_df_eff,
        max_df=max_df,
        token_pattern=token_pattern,
    )
    X = cv.fit_transform(docs_clean)

    # Fallback if empty vocab
    if X.shape[1] == 0:
        cv = CountVectorizer(
            lowercase=True,
            stop_words=None,
            ngram_range=(1, 1),
            min_df=1,
            max_df=0.95,
            token_pattern=token_pattern,
        )
        X = cv.fit_transform(docs_clean)

    if X.shape[1] == 0:
        return [[] for _ in docs], []

    k = min(n_topics, X.shape[0])
    if k < 2:
        tags_per_doc = [[] for _ in docs]
        for i in keep_idx:
            tags_per_doc[i] = ["lda_00: single_doc"]
        return tags_per_doc, ["single_doc"]

    lda = LatentDirichletAllocation(
        n_components=k,
        random_state=random_state,
        learning_method="batch",
        max_iter=50,
        evaluate_every=-1,
    )
    doc_topic = lda.fit_transform(X)  # shape: (n_docs, k)

    vocab = np.array(cv.get_feature_names_out())
    topic_labels: List[str] = []
    for t in range(k):
        top_idx = lda.components_[t].argsort()[-top_terms:][::-1]
        # reuse your label cleaner if you have it
        top_feats = vocab[top_idx].tolist()
        try:
            label = clean_topic_label(top_feats, max_items=top_terms)
        except NameError:
            label = " ".join(top_feats)
        topic_labels.append(label)

    # Assign top topic per doc
    tags_per_doc = [[] for _ in docs]
    for j, orig_i in enumerate(keep_idx):
        t = int(np.argmax(doc_topic[j]))
        tags_per_doc[orig_i] = [f"lda_{t:02d}: {topic_labels[t]}"]

    return tags_per_doc, topic_labels


# ----------------------------
# Storage: SQLite + exports
# ----------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS protests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    source_name TEXT,
    date_raw TEXT,
    date_iso TEXT,
    time_raw TEXT,
    author TEXT,
    link TEXT,
    title TEXT,
    text TEXT NOT NULL,
    tags_tfidf TEXT NOT NULL DEFAULT '',
    tag_bert TEXT NOT NULL DEFAULT '',
    tags_lda TEXT NOT NULL DEFAULT '',
    tags_hdbscan TEXT NOT NULL DEFAULT '',
    tags_umap_hdbscan TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    content_hash TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_protests_hash ON protests(content_hash);
CREATE INDEX IF NOT EXISTS idx_protests_date_iso ON protests(date_iso);
CREATE INDEX IF NOT EXISTS idx_protests_author ON protests(author);
"""

INSERT_SQL = """
INSERT INTO protests (
    source_file, source_name, date_raw, date_iso, time_raw, author, link, title, text, tags_tfidf, tag_bert, tags_lda, tags_hdbscan, tags_umap_hdbscan, tags, content_hash
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(content_hash) DO UPDATE SET
    source_file=excluded.source_file,
    source_name=excluded.source_name,
    date_raw=excluded.date_raw,
    date_iso=excluded.date_iso,
    time_raw=excluded.time_raw,
    author=excluded.author,
    link=excluded.link,
    title=excluded.title,
    text=excluded.text,
    tags_tfidf=excluded.tags_tfidf,
    tag_bert=excluded.tag_bert,
    tags_lda=excluded.tags_lda,
    tags_hdbscan=excluded.tags_hdbscan,
    tags_umap_hdbscan=excluded.tags_umap_hdbscan,
    tags=excluded.tags;
"""


def write_sqlite(db_path: Path, records: List[ProtestRecord]) -> Tuple[int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(SCHEMA_SQL)
        # --- lightweight schema migration (SQLite doesn't auto-add columns on CREATE IF NOT EXISTS) ---
        cur = con.cursor()
        existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(protests)").fetchall()}
        desired = {
            "tags_tfidf": "TEXT NOT NULL DEFAULT ''",
            "tag_bert": "TEXT NOT NULL DEFAULT ''",
            "tags_lda": "TEXT NOT NULL DEFAULT ''",
            "tags_hdbscan": "TEXT NOT NULL DEFAULT ''",
            "tags_umap_hdbscan": "TEXT NOT NULL DEFAULT ''",
        }
        for col, ddl in desired.items():
            if col not in existing_cols:
                cur.execute(f"ALTER TABLE protests ADD COLUMN {col} {ddl}")
        # ---------------------------------------------------------------------------

        before = cur.execute("SELECT COUNT(*) FROM protests").fetchone()[0]
        cur.executemany(
            INSERT_SQL,
            [
                (
                    r.source_file, r.source_name, r.date_raw, r.date_iso, r.time_raw,
                    r.author, r.link, r.title, r.text, r.tags_tfidf, r.tag_bert, r.tags_lda, r.tags_hdbscan, r.tags_umap_hdbscan, r.tags, r.content_hash
                )
                for r in records
            ],
        )
        con.commit()
        after = cur.execute("SELECT COUNT(*) FROM protests").fetchone()[0]
        return before, after
    finally:
        con.close()


def export_jsonl(path: Path, records: List[ProtestRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def export_csv(path: Path, records: List[ProtestRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def export_xlsx(path: Path, records: List[ProtestRecord]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "protests"

    rows = [asdict(r) for r in records]
    if not rows:
        wb.save(path)
        return

    headers = list(rows[0].keys())
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h) for h in headers])

    wb.save(path)

_BERT_MODEL = None
_KEYBERT = None

def init_keybert(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    Lazy init KeyBERT + SentenceTransformer.
    """
    global _BERT_MODEL, _KEYBERT
    if _KEYBERT is not None:
        return
    from sentence_transformers import SentenceTransformer
    from keybert import KeyBERT

    _BERT_MODEL = SentenceTransformer(model_name)
    _KEYBERT = KeyBERT(model=_BERT_MODEL)

def bert_keyword_tag(
    doc: str,
    top_n: int = 1,
    keyphrase_ngram_range: tuple[int, int] = (1, 3),
    use_mmr: bool = True,
    diversity: float = 0.6,
    stopwords: Optional[set] = None,
) -> str:
    """
    Returns a single BERT-derived keyword/phrase tag for a document.
    Uses KeyBERT, which extracts keyphrases by embedding similarity.
    """
    if not doc or not doc.strip():
        return ""

    init_keybert()

    # KeyBERT expects stop_words as list/str; we pass a list of Romanian stopwords you already maintain.
    sw = list(stopwords) if stopwords else None

    kws = _KEYBERT.extract_keywords(
        doc,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=sw,
        top_n=top_n,
        use_mmr=use_mmr,
        diversity=diversity,
    )

    # kws is list of (phrase, score)
    if not kws:
        return ""
    return kws[0][0]

_EMBEDDER = None

def tag_topics_embeddings_hdbscan(
    docs: List[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    min_cluster_size: int = 6,
    min_samples: Optional[int] = 1,
    metric: str = "cosine",
    top_terms: int = 8,
    stopwords: Optional[set] = None,
) -> Tuple[List[List[str]], Dict[int, str]]:
    stopwords = stopwords or set()

    keep_idx = [i for i, d in enumerate(docs) if d and d.strip()]
    docs_clean = [docs[i] for i in keep_idx]
    if not docs_clean:
        return [[] for _ in docs], {}

    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name)

    # embeddings; keep normalize_embeddings=True (works well with cosine)
    E = _EMBEDDER.encode(docs_clean, normalize_embeddings=True, show_progress_bar=True)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    clusters = clusterer.fit_predict(E)

    # Label clusters with TF-IDF terms
    token_pattern = r"(?u)\b[^\W\d_]{2,}\b"
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=list(stopwords) if stopwords else None,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        token_pattern=token_pattern,
    )
    X = vec.fit_transform(docs_clean)
    terms = np.array(vec.get_feature_names_out())

    cluster_label_map: Dict[int, str] = {}
    for cid in sorted(set(clusters)):
        if cid == -1:
            cluster_label_map[cid] = "outlier"
            continue
        idx = np.where(clusters == cid)[0]
        centroid = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = centroid.argsort()[-top_terms:][::-1]
        feats = terms[top_idx].tolist()
        label = clean_topic_label(feats, max_items=top_terms) if "clean_topic_label" in globals() else " ".join(feats)
        cluster_label_map[cid] = label

    tags_per_doc = [[] for _ in docs]
    for j, orig_i in enumerate(keep_idx):
        cid = int(clusters[j])
        if cid == -1:
            tags_per_doc[orig_i] = ["hdb_-1: outlier"]
        else:
            tags_per_doc[orig_i] = [f"hdb_{cid:02d}: {cluster_label_map.get(cid, 'unknown')}"]

    return tags_per_doc, cluster_label_map

def tag_topics_embeddings_umap_hdbscan(
    docs: List[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    umap_n_components: int = 10,
    umap_n_neighbors: int = 15,
    min_cluster_size: int = 8,
    min_samples: Optional[int] = 1,
    top_terms: int = 8,
    stopwords: Optional[set] = None,
    random_state: int = 42,
) -> Tuple[List[List[str]], Dict[int, str]]:


    stopwords = stopwords or set()
    keep_idx = [i for i, d in enumerate(docs) if d and d.strip()]
    docs_clean = [docs[i] for i in keep_idx]
    if not docs_clean:
        return [[] for _ in docs], {}

    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(model_name)

    E = _EMBEDDER.encode(docs_clean, normalize_embeddings=True, show_progress_bar=True)

    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        metric="cosine",
        random_state=random_state,
    )
    Z = reducer.fit_transform(E)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    clusters = clusterer.fit_predict(Z)

    # Label clusters with TF-IDF terms
    token_pattern = r"(?u)\b[^\W\d_]{2,}\b"
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=list(stopwords) if stopwords else None,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        token_pattern=token_pattern,
    )
    X = vec.fit_transform(docs_clean)
    terms = np.array(vec.get_feature_names_out())

    cluster_label_map: Dict[int, str] = {}
    for cid in sorted(set(clusters)):
        if cid == -1:
            cluster_label_map[cid] = "outlier"
            continue
        idx = np.where(clusters == cid)[0]
        centroid = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = centroid.argsort()[-top_terms:][::-1]
        feats = terms[top_idx].tolist()
        label = clean_topic_label(feats, max_items=top_terms) if "clean_topic_label" in globals() else " ".join(feats)
        cluster_label_map[cid] = label

    tags_per_doc = [[] for _ in docs]
    for j, orig_i in enumerate(keep_idx):
        cid = int(clusters[j])
        tags_per_doc[orig_i] = ["hdb_-1: outlier"] if cid == -1 else [f"hdb_{cid:02d}: {cluster_label_map.get(cid, 'unknown')}"]

    return tags_per_doc, cluster_label_map

# ----------------------------
# Main
# ----------------------------

def main():
    input_files = [
        Path("/Users/bogdanoancea/OneDrive/papers/2026/Paduri/Declic Proteste.docx"),
        Path("/Users/bogdanoancea/OneDrive/papers/2026/Paduri/Proteste Web.docx"),
    ]

    for fp in input_files:
        if not fp.exists():
            raise FileNotFoundError(f"Missing input file: {fp}")

    all_records: List[ProtestRecord] = []
    for fp in input_files:
        paragraphs = read_docx_paragraphs(fp)
        recs = parse_records_from_paragraphs(paragraphs, source_file=fp.name)
        all_records.extend(recs)

    print(f"Parsed records (raw): {len(all_records)}")

    # ----------------------------
    # TF-IDF tagging (global, once)
    # ----------------------------
    if all_records:
        docs = [build_doc_text(r.title, r.text) for r in all_records]
        # (A) TF-IDF topic tag
        tfidf_tags_per_doc, topic_labels = tag_topics_tfidf_kmeans_safe(
            docs, n_topics=12, stopwords=RO_STOPWORDS_MIN
        )
        # (B) BERT tag
        bert_tags = [""] * len(docs)
        try:
            for i, d in enumerate(docs):
                bert_tags[i] = bert_keyword_tag(
                    d,
                    top_n=1,
                    keyphrase_ngram_range=(1, 3),
                    use_mmr=True,
                    diversity=0.6,
                    stopwords=RO_STOPWORDS_MIN,
                )
        except Exception as e:
            print(f"[WARN] BERT tagging disabled (fallback to TF-IDF only): {e}")

        # (C) LDA topic tag
        lda_tags_per_doc, lda_topic_labels = tag_topics_lda_safe(
            docs, n_topics=12, stopwords=RO_STOPWORDS_MIN
        )
        print("LDA topic labels:")
        for i, lab in enumerate(lda_topic_labels):
            print(f"  lda_{i:02d}: {lab}")

        # (D) Embeddings + HDBSCAN tag
        hdb_tags_per_doc, hdb_label_map = tag_topics_embeddings_hdbscan(
            docs,
            min_cluster_size=6,  # tune this
            min_samples=1,  # or try 5–10 for stricter clusters
            metric= "cosine",
            stopwords=RO_STOPWORDS_MIN,
        )
        print("HDBSCAN clusters:", len([c for c in hdb_label_map.keys() if c != -1]), " (+ outliers)")

        # (E) Embeddings + HDBSCAN tag
        umap_hdb_tags_per_doc, umap_hdb_label_map = tag_topics_embeddings_umap_hdbscan(
            docs,
            umap_n_components=10,
            umap_n_neighbors=15,
            min_cluster_size=8,
            min_samples=1,
            stopwords=RO_STOPWORDS_MIN,
        )
        # Store all 5 tags (separate columns + combined JSON for backward compatibility)
        for rec, tfidf_tags, btag, lda_tags, hdb_tags, umap_hdb_tags in zip(
                all_records, tfidf_tags_per_doc, bert_tags, lda_tags_per_doc, hdb_tags_per_doc, umap_hdb_tags_per_doc
        ):
            # Per-method columns (human-friendly plain strings)
            rec.tags_tfidf = "; ".join(tfidf_tags) if tfidf_tags else ""
            rec.tag_bert = btag or ""
            rec.tags_lda = "; ".join(lda_tags) if lda_tags else ""
            rec.tags_hdbscan = "; ".join(hdb_tags) if hdb_tags else ""
            rec.tags_umap_hdbscan = "; ".join(umap_hdb_tags) if umap_hdb_tags else ""

            # Optional combined field (JSON array string)
            combined = []
            if rec.tags_tfidf:
                combined.append(f"TFIDF: {rec.tags_tfidf}")
            if rec.tag_bert:
                combined.append(f"BERT: {rec.tag_bert}")
            if rec.tags_lda:
                combined.append(f"LDA: {rec.tags_lda}")
            if rec.tags_hdbscan:
                combined.append(f"HDBSCAN: {rec.tags_hdbscan}")
            if rec.tags_umap_hdbscan:
                combined.append(f"UMAP_HDBSCAN: {rec.tags_umap_hdbscan}")
            rec.tags = json.dumps(combined, ensure_ascii=False)

        print("Topic labels:")
        for i, lab in enumerate(topic_labels):
            print(f"  topic_{i:02d}: {lab}")

    out_dir = Path("/Users/bogdanoancea/OneDrive/papers/2026/Paduri/output_proteste")
    out_dir.mkdir(exist_ok=True, parents=True)

    db_path = out_dir / "proteste.sqlite"
    before, after = write_sqlite(db_path, all_records)

    export_jsonl(out_dir / "proteste.jsonl", all_records)
    if all_records:
        export_csv(out_dir / "proteste.csv", all_records)
        export_xlsx(out_dir / "proteste.xlsx", all_records)

    print(f"SQLite rows before: {before}, after: {after} (dedup via content_hash)")
    print(f"Wrote: {db_path}")
    print(f"Wrote: {out_dir / 'proteste.jsonl'}")
    if all_records:
        print(f"Wrote: {out_dir / 'proteste.csv'}")
        print(f"Wrote: {out_dir / 'proteste.xlsx'}")


if __name__ == "__main__":
    main()