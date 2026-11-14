"""Search an organization's HR documents: the retrieval half of week-03/ask.py, without Claude.

Each organization folder (see orgs.py) has its own hr-docs/ and its own database. Each HTML document is split into one chunk per section heading. Every chunk starts with the
document title and section heading, the source file and the document's version line, so it makes
sense on its own. Chunks are stored in a Chroma database in week-04/chroma_db/<organization folder
name>/ (gitignored) using Chroma's built-in embedding model. Building the database makes no API calls, so it is rebuilt automatically
whenever the documents change.
"""

import contextlib
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from chromadb.config import Settings

HERE = Path(__file__).resolve().parent
DB_ROOT = HERE / "chroma_db"
COLLECTION = "hr-docs"
TOP_K = 4
WEAK_MATCH = 0.4  # below this, a chunk is unlikely to answer the question (see the Week 3 runs)
SUPPORTED = {".html", ".htm"}


@dataclass
class Section:
    heading: str
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class Document:
    title: str
    intro: str  # short description under the title, e.g. "Version 6.1, current. Owner: Human Resources."
    sections: list[Section]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def join_sentences(parts) -> str:
    """Join separate lines of text, adding a period where a line has no closing punctuation."""
    out = [clean(p) for p in parts if clean(p)]
    return " ".join(p if p[-1] in ".:;!?" else p + "." for p in out)


def read_html(path: Path) -> Document:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    # The footer only says the company is fictional, so it is left out rather than made a chunk.
    for tag in soup(["script", "style", "noscript", "nav", "template", "footer"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup

    h1 = root.find("h1")
    page_title = clean(soup.title.get_text()) if soup.title else ""
    title = clean(h1.get_text()) if h1 else (page_title.split("|")[0].strip() or path.stem.replace("-", " ").title())

    intro: list[str] = []
    sections: list[Section] = []
    current: Section | None = None
    parent_h2 = ""
    for el in root.find_all(["h2", "h3", "h4", "p", "li", "tr", "pre", "dt", "dd", "blockquote"]):
        # Skip blocks nested inside another block we already read (e.g. a <p> inside an <li>).
        if el.find_parent(["li", "tr", "pre", "dd", "blockquote"]):
            continue
        if el.name in ("h2", "h3", "h4"):
            text = clean(el.get_text(" "))
            if el.name == "h2":
                parent_h2 = text
            current = Section(text if el.name == "h2" or not parent_h2 else f"{parent_h2} > {text}")
            sections.append(current)
            continue
        if el.name == "li":
            text = "- " + clean(el.get_text(" "))
        elif el.name == "tr":
            text = " | ".join(clean(c.get_text(" ")) for c in el.find_all(["th", "td"]))
        else:
            text = clean(el.get_text(" "))
        if not text.strip("- |"):
            continue
        (intro if current is None else current.paragraphs).append(text)

    return Document(title, join_sentences(intro), sections)


def make_chunks(path: Path) -> list[dict]:
    doc = read_html(path)
    chunks = []
    for i, s in enumerate(sec for sec in doc.sections if sec.paragraphs):
        lines = [f"{doc.title} > {s.heading}", f"Source file: {path.name}"]
        if doc.intro and len(doc.intro) <= 300:
            lines.append(f"About this document: {doc.intro}")
        text = "\n".join(lines + [""] + s.paragraphs)
        chunks.append({"id": f"{path.name}::{i}", "text": text,
                       "meta": {"source": path.name, "title": doc.title, "section": s.heading}})
    return chunks


def docs_dir(org: Path) -> Path:
    return org / "hr-docs"


def db_dir(org: Path) -> Path:
    """Each organization gets its own database folder, named after its folder."""
    return DB_ROOT / org.name


def doc_files(org: Path) -> list[Path]:
    return sorted(p for p in docs_dir(org).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def fingerprint(org: Path, files: list[Path]) -> str:
    """Changes whenever a document is added, removed or edited."""
    h = hashlib.sha256()
    h.update(f"{org}\n".encode())  # a different folder with the same name gets a rebuild, not stale results
    for p in files:
        h.update(f"{p.relative_to(docs_dir(org))}|{p.stat().st_size}|{p.stat().st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def log(message: str) -> None:
    # stdout carries the MCP messages, so anything for a person goes to stderr.
    print(message, file=sys.stderr)


def open_db(org: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(db_dir(org)), settings=Settings(anonymized_telemetry=False))


def build(org: Path) -> int:
    """(Re)build an organization's database from its documents folder and return the number of chunks."""
    files = doc_files(org)
    if not files:
        raise RuntimeError(f"No HTML documents found in {docs_dir(org)}")
    chunks = [c for p in files for c in make_chunks(p)]
    db = open_db(org)
    if COLLECTION in [c.name for c in db.list_collections()]:
        db.delete_collection(COLLECTION)
    collection = db.create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},  # so similarity = 1 - distance
        metadata={"fingerprint": fingerprint(org, files)},
    )
    with contextlib.redirect_stdout(sys.stderr):  # keep any library output off the MCP channel
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["meta"] for c in chunks],
        )
    log(f"Built the {org.name} HR database: {len(chunks)} chunks from {len(files)} documents, "
        f"saved in {db_dir(org).relative_to(HERE)}/.")
    return len(chunks)


def get_collection(org: Path):
    """Open the organization's database, building or rebuilding it first if it is missing or out of date."""
    db = open_db(org)
    names = [c.name for c in db.list_collections()]
    if COLLECTION not in names:
        log(f"No {org.name} HR database yet. Building it now…")
        build(org)
    elif (db.get_collection(COLLECTION).metadata or {}).get("fingerprint") != fingerprint(org, doc_files(org)):
        log(f"The {org.name} HR documents changed since the database was built. Rebuilding it now…")
        build(org)
    return open_db(org).get_collection(COLLECTION)


def search(collection, query: str, k: int = TOP_K) -> list[dict]:
    with contextlib.redirect_stdout(sys.stderr):
        res = collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
    return [
        {
            "rank": rank,
            "similarity": round(1 - dist, 3),
            "source_file": meta["source"],
            "section": f"{meta['title']} > {meta['section']}",
            "weak_match": (1 - dist) < WEAK_MATCH,
            "text": doc,
        }
        for rank, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1)
    ]
