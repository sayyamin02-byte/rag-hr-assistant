"""
STEP 1 — Ingestion & Chunking.

Why chunk at all? An embedding model turns a piece of text into ONE vector.
If we embed a whole 5-page document as one vector, the meaning gets averaged
into mush and retrieval can't find the one paragraph that answers the question.
So we split the document into small, self-contained pieces ("chunks"), embed
each one, and retrieve only the chunks that match the query.

Two knobs to understand:
  - chunk_size : how big each piece is. Too small = loses context; too big =
                 dilutes the embedding and adds noise.
  - overlap    : we repeat a little text between neighbouring chunks so an idea
                 split across a boundary still appears whole in at least one chunk.

We also keep METADATA (which section a chunk came from). Real RAG systems use
metadata to filter (e.g. only "Sick Leave") and to cite the source.
"""
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    section: str
    chunk_id: int
    meta: dict = field(default_factory=dict)


def load_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_sections(md_text: str):
    """Document-aware split: break the handbook on its '## Heading' lines so each
    section stays together. This is 'semantic/structural' chunking — smarter than
    blind fixed-size cuts because it respects the document's own structure."""
    sections = []
    current_title, current_lines = "Intro", []
    for line in md_text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[3:].strip()
            current_lines = []
        elif line.startswith("# "):
            continue  # skip the document title
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(t, b) for t, b in sections if b]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """Fixed-size, character-based, OVERLAPPING chunking within a section.
    (Our sections are short, so most become a single chunk — but this is the
    exact mechanic you'd apply to long documents.)"""
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap  # step back by `overlap` so pieces share context
    return chunks


def build_chunks(path: str, chunk_size: int = 500, overlap: int = 100):
    """Full ingestion: file -> sections -> overlapping chunks (with metadata)."""
    md = load_markdown(path)
    chunks, cid = [], 0
    for section, body in split_sections(md):
        for piece in chunk_text(body, chunk_size, overlap):
            chunks.append(Chunk(text=piece, section=section, chunk_id=cid,
                                meta={"section": section}))
            cid += 1
    return chunks


if __name__ == "__main__":
    cs = build_chunks("data/hr_policy.md")
    print(f"Built {len(cs)} chunks:")
    for c in cs:
        print(f"  [{c.chunk_id}] ({c.section}) {c.text[:70]}...")
