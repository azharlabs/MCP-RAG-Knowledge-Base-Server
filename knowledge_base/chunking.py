import re
from typing import Dict, List

from knowledge_base.runtime import client
from knowledge_base.settings import MODEL_NAME


def _normalize_chunk_settings(chunk_size: int, overlap: int) -> tuple[int, int]:
    size = max(1, int(chunk_size or 0))
    ov = max(0, int(overlap or 0))
    if ov >= size:
        ov = max(0, size - 1)
    return size, ov


def _split_sentences(text: str, sentence_split_regex: str = r"(?<=[.!?])\s+") -> List[str]:
    sentences = [s.strip() for s in re.split(sentence_split_regex, text) if s.strip()]
    return sentences


def _group_sentences_by_size(sentences: List[str], max_chars: int) -> List[str]:
    if not sentences:
        return []
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent) + (1 if current else 0)
        if current and current_len + sent_len > max_chars:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def _fixed_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return [c for c in chunks if c]


def _overlapping_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    return _fixed_chunking(text, chunk_size, overlap)


def _semantic_chunking(text: str, chunk_size: int) -> List[str]:
    sentences = _split_sentences(text)
    return _group_sentences_by_size(sentences, max_chars=chunk_size)


def _recursive_character_chunking(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    split_at = text.rfind(" ", 0, max_chars)
    if split_at == -1:
        split_at = max_chars
    head = text[:split_at].strip()
    tail = text[split_at:].strip()
    return ([head] if head else []) + _recursive_character_chunking(tail, max_chars=max_chars)


def _agentic_chunking(text: str, task: str = "rag_qa") -> List[str]:
    prompt = f"""
You are splitting a document into chunks for task={task}.
Return chunks separated by a line containing only: ---
Rules:
- Keep chunks semantically coherent
- Prefer headings, paragraphs, and section boundaries
- Keep each chunk roughly 200-500 tokens if possible
TEXT:
{text}
""".strip()
    resp = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    content = resp.choices[0].message.content or ""
    chunks = [c.strip() for c in content.split("\n---\n") if c.strip()]
    return chunks


def _advanced_semantic_chunking(text: str, chunk_size: int) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    num_clusters = max(1, round(len(text) / max(1, chunk_size)))
    num_clusters = max(1, min(len(sentences), num_clusters))
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
    except Exception:
        return _semantic_chunking(text, chunk_size)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    buckets: Dict[int, List[str]] = {}
    for sent, label in zip(sentences, labels):
        buckets.setdefault(int(label), []).append(sent)
    return [" ".join(buckets[k]) for k in sorted(buckets.keys())]


def _context_enriched_chunking(text: str, overlap: int, chunk_size: int) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    ratio = overlap / max(1, chunk_size)
    window_size = max(1, min(3, int(round(ratio * 3)) or 1))
    chunks = []
    for i in range(len(sentences)):
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)
        chunks.append(" ".join(sentences[start:end]))
    return chunks


def _paragraph_chunking(text: str, chunk_size: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []
    chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para) + (2 if current else 0)
        if current and current_len + para_len > chunk_size:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += para_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _recursive_sentence_chunking(text: str, chunk_size: int) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    max_sentences = max(1, min(10, int(round(chunk_size / 200)) or 1))
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks


def _token_based_chunking(text: str, token_limit: int) -> List[str]:
    if token_limit <= 0:
        raise ValueError("token_limit must be > 0")
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("o200k_base")
        tokenizer = enc.encode
        detokenizer = enc.decode
    except Exception:
        tokenizer = None
        detokenizer = None
    if tokenizer is None or detokenizer is None:
        words = text.split()
        chunks = [" ".join(words[i:i + token_limit]) for i in range(0, len(words), token_limit)]
        return [c for c in chunks if c.strip()]
    toks = tokenizer(text)
    out = []
    for i in range(0, len(toks), token_limit):
        out.append(detokenizer(toks[i:i + token_limit]).strip())
    return [c for c in out if c]


def chunk_text(text: str, chunk_size: int, overlap: int, chunking_method: str = "fixed") -> List[str]:
    size, ov = _normalize_chunk_settings(chunk_size, overlap)
    method = (chunking_method or "fixed").strip().lower()
    if method == "fixed":
        return _fixed_chunking(text, size, ov)
    if method == "overlapping":
        return _overlapping_chunking(text, size, ov)
    if method == "semantic":
        return _semantic_chunking(text, size)
    if method == "recursive_character":
        return _recursive_character_chunking(text, max_chars=size)
    if method == "agentic":
        try:
            return _agentic_chunking(text)
        except Exception:
            return _semantic_chunking(text, size)
    if method == "advanced_semantic":
        return _advanced_semantic_chunking(text, size)
    if method == "context_enriched":
        return _context_enriched_chunking(text, ov, size)
    if method == "paragraph":
        return _paragraph_chunking(text, size)
    if method == "recursive_sentence":
        return _recursive_sentence_chunking(text, size)
    if method == "token_based":
        return _token_based_chunking(text, size)
    return _fixed_chunking(text, size, ov)
