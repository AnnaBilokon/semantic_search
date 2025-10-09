import faiss
# pyarrow / parquet — columnar table & file format to store metadata for each chunk (id, title, lang, version…).
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

from ingest.parse_xml import extract_chunks
from embed.embeddings import Embedder, EmbeddingConfig


DATA_DIR = Path('data')
XML_DIR = DATA_DIR / 'xml_export'
PARQUET = DATA_DIR / 'meta.parquet'
INDEX = DATA_DIR / 'faiss.index'

META_DEFAULTS = {'lang': 'en', 'product': 'AcmeX', 'version': 'v3.2'}


def main():
    # 1) Parse XML into chunk dicts
    chunks = extract_chunks(XML_DIR, META_DEFAULTS)
    # At this point, chunks[i] contains metadata (id, path, lang…) and texts[i] is the plain string we’ll embed.
    texts = [c['text'] for c in chunks]

# 2) Compute embeddings (local HF model)
    embedder = Embedder(EmbeddingConfig(
        provider="hf", model="sentence-transformers/all-MiniLM-L6-v2"))
    X = embedder.encode(texts)  # (N, D) normalized float32

   # 3) Build FAISS index for cosine similarity via inner product
    d = X.shape[1]  # embedding dimension.
    index = faiss.IndexFlatIP(d)
    index.add(X)
    faiss.write_index(index, str(INDEX))

    # 4) Store metadata in Parquet table
    table = pa.Table.from_pylist(chunks)
    pq.write_table(table, PARQUET)

    print(f"Build index with {len(chunks)} chunks, dim-{d}")


if __name__ == '__main__':
    main()
