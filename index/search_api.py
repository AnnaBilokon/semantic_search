# Web framework + query parameters
from fastapi import FastAPI, Query
# Response model validation
from pydantic import BaseModel
import faiss                                          # Vector search engine
# Load metadata from Parquet
import pyarrow.parquet as pq
import numpy as np                                    # Numerical operations

from embed.embeddings import Embedder, EmbeddingConfig


app = FastAPI(title='Mini Semantic Search')


INDEX_PATH = 'data/faiss.index'
PARQUET = 'data/meta.parquet'
index = faiss.read_index(INDEX_PATH)
meta = pq.read_table(PARQUET).to_pydict()


embedder = Embedder(
    EmbeddingConfig(
        provider='hf', model='sentence-transformers/all-MiniLM-L6-v2')
)


class SearchResponseItem(BaseModel):
    score: float
    id: str
    text: str
    path: str
    title: str
    product: str
    version: str
    lang: str


@app.get('/search', response_model=list[SearchResponseItem])
def search(
    q: str = Query(..., description="User query"),
    k: int = 5,
    product: str | None = None,
    version: str | None = None,
    lang: str | None = None
):

    qv = embedder.encode([q]).astype('float32')  # (1, D)

    overfetch = max(k, 20)
    D, I = index.search(qv, overfetch)  # (1, overfetch)

    results = []
    for d, i in zip(D[0], I[0]):
        if i < 0:
            continue
        row = {k: meta[k][i] for k in meta}
        if product and row['product'] != product:
            continue
        if version and row['version'] != version:
            continue
        if lang and row['lang'] != lang:
            continue
        results.append(SearchResponseItem(
            score=float(d),
            id=row['id'], text=row['text'],
            path=row['path'], title=row['title'], product=row['product'],
            version=row['version'], lang=row['lang']
        ))
        if len(results) >= k:
            break
    return results
