from pathlib import Path
import numpy as np
import faiss
import pyarrow.parquet as pq
import csv

DATA_DIR = Path('data')
INDEX_PATH = DATA_DIR / 'faiss.index'
PARQUET = DATA_DIR / 'meta.parquet'
VECTORS = DATA_DIR / 'vectors.npy'
OUT_CSV = DATA_DIR / 'reuse_candidates.csv'

K = 10
THRESH = 0.78
LANG_SCOPE = None  # 'en' to limit to English only
DIFFERENT_DOCS = True  # only consider chunks from different source documents


def main():
    index = faiss.read_index(str(INDEX_PATH))
    meta = pq.read_table(PARQUET).to_pydict()
    X = np.load(VECTORS).astype('float32')

    N = X.shape[0]

    BATCH = 2048
    pairs = set()  # use a set of tuple keys to avoid duplicates (a,b) and (b,a)
    rows = []

    for start in range(0, N, BATCH):
        end = min(N, start + BATCH)
        # neighbors for this slice. I: indices of nearest neighbors; D: similarity scores (inner product == cosine since vectors are normalized).
        D, I = index.search(X[start:end], K)

        for row_idx in range(end - start):
            # We calculate the absolute index of the current chunk in the full dataset. start is the beginning index of the batch (e.g. 0, 2048, 4096…).row_idx is the position inside this batch (0 to batch size).Adding them gives us the true index i of the chunk in the entire dataset.
            i = start + row_idx
            for j in range(1, K):  # we skip j=0 and only look at 1...K-1.
                nb = I[row_idx, j]
                if nb < 0:
                    continue  # no more neighbors
                sim = float(D[row_idx, j])
                if sim < THRESH:
                    continue  # too low similarity
                if LANG_SCOPE and meta["lang"][i] != LANG_SCOPE:
                    continue
                if LANG_SCOPE and meta["lang"][nb] != LANG_SCOPE:
                    continue
                if DIFFERENT_DOCS and meta["doc_id"][i] == meta["doc_id"][nb]:
                    continue

                key = tuple(sorted([meta["id"][i], meta["id"][nb]]))
                if key in pairs:
                    continue  # already recorded (a,b) or (b,a)
                pairs.add(key)

                rows.append({
                    "sim": round(sim, 4),
                    "id_a": meta["id"][i],
                    "id_b": meta["id"][nb],
                    "doc_a": meta["doc_id"][i],
                    "doc_b": meta["doc_id"][nb],
                    "title_a": meta["title"][i],
                    "title_b": meta["title"][nb],
                    "path_a": meta["path"][i],
                    "path_b": meta["path"][nb],
                    "lang_a": meta["lang"][i],
                    "lang_b": meta["lang"][nb],
                    "version_a": meta["version"][i],
                    "version_b": meta["version"][nb],
                    "product_a": meta["product"][i],
                    "product_b": meta["product"][nb],
                    "text_a": meta["text"][i][:200],
                    "text_b": meta["text"][nb][:200],
                })

                # sorts your entire list of reuse candidates so the highest similarity (strongest matches) appear first.
    rows.sort(key=lambda r: r["sim"], reverse=True)

    fieldnames = [
        "sim", "id_a", "id_b", "doc_a", "doc_b", "title_a", "title_b", "path_a", "path_b",
        "lang_a", "lang_b", "version_a", "version_b", "product_a", "product_b", "text_a", "text_b"
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[reuse] DONE. candidates >= {THRESH}: {len(rows)}")
    print(f"[reuse] CSV → {OUT_CSV.resolve()}")


if __name__ == '__main__':
    main()
