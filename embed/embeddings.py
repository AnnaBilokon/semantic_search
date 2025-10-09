from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class EmbeddingConfig:
    provider: str
    model: str


class Embedder:
    def __init__(self, cfg: EmbeddingConfig):

        # Store the config; the instance “remembers” provider/model.
        self.cfg = cfg

        if cfg.provider == 'hf':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(cfg.model)
        elif cfg.provider == 'openai':
            from openai import OpenAI
            self.client = OpenAI()
        else:
            raise ValueError(f'Uknown provider')

    def encode(self, texts: List[str]) -> np.ndarray:
        if self. cfg.provider == 'hf':
            arr = self.model.encode(texts, normalize_embeddings=True)
            arr = arr.astype('float32')
        else:
            resp = self.model.embeddings.create(
                model=self.cfg.model, input=texts)
            vecs = [d.embedding for d in resp.data]
            arr = np.asarray(vecs, dtype='float32')

            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = (arr / norms).astype('float32')
        return arr
