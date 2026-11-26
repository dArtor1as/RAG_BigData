"""Student 3/4 integration: from src.retriever import search."""
import argparse
import json
import os
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import faiss

from .common import DEFAULT_WORK, INDEXES, MODELS, encode_texts, load_encoder, positive, read_json, selected_model
from .corpus import catalog_info, connect_catalog


def index_file(work, model, kind):
    corpus = catalog_info(work)
    folder = Path(work) / model / "indexes" / kind
    info = read_json(folder / "build.json")
    if (info["source_sha256"] != corpus["source_sha256"] or info["rows"] != corpus["rows"]
            or info["encoder"] != MODELS[model] or info["index"] != kind or info["model"] != model):
        raise ValueError("This index belongs to a different corpus or encoder.")
    path = folder / "index.faiss"
    if path.stat().st_size != info["bytes"]:
        raise ValueError(f"Index file is incomplete: {path}")
    return path, info


def open_index(work, model, kind):
    path, info = index_file(work, model, kind)
    index = faiss.read_index(str(path))
    if (index.ntotal != info["rows"] or index.d != info["dimension"]
            or index.metric_type != faiss.METRIC_INNER_PRODUCT):
        raise ValueError("The index binary does not match its metadata.")
    return index


class Retriever:
    def __init__(self, work=DEFAULT_WORK, model=None, index="hnsw", device="cpu"):
        self.work = Path(work)
        self.model = model or selected_model(work)
        self.index = open_index(self.work, self.model, index)
        self.encoder = load_encoder(self.model, device)

    def encode(self, texts: list[str], side: str = "query"):
        """Кодує тексти тим самим енкодером, що й індекс.

        side="query" додає префікс запиту моделі (для BGE — інструкцію пошуку),
        side="passage" кодує текст як документ корпусу. HyDE і перевірка цитат
        працюють саме з passage, бо порівнюють текст з анотаціями.
        """
        if side not in ("query", "passage"):
            raise ValueError("side must be 'query' or 'passage'.")
        return encode_texts(self.encoder, [t.strip() for t in texts], self.model, side, 32)

    def search(self, query: str, k: int) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The query must be a nonempty string.")
        return self.search_vector(self.encode([query], "query"), k)

    def search_vector(self, vector, k: int) -> list[dict]:
        """Пошук за готовим нормалізованим вектором форми (1, d); формат результату як у search()."""
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(vector, k)
        if isinstance(self.index, faiss.IndexIVFPQ):
            # A large k can exhaust the probed lists. Per-call parameters keep other queries independent.
            probes = self.index.nprobe
            while ids[0, -1] < 0 and probes < self.index.nlist:
                probes = min(self.index.nlist, 2 * probes)
                params = faiss.SearchParametersIVF()
                params.nprobe = probes
                scores, ids = self.index.search(vector, k, params=params)
        ranked = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        papers = {}
        with closing(connect_catalog(self.work)) as db:
            for start in range(0, len(ranked), 500):
                keys = [i for i, _ in ranked[start:start + 500]]
                marks = ",".join("?" for _ in keys)
                sql = f"SELECT row_id, arxiv_id, title, text FROM papers WHERE row_id IN ({marks})"
                for i, arxiv_id, title, text in db.execute(sql, keys):
                    papers[i] = {"row_id": i, "arxiv_id": arxiv_id, "title": title, "text": text}
        return [{**papers[i], "score": score} for i, score in ranked]


@lru_cache(maxsize=1)
def _default(work, model, index):
    return Retriever(work, model, index, device="cpu")


def get_retriever() -> Retriever:
    """Кешований Retriever за змінними RAG_WORK, RAG_MODEL, RAG_INDEX — той самий, що в search()."""
    work = Path(os.environ.get("RAG_WORK", DEFAULT_WORK))
    model = os.environ.get("RAG_MODEL") or selected_model(work)
    return _default(work.resolve(), model, os.environ.get("RAG_INDEX", "hnsw"))


def search(query: str, k: int) -> list[dict]:
    """Return ranked row_id, arxiv_id, title, text and score records."""
    return get_retriever().search(query, k)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query")
    p.add_argument("-k", type=positive, default=5)
    p.add_argument("--work", type=Path, default=Path(os.environ.get("RAG_WORK", DEFAULT_WORK)))
    p.add_argument("--model", choices=MODELS, default=os.environ.get("RAG_MODEL"))
    p.add_argument("--index", choices=INDEXES, default=os.environ.get("RAG_INDEX", "hnsw"))
    args = p.parse_args()
    hits = Retriever(args.work, args.model, args.index).search(args.query, args.k)
    print(json.dumps(hits, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
