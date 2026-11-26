"""Порівняння режимів пошуку: на якому місці опиняється стаття, з якої зроблено питання.

    python -m src.eval_modes --questions eval/sample_dev.json eval/dev_full.json --out results/full

Для кожного питання рахує місце цільової статті серед 50 кандидатів у чотирьох режимах
(base, +rerank, +hyde, +hyde+rerank). Порожнє місце означає, що статті немає в топ-50.
Пише eval_modes.csv (по питаннях) і eval_modes.md (підсумок: Recall@1, Recall@5, MRR@5).
"""
import argparse
import csv
import json
from pathlib import Path

from .llm import default_backend, default_model
from .query import RERANK_CANDIDATES, hyde_passage, hyde_vector, rerank
from .retriever import get_retriever

MODES = ("base", "rerank", "hyde", "hyde_rerank")


def _rank(ids: list[str], gold: str) -> int | None:
    return ids.index(gold) + 1 if gold in ids else None


def _ranks(question: str, hits: list[dict], gold: str) -> tuple[int | None, int | None]:
    """Місце за косинусом і місце після reranker'а серед тих самих кандидатів."""
    reranked = rerank(question, hits, len(hits))
    return _rank([h["arxiv_id"] for h in hits], gold), _rank([h["arxiv_id"] for h in reranked], gold)


def ranks_for(question: str, gold: str, model: str, backend: str) -> dict:
    retriever = get_retriever()
    q_vec = retriever.encode([question], "query")
    hits = retriever.search_vector(q_vec, RERANK_CANDIDATES)
    base, base_rr = _ranks(question, hits, gold)

    passage = hyde_passage(question, model=model, backend=backend)
    vec = hyde_vector(q_vec, retriever.encode([passage], "passage"))
    hyde, hyde_rr = _ranks(question, retriever.search_vector(vec, RERANK_CANDIDATES), gold)
    return {"relevance": round(hits[0]["score"], 4), "base": base, "rerank": base_rr,
            "hyde": hyde, "hyde_rerank": hyde_rr}


def summary(rows: list[dict]) -> list[dict]:
    out = []
    for mode in MODES:
        ranks = [r[mode] for r in rows]
        out.append({"mode": mode,
                    "recall_at_1": sum(x == 1 for x in ranks if x),
                    "recall_at_5": sum(x <= 5 for x in ranks if x),
                    "mrr_at_5": sum(1 / x for x in ranks if x and x <= 5) / len(rows)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", type=Path, nargs="+", default=[Path("eval/sample_dev.json")])
    p.add_argument("--model", default=None)
    p.add_argument("--backend", default=None)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()
    backend = args.backend or default_backend()
    model = args.model or default_model(backend)

    rows = []
    for path in args.questions:
        for item in json.loads(path.read_text(encoding="utf-8")):
            row = {"source": path.name, "question": item["question"], "arxiv_id": item["arxiv_id"],
                   **ranks_for(item["question"], item["arxiv_id"], model, backend)}
            rows.append(row)
            print(f"{len(rows):3} " + " ".join(f"{m}={row[m] or '-'}" for m in MODES), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "eval_modes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "question", "arxiv_id", "relevance", *MODES])
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    lines = [f"Місце цільової статті, {n} питань ({', '.join(p.name for p in args.questions)}), "
             f"HyDE: {model} ({backend}). HyDE пише анотацію з температурою 0.7, тож від запуску "
             f"до запуску числа трохи відрізняються.", "",
             "| Пошук | Recall@1 | Recall@5 | MRR@5 |", "| --- | ---: | ---: | ---: |"]
    for s in summary(rows):
        lines.append(f"| {s['mode']} | {s['recall_at_1']}/{n} | {s['recall_at_5']}/{n} | {s['mrr_at_5']:.3f} |")
    (args.out / "eval_modes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
