"""Покращення пошуку на рівні запиту: HyDE, reranker і поріг відмови.

Кожне покращення вмикається прапорцем, щоб можна було виміряти його внесок окремо.

Калібрування порогу відмови:
    python -m src.query --calibrate --inside eval/sample_dev.json --outside eval/sample_offtopic.json
"""
import argparse
import sys
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .llm import generate
from .profiling import Timings
from .retriever import get_retriever

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 50
# Пороги за замовчуванням, поки немає калібрування (файл thresholds.json у папці RAG_WORK).
# "dense" — косинус між питанням і найкращою анотацією, "rerank" — оцінка cross-encoder.
# Шкали різні, тому і пороги окремі.
DEFAULT_THRESHOLDS = {"dense": 0.60, "rerank": 0.0}

HYDE_SYSTEM = ("You write abstracts of computer science research papers. "
               "Reply with the abstract text only, in English, 4-6 sentences, no title.")
HYDE_PROMPT = "Write the abstract of a paper that answers this question:\n{query}"


def hyde_passage(query: str, model: str | None = None, backend: str | None = None) -> str:
    """HyDE (Gao et al., 2022): LLM пише гіпотетичну анотацію, яка відповідає на питання.

    Коротке питання і довга анотація лежать у просторі ембедінгів далеко одне від одного,
    а дві анотації — близько. Тому шукати за вектором вигаданої анотації точніше.
    Факти в ній можуть бути вигадані — це не страшно, вона потрібна лише для пошуку.
    """
    return generate(HYDE_PROMPT.format(query=query), system=HYDE_SYSTEM, model=model,
                    backend=backend, temperature=0.7, max_tokens=256)


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER, device="cpu")


def rerank(query: str, hits: list[dict], top_n: int) -> list[dict]:
    """Cross-encoder читає пару «питання — анотація» разом і оцінює її точніше за косинус.

    Дорожче за векторний пошук, тому застосовується лише до топ-50 кандидатів.
    Модель навчена на англійських запитах MS MARCO.
    """
    if not hits:
        return []
    scores = _cross_encoder().predict([(query, h["text"]) for h in hits], show_progress_bar=False)
    ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [{**h, "rerank_score": float(s)} for h, s in ranked[:top_n]]


def retrieve(query: str, k: int = 5, hyde: bool = False, use_rerank: bool = False,
             model: str | None = None, backend: str | None = None, timings: Timings | None = None):
    """Повертає (hits, info).

    info: mode ("dense" або "rerank"), relevance (число для порогу відмови),
    hyde_passage (текст гіпотетичної анотації або None).
    """
    timings = timings if timings is not None else Timings()
    retriever = get_retriever()
    passage = None

    with timings.stage("encode"):
        q_vec = retriever.encode([query], "query")
    if hyde:
        with timings.stage("hyde"):
            passage = hyde_passage(query, model=model, backend=backend)
        with timings.stage("encode"):
            p_vec = retriever.encode([passage], "passage")
            # Як у статті HyDE, усереднюємо вектор гіпотетичного документа з вектором питання:
            # якщо LLM пішла не в той бік, питання утримує пошук біля теми.
            vec = q_vec + p_vec
            vec = (vec / np.linalg.norm(vec, axis=1, keepdims=True)).astype(np.float32)
    else:
        vec = q_vec

    with timings.stage("search"):
        hits = retriever.search_vector(vec, RERANK_CANDIDATES if use_rerank else k)

    if use_rerank:
        with timings.stage("rerank"):
            hits = rerank(query, hits, k)
        relevance = hits[0]["rerank_score"] if hits else float("-inf")
        mode = "rerank"
    else:
        mode = "dense"
        if not hits:
            relevance = float("-inf")
        elif hyde:
            # score при HyDE — схожість із гібридним вектором; для порогу рахуємо
            # схожість саме питання з найкращою анотацією, щоб поріг був один.
            with timings.stage("encode"):
                relevance = float(q_vec[0] @ retriever.encode([hits[0]["text"]], "passage")[0])
        else:
            relevance = hits[0]["score"]

    return hits, {"mode": mode, "relevance": float(relevance), "hyde_passage": passage}


def thresholds_path() -> Path:
    return get_retriever().work / "thresholds.json"


def load_thresholds() -> dict:
    path = thresholds_path()
    if path.is_file():
        return {**DEFAULT_THRESHOLDS, **json.loads(path.read_text(encoding="utf-8"))["thresholds"]}
    return dict(DEFAULT_THRESHOLDS)


def is_out_of_corpus(info: dict, thresholds: dict | None = None) -> bool:
    """Відмова: найкраща знайдена анотація недостатньо схожа на питання."""
    thresholds = thresholds or load_thresholds()
    return info["relevance"] < thresholds[info["mode"]]


def best_threshold(inside: list[float], outside: list[float]) -> tuple[float, float]:
    """Поріг, що найкраще розділяє питання «в корпусі» і «поза корпусом» (balanced accuracy)."""
    values = sorted(set(inside) | set(outside))
    candidates = [(a + b) / 2 for a, b in zip(values, values[1:])] or values
    best = (float("-inf"), 0.0)
    for t in candidates:
        tpr = sum(v >= t for v in inside) / len(inside)
        tnr = sum(v < t for v in outside) / len(outside)
        score = (tpr + tnr) / 2
        # За рівної точності беремо менший поріг: краще відповісти, ніж зайво відмовити.
        if score > best[1] + 1e-9:
            best = (t, score)
    return best


def calibrate(inside_file: Path, outside_file: Path, write: bool = True) -> dict:
    def load(path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return [item["question"] if isinstance(item, dict) else item for item in data]

    inside, outside = load(inside_file), load(outside_file)
    report = {"inside_file": str(inside_file), "outside_file": str(outside_file),
              "inside": len(inside), "outside": len(outside), "thresholds": {}, "balanced_accuracy": {},
              "relevance": {}}
    for mode, use_rerank in (("dense", False), ("rerank", True)):
        rel_in = [retrieve(q, k=5, use_rerank=use_rerank)[1]["relevance"] for q in inside]
        rel_out = [retrieve(q, k=5, use_rerank=use_rerank)[1]["relevance"] for q in outside]
        threshold, accuracy = best_threshold(rel_in, rel_out)
        report["thresholds"][mode] = round(threshold, 4)
        report["balanced_accuracy"][mode] = round(accuracy, 4)
        report["relevance"][mode] = {"inside": [round(v, 4) for v in rel_in],
                                     "outside": [round(v, 4) for v in rel_out]}
        print(f"{mode:6}: в корпусі min={min(rel_in):.3f} median={np.median(rel_in):.3f} | "
              f"поза корпусом max={max(rel_out):.3f} median={np.median(rel_out):.3f} | "
              f"поріг={threshold:.3f}, balanced accuracy={accuracy:.2%}")
    if write:
        thresholds_path().write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Збережено: {thresholds_path()}")
    return report


def main():
    # Консоль Windows за замовчуванням не UTF-8 — без цього українські літери ламаються.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate", action="store_true", help="підібрати поріг відмови")
    p.add_argument("--inside", type=Path, default=Path("eval/sample_dev.json"))
    p.add_argument("--outside", type=Path, default=Path("eval/sample_offtopic.json"))
    p.add_argument("--dry-run", action="store_true", help="не записувати thresholds.json")
    args = p.parse_args()
    if not args.calibrate:
        p.error("наразі підтримується лише --calibrate")
    calibrate(args.inside, args.outside, write=not args.dry_run)


if __name__ == "__main__":
    main()
