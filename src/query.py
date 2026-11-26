"""Покращення пошуку на рівні запиту: HyDE, reranker і дворівневий поріг відмови.

Кожне покращення вмикається прапорцем, щоб можна було виміряти його внесок окремо.

Калібрування порогів (вибірка і повний корпус відповідно):
    python -m src.query --calibrate
    python -m src.query --calibrate --inside eval/sample_dev.json eval/dev_full.json
"""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .llm import generate
from .profiling import Timings
from .retriever import get_retriever

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 50
# Пороги за косинусом питання з найкращою анотацією, якщо немає thresholds.json у RAG_WORK:
# нижче low — відмова без LLM, між low і high — відповідь із попередженням.
# Оцінка cross-encoder для цього не годиться: для правильної статті вона буває −9…−10.
DEFAULT_THRESHOLDS = {"low": 0.60, "high": 0.70}
LOW_MARGIN = 0.03  # запас нижнього порогу під найслабшим питанням з корпусу
HIGH_PERCENTILE = 95  # верхній поріг: вище нього лише 5% питань поза корпусом

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


def hyde_vector(q_vec: np.ndarray, p_vec: np.ndarray) -> np.ndarray:
    """Як у статті HyDE, усереднюємо вектор гіпотетичного документа з вектором питання:
    якщо LLM пішла не в той бік, питання утримує пошук біля теми."""
    vec = q_vec + p_vec
    return (vec / np.linalg.norm(vec, axis=1, keepdims=True)).astype(np.float32)


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

    info: relevance (косинус питання з найкращою анотацією, для порогу відмови),
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
            vec = hyde_vector(q_vec, retriever.encode([passage], "passage"))
    else:
        vec = q_vec

    with timings.stage("search"):
        hits = retriever.search_vector(vec, RERANK_CANDIDATES if use_rerank else k)

    if not hits:
        relevance = float("-inf")
    elif hyde:
        # Поріг калібрувався на схожості самого питання, а при HyDE score рахується для
        # гібридного вектора. Тому окремо шукаємо найкращу анотацію для питання (~20 мс).
        with timings.stage("search"):
            relevance = retriever.search_vector(q_vec, 1)[0]["score"]
    else:
        relevance = hits[0]["score"]

    if use_rerank:
        with timings.stage("rerank"):
            hits = rerank(query, hits, k)

    return hits, {"relevance": float(relevance), "hyde_passage": passage}


def thresholds_path() -> Path:
    return get_retriever().work / "thresholds.json"


def load_thresholds() -> dict:
    path = thresholds_path()
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"low": data["low"], "high": data["high"]}
    return dict(DEFAULT_THRESHOLDS)


def confidence(relevance: float, thresholds: dict | None = None) -> str:
    """"low" — відмова, "uncertain" — відповідь із попередженням, "high" — звичайна відповідь."""
    t = thresholds or load_thresholds()
    if relevance < t["low"]:
        return "low"
    return "uncertain" if relevance < t["high"] else "high"


def _questions(path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [item["question"] if isinstance(item, dict) else item for item in data]


def calibrate(inside_files: list[Path], outside_file: Path, write: bool = True) -> dict:
    """Нижній поріг не відкидає жодного питання з корпусу (із запасом LOW_MARGIN),
    верхній відсікає HIGH_PERCENTILE% питань поза корпусом."""
    inside = [q for f in inside_files for q in _questions(f)]
    outside = _questions(outside_file)
    rel_in = np.array([retrieve(q, k=1)[1]["relevance"] for q in inside])
    rel_out = np.array([retrieve(q, k=1)[1]["relevance"] for q in outside])
    low = round(float(rel_in.min()) - LOW_MARGIN, 3)
    high = round(max(float(np.percentile(rel_out, HIGH_PERCENTILE)), low), 3)
    zones = {name: {"refused": float(np.mean(v < low)), "uncertain": float(np.mean((v >= low) & (v < high))),
                    "confident": float(np.mean(v >= high))} for name, v in (("inside", rel_in), ("outside", rel_out))}
    report = {"low": low, "high": high, "zones": zones,
              "inside_files": [str(f) for f in inside_files], "outside_file": str(outside_file),
              "relevance_inside": [round(float(v), 4) for v in rel_in],
              "relevance_outside": [round(float(v), 4) for v in rel_out]}
    print(f"в корпусі ({len(inside)}): min={rel_in.min():.3f} median={np.median(rel_in):.3f} | "
          f"поза корпусом ({len(outside)}): max={rel_out.max():.3f} median={np.median(rel_out):.3f}")
    print(f"пороги: low={low:.3f}, high={high:.3f}")
    for name, label in (("inside", "в корпусі"), ("outside", "поза корпусом")):
        z = zones[name]
        print(f"  {label:14} відмова {z['refused']:.0%} | попередження {z['uncertain']:.0%} | впевнено {z['confident']:.0%}")
    if write:
        thresholds_path().write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Збережено: {thresholds_path()}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate", action="store_true", help="підібрати пороги відмови")
    p.add_argument("--inside", type=Path, nargs="+", default=[Path("eval/sample_dev.json")])
    p.add_argument("--outside", type=Path, default=Path("eval/offtopic.json"))
    p.add_argument("--dry-run", action="store_true", help="не записувати thresholds.json")
    args = p.parse_args()
    if not args.calibrate:
        p.error("наразі підтримується лише --calibrate")
    calibrate(args.inside, args.outside, write=not args.dry_run)


if __name__ == "__main__":
    main()
