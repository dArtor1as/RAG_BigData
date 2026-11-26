"""Генерація відповіді за знайденими анотаціями: answer() — спільний контракт команди.

    from src.generator import answer
    result = answer("How can sensor networks save battery?", k=5, model="qwen3:4b-instruct",
                    use_rag=True, hyde=False, rerank=True)
    result["answer"], result["sources"]

Крім обов'язкових ключів answer і sources, результат містить:
refused, refusal_reason, confidence ("high" / "uncertain" / "low"), relevance,
claims (перевірка цитат), supported_ratio, timings, config, hyde_passage.
"""
import re
from functools import partial

from .citations import supported_ratio, verify_citations
from .llm import default_backend, default_model, generate
from .profiling import Timings
from .query import confidence, load_thresholds, retrieve
from .retriever import get_retriever

MAX_SOURCE_CHARS = 1500  # ~350 токенів на анотацію: вистачає майже для всіх анотацій arXiv
NOT_FOUND = "NOT_FOUND"
# Модель інколи відмовляє словами замість NOT_FOUND: «The provided sources do not explain…».
_DECLINE = re.compile(r"^\W*(the\s+)?(provided\s+|given\s+)?(sources|abstracts|papers)\s+do\s+not\b"
                      r"|^\W*(надані\s+)?джерела\s+не\b", re.IGNORECASE)

SYSTEM_RAG = f"""You are a research assistant that answers questions using arXiv paper abstracts.
Rules:
- Use ONLY the numbered sources below. Do not use outside knowledge.
- After every sentence with a factual claim, cite its sources as [1], [2] or [1][3].
- Cite only numbers that exist in the sources list.
- Be concise: 2-6 sentences.
- Answer in the same language as the question.
- If the sources do not contain the answer, reply with exactly {NOT_FOUND} and nothing else."""

SYSTEM_PLAIN = """You are a research assistant. Answer the question concisely (2-6 sentences)
in the same language as the question."""

REFUSAL = {
    "uk": "У корпусі немає статей, які відповідають на це питання.",
    "en": "The corpus has no papers that answer this question.",
}


def _lang(text: str) -> str:
    return "uk" if re.search(r"[а-яіїєґ]", text, re.IGNORECASE) else "en"


def build_context(sources: list[dict]) -> str:
    blocks = []
    for n, s in enumerate(sources, 1):
        # text = «назва. анотація» (так його зібрав prepare_data.py), назву показуємо окремо.
        abstract = s["text"][len(s["title"]):].lstrip(". ") if s["text"].startswith(s["title"]) else s["text"]
        if len(abstract) > MAX_SOURCE_CHARS:
            abstract = abstract[:MAX_SOURCE_CHARS] + "…"
        blocks.append(f"Source [{n}]\nTitle: {s['title']}\nAbstract: {abstract}")
    return "\n\n".join(blocks)


def build_prompt(query: str, sources: list[dict]) -> str:
    # Явна вимога писати своїми словами: інакше qwen3:0.6b копіює шапки джерел у відповідь.
    return (f"{build_context(sources)}\n\nQuestion: {query}\n\n"
            f"Write the answer in your own words, citing sources like [1]. "
            f"If the sources do not answer the question, reply {NOT_FOUND}.")


def _source_record(n: int, hit: dict) -> dict:
    return {"n": n, **hit, "url": f"https://arxiv.org/abs/{hit['arxiv_id']}"}


def answer(query: str, k: int = 5, model: str | None = None, use_rag: bool = True,
           hyde: bool = False, rerank: bool = False, backend: str | None = None,
           verify: bool = True, refuse: bool = True, thresholds: dict | None = None) -> dict:
    """Відповідь на питання з пронумерованими джерелами.

    use_rag=False — питання йде в LLM без пошуку (бейзлайн для порівняння).
    Якщо схожість питання з найкращою анотацією нижча за нижній поріг — відмова без LLM;
    між нижнім і верхнім — відповідь з confidence="uncertain", щоб інтерфейс показав попередження.
    refuse=False вимикає відмову за порогом (наприклад, щоб оцінити генерацію на всіх питаннях).
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Питання має бути непорожнім рядком.")
    query = query.strip()
    backend = backend or default_backend()
    model = model or default_model(backend)
    timings = Timings()
    result = {
        "answer": "", "sources": [], "refused": False, "refusal_reason": None,
        "confidence": None, "relevance": None,
        "claims": [], "supported_ratio": None, "hyde_passage": None,
        "timings": timings,
        "config": {"k": k, "model": model, "backend": backend, "use_rag": use_rag,
                   "hyde": hyde, "rerank": rerank},
    }

    if not use_rag:
        with timings.stage("llm"):
            result["answer"] = generate(query, system=SYSTEM_PLAIN, model=model, backend=backend)
        return result

    hits, info = retrieve(query, k=k, hyde=hyde, use_rerank=rerank, model=model, backend=backend,
                          timings=timings)
    result["sources"] = [_source_record(n, h) for n, h in enumerate(hits, 1)]
    result["hyde_passage"] = info["hyde_passage"]
    result["relevance"] = info["relevance"]
    thresholds = thresholds or load_thresholds()
    result["config"]["thresholds"] = thresholds
    result["confidence"] = confidence(info["relevance"], thresholds)

    if refuse and result["confidence"] == "low":
        result.update(answer=REFUSAL[_lang(query)], refused=True, refusal_reason="low_score")
        return result

    with timings.stage("llm"):
        text = generate(build_prompt(query, hits), system=SYSTEM_RAG, model=model, backend=backend)
    if not text.strip() or NOT_FOUND in text.upper() or _DECLINE.search(text):
        result.update(answer=REFUSAL[_lang(query)], refused=True, refusal_reason="llm")
        return result
    result["answer"] = text

    if verify:
        with timings.stage("verify"):
            encode = partial(get_retriever().encode, side="passage")
            result["claims"] = verify_citations(text, hits, encode)
            result["supported_ratio"] = supported_ratio(result["claims"])
    return result
