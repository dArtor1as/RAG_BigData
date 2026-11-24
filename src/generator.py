"""Генерація відповіді за знайденими анотаціями: answer() — спільний контракт команди.

    from src.generator import answer
    result = answer("How can sensor networks save battery?", k=5, model="qwen3:4b",
                    use_rag=True, hyde=False, rerank=True)
    result["answer"], result["sources"]

Крім обов'язкових ключів answer і sources, результат містить:
refused, refusal_reason, claims (перевірка цитат), supported_ratio, timings, config, hyde_passage.
"""
import re

from .citations import supported_ratio, verify_citations
from .llm import default_backend, default_model, generate
from .profiling import Timings
from .query import is_out_of_corpus, load_thresholds, retrieve
from .retriever import get_retriever

MAX_SOURCE_CHARS = 1500  # ~350 токенів на анотацію: вистачає майже для всіх анотацій arXiv
NOT_FOUND = "NOT_FOUND"

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
        text = s["text"] if len(s["text"]) <= MAX_SOURCE_CHARS else s["text"][:MAX_SOURCE_CHARS] + "…"
        blocks.append(f"[{n}] (arXiv:{s['arxiv_id']}) {text}")
    return "\n\n".join(blocks)


def build_prompt(query: str, sources: list[dict]) -> str:
    return f"Sources:\n\n{build_context(sources)}\n\nQuestion: {query}\nAnswer:"


def _source_record(n: int, hit: dict) -> dict:
    return {"n": n, **hit, "url": f"https://arxiv.org/abs/{hit['arxiv_id']}"}


def answer(query: str, k: int = 5, model: str | None = None, use_rag: bool = True,
           hyde: bool = False, rerank: bool = False, backend: str | None = None,
           verify: bool = True, refuse: bool = True, thresholds: dict | None = None) -> dict:
    """Відповідь на питання з пронумерованими джерелами.

    use_rag=False — питання йде в LLM без пошуку (бейзлайн для порівняння).
    refuse=False вимикає поріг відмови (наприклад, щоб оцінити генерацію на всіх питаннях).
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Питання має бути непорожнім рядком.")
    query = query.strip()
    backend = backend or default_backend()
    model = model or default_model(backend)
    timings = Timings()
    result = {
        "answer": "", "sources": [], "refused": False, "refusal_reason": None,
        "claims": [], "supported_ratio": None, "hyde_passage": None, "relevance": None,
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
    result["config"]["threshold"] = (thresholds or load_thresholds())[info["mode"]]

    if refuse and is_out_of_corpus(info, thresholds):
        result.update(answer=REFUSAL[_lang(query)], refused=True, refusal_reason="low_score")
        return result

    with timings.stage("llm"):
        text = generate(build_prompt(query, hits), system=SYSTEM_RAG, model=model, backend=backend)
    if text.strip().strip(".").upper() == NOT_FOUND or not text.strip():
        result.update(answer=REFUSAL[_lang(query)], refused=True, refusal_reason="llm")
        return result
    result["answer"] = text

    if verify:
        with timings.stage("verify"):
            encode = lambda texts: get_retriever().encode(texts, "passage")  # noqa: E731
            result["claims"] = verify_citations(text, hits, encode)
            result["supported_ratio"] = supported_ratio(result["claims"])
    return result
