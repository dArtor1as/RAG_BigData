"""Єдина точка виклику LLM для генерації, HyDE і LLM-судді.

Бекенди:
- "ollama" — локальні моделі (qwen3:4b-instruct, qwen3:0.6b), працює без інтернету.
  Тег qwen3:4b в Ollama — це Qwen3-4B-Thinking-2507, яка міркує завжди і ігнорує think=False,
  тому для RAG беремо instruct-версію тієї ж моделі;
- "gemini" — Google Gemini API, ключ береться зі змінної середовища GEMINI_API_KEY.

Бекенд і модель за замовчуванням задаються змінними RAG_BACKEND і RAG_LLM.
"""
import os
import re
from functools import lru_cache

BACKENDS = ("ollama", "gemini")
DEFAULT_MODELS = {"ollama": "qwen3:4b-instruct", "gemini": "gemini-2.5-flash"}
# 5 анотацій + інструкція ≈ 1–2 тис. токенів, тож 4096 вистачає з запасом.
OLLAMA_NUM_CTX = 4096

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def default_backend() -> str:
    return os.environ.get("RAG_BACKEND", "ollama")


def default_model(backend: str | None = None) -> str:
    backend = backend or default_backend()
    return os.environ.get("RAG_LLM") or DEFAULT_MODELS[backend]


@lru_cache(maxsize=1)
def _ollama_client():
    import ollama
    return ollama.Client(host=os.environ.get("OLLAMA_HOST"))


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Для бекенда gemini задайте змінну середовища GEMINI_API_KEY.")
    return genai.Client(api_key=key)


def generate(prompt: str, system: str | None = None, model: str | None = None,
             backend: str | None = None, temperature: float = 0.2, max_tokens: int = 768) -> str:
    """Повертає текст відповіді моделі без блоку міркувань."""
    backend = backend or default_backend()
    if backend not in BACKENDS:
        raise ValueError(f"Невідомий бекенд {backend!r}; доступні: {BACKENDS}")
    model = model or default_model(backend)

    if backend == "ollama":
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        response = _ollama_client().chat(
            model=model, messages=messages, think=False,  # режим міркувань qwen3 вимкнено
            options={"temperature": temperature, "num_predict": max_tokens, "num_ctx": OLLAMA_NUM_CTX},
        )
        text = response.message.content or ""
    else:
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        response = _gemini_client().models.generate_content(model=model, contents=prompt, config=config)
        text = response.text or ""

    # qwen3 інколи лишає порожній <think></think> навіть з think=False.
    return _THINK.sub("", text).strip()
