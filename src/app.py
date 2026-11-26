"""Streamlit-демо RAG-асистента.

    streamlit run src/app.py

Корпус і індекс задаються змінними RAG_WORK, RAG_MODEL, RAG_INDEX (як для retriever).
"""
import html
import sys
from pathlib import Path

import streamlit as st

# Streamlit запускає файл як скрипт, тому корінь репозиторію додаємо в шлях вручну.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.citations import STATUS_LABELS
from src.generator import answer
from src.llm import DEFAULT_MODELS
from src.query import load_thresholds
from src.retriever import get_retriever

st.set_page_config(page_title="arXiv RAG", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Завантажую індекс і енкодер…")
def warm_up():
    retriever = get_retriever()
    return retriever.index.ntotal, retriever.model, type(retriever.index).__name__


def highlighted(claims: list[dict]) -> str:
    """Відповідь, у якій непідтверджені речення підсвічені, з поясненням у підказці."""
    parts = []
    for c in claims:
        text = html.escape(c["sentence"])
        if c["status"] in ("ok", "short"):
            parts.append(text)
        else:
            sim = f", схожість {c['similarity']:.2f}" if c["similarity"] is not None else ""
            tip = html.escape(STATUS_LABELS[c["status"]] + sim)
            # Напівпрозоре тло і колір тексту теми: читається і в світлій, і в темній темі.
            parts.append(f'<span title="{tip}" style="background:rgba(255,170,0,0.18);'
                         f'border-bottom:2px dotted #e8a000;padding:0 2px">{text}</span>')
    return " ".join(parts)


papers, encoder, index_kind = warm_up()

with st.sidebar:
    st.header("Налаштування")
    backend = st.radio("Бекенд LLM", ["ollama", "gemini"], horizontal=True)
    models = ["qwen3:4b-instruct", "qwen3:0.6b"] if backend == "ollama" else [DEFAULT_MODELS["gemini"], "gemini-2.5-flash-lite"]
    model = st.selectbox("Модель", models)
    k = st.slider("Кількість джерел k", 1, 10, 5)
    use_rag = st.toggle("RAG (пошук у корпусі)", value=True)
    hyde = st.toggle("HyDE", value=False, disabled=not use_rag)
    rerank = st.toggle("Reranker (cross-encoder)", value=False, disabled=not use_rag)
    refuse = st.toggle("Відмова за порогом", value=True, disabled=not use_rag)
    defaults = load_thresholds()
    low = st.number_input("Нижче — відмова (косинус)", value=float(defaults["low"]), step=0.01, format="%.3f",
                          disabled=not use_rag)
    high = st.number_input("Нижче — попередження (косинус)", value=float(defaults["high"]), step=0.01,
                           format="%.3f", disabled=not use_rag)
    st.caption(f"Корпус: {papers:,} статей · енкодер {encoder} · індекс {index_kind}".replace(",", " "))

st.title("Пошук по анотаціях arXiv")
query = st.text_input("Питання", placeholder="How can neighboring sensors conserve battery power?")

if st.button("Запитати", type="primary", disabled=not query.strip()):
    with st.spinner("Шукаю і генерую відповідь…"):
        try:
            result = answer(query, k=k, model=model, use_rag=use_rag, hyde=hyde, rerank=rerank,
                            backend=backend, refuse=refuse, thresholds={"low": low, "high": high})
        except Exception as error:  # показуємо помилку в інтерфейсі замість падіння сторінки
            st.error(f"{type(error).__name__}: {error}")
            st.stop()

    if result["refused"]:
        st.warning(result["answer"])
        if result["refusal_reason"] == "low_score":
            st.caption(f"Релевантність найкращої анотації {result['relevance']:.3f} нижча за поріг {low:.3f}.")
        else:
            st.caption("Модель не знайшла відповіді в знайдених анотаціях.")
    else:
        if result["confidence"] == "uncertain":
            st.warning(f"Знайдені статті лише частково схожі на питання (релевантність {result['relevance']:.3f}). "
                       "Відповідь може спиратися на дотичні роботи — перевірте джерела.")
        st.subheader("Відповідь")
        with st.container(border=True):
            if result["claims"]:
                st.markdown(highlighted(result["claims"]), unsafe_allow_html=True)
            else:
                st.markdown(result["answer"])
        unsupported = [c for c in result["claims"] if c["status"] not in ("ok", "short")]
        if result["supported_ratio"] is not None:
            st.caption(f"Підтверджено джерелами: {result['supported_ratio']:.0%} речень. "
                       "Підкреслені пунктиром речення не підтверджені.")
        if unsupported:
            with st.expander(f"Непідтверджені речення ({len(unsupported)})"):
                for c in unsupported:
                    sim = f" (схожість {c['similarity']:.2f})" if c["similarity"] is not None else ""
                    st.markdown(f"- **{STATUS_LABELS[c['status']]}**{sim}: {c['sentence']}")

    timings = result["timings"]
    st.caption("Час, с: " + " · ".join(f"{name} {sec:.2f}" for name, sec in timings.items())
               + f" · разом {sum(timings.values()):.2f}")

    if result["hyde_passage"]:
        with st.expander("Гіпотетична анотація (HyDE): вигадана LLM лише для пошуку, це не відповідь"):
            st.write(result["hyde_passage"])

    if result["sources"]:
        st.subheader("Джерела")
        for s in result["sources"]:
            score = f"score {s['score']:.3f}" + (f" · rerank {s['rerank_score']:.2f}" if "rerank_score" in s else "")
            with st.expander(f"[{s['n']}] {s['title']}  ·  {score}"):
                st.markdown(f"[arxiv.org/abs/{s['arxiv_id']}]({s['url']})")
                st.write(s["text"])
