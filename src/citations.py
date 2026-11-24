"""Перевірка цитат — простий детектор галюцинацій.

Для кожного речення відповіді беремо джерела, на які воно посилається ([1], [2, 3], [1-3]),
і рахуємо косинус між ембедінгом речення та ембедінгом анотації. Речення без цитати,
з неіснуючим номером джерела або з низькою схожістю позначаються як «не підтверджено».
"""
import re

import numpy as np

# Мінімальна косинусна схожість речення з анотацією (BGE-small, passage–passage),
# за якої твердження вважаємо підтвердженим. Уточнюється на тестових відповідях.
SIM_THRESHOLD = 0.60
MIN_WORDS = 4  # коротші фрагменти (заголовки, «Отже:») не перевіряємо

STATUS_LABELS = {
    "ok": "підтверджено",
    "no_citation": "немає посилання",
    "bad_index": "посилання на неіснуюче джерело",
    "low_similarity": "джерело не підтверджує",
    "short": "не перевірялось",
}

_CITATION = re.compile(r"\[(\d+(?:\s*[,;\-–]\s*\d+)*)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[^\s\[])")
_LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def parse_citations(text: str) -> list[int]:
    """'[1][3]', '[1, 3]', '[1-3]' → відсортовані унікальні номери."""
    numbers = set()
    for group in _CITATION.findall(text):
        for part in re.split(r"\s*[,;]\s*", group):
            bounds = re.split(r"\s*[\-–]\s*", part)
            if len(bounds) == 2:
                lo, hi = int(bounds[0]), int(bounds[1])
                numbers.update(range(min(lo, hi), max(lo, hi) + 1))
            else:
                numbers.add(int(bounds[0]))
    return sorted(numbers)


def strip_citations(text: str) -> str:
    return re.sub(r"\s+", " ", _CITATION.sub("", text)).strip()


_ABBREV = re.compile(r"\b(?:e\.g|i\.e|et al|etc|vs|Fig|Eq|cf|т\. зв|т\. ч|напр|див)\.", re.IGNORECASE)
_CITE_AFTER_END = re.compile(r"([.!?…])((?:\s*\[\d[^\]]*\])+)")


def split_sentences(answer: str) -> list[str]:
    """Ділить відповідь на речення.

    Цитату, що стоїть після крапки ('... done. [2] Next'), спершу переносимо перед крапку
    ('... done [2]. Next'), щоб вона лишилась із реченням, яке підтверджує.
    """
    sentences = []
    for line in answer.splitlines():
        line = _LIST_MARK.sub("", line).strip()
        line = _CITE_AFTER_END.sub(lambda m: " " + m.group(2).strip() + m.group(1), line)
        # Крапки в скороченнях («e.g.», «et al.», «т. зв.») не кінець речення.
        line = _ABBREV.sub(lambda m: m.group(0).replace(".", "\x00"), line)
        sentences.extend(part.strip().replace("\x00", ".") for part in _SENTENCE_END.split(line) if part.strip())
    return sentences


def verify_citations(answer: str, sources: list[dict], encode, threshold: float = SIM_THRESHOLD) -> list[dict]:
    """Повертає по одному запису на речення: sentence, citations, status, similarity.

    encode(texts) -> нормалізовані вектори (n, d); у системі це Retriever.encode(..., "passage").
    sources — список у тому ж порядку, що й номери [1], [2], … у промпті.
    """
    sentences = split_sentences(answer)
    claims = []
    to_check = []  # (індекс речення, номери валідних джерел)
    for sentence in sentences:
        cited = parse_citations(sentence)
        plain = strip_citations(sentence)
        claim = {"sentence": sentence, "citations": cited, "status": None, "similarity": None}
        if len(plain.split()) < MIN_WORDS:
            claim["status"] = "short" if not cited or all(1 <= n <= len(sources) for n in cited) else "bad_index"
        elif not cited:
            claim["status"] = "no_citation"
        elif any(n < 1 or n > len(sources) for n in cited):
            claim["status"] = "bad_index"
        else:
            to_check.append((len(claims), cited))
        claims.append(claim)

    if to_check:
        texts = [strip_citations(claims[i]["sentence"]) for i, _ in to_check]
        used = sorted({n for _, cited in to_check for n in cited})
        vectors = np.asarray(encode(texts + [sources[n - 1]["text"] for n in used]), dtype=np.float32)
        sent_vecs, src_vecs = vectors[:len(texts)], dict(zip(used, vectors[len(texts):]))
        for (i, cited), vec in zip(to_check, sent_vecs):
            similarity = max(float(vec @ src_vecs[n]) for n in cited)
            claims[i]["similarity"] = round(similarity, 4)
            claims[i]["status"] = "ok" if similarity >= threshold else "low_similarity"
    return claims


def supported_ratio(claims: list[dict]) -> float | None:
    """Частка підтверджених речень серед перевірених (без коротких)."""
    checked = [c for c in claims if c["status"] != "short"]
    return round(sum(c["status"] == "ok" for c in checked) / len(checked), 4) if checked else None
