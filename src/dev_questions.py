"""Синтетичні dev-питання для калібрування порогів відмови.

Беремо випадкові статті з корпусу, і LLM пише до кожної питання так, як його поставив би
дослідник у пошуку: своїми словами, без назви статті. Відповідь — arxiv_id цієї статті.

    python -m src.dev_questions --n 100 --out eval/dev_full.json

Це набір для налаштування (dev). Фінальне оцінювання робиться на окремих питаннях студента 4.
"""
import argparse
import json
from contextlib import closing
from pathlib import Path

import numpy as np

from .corpus import connect_catalog
from .llm import generate
from .retriever import get_retriever

SYSTEM = ("You write search queries. Reply with the question only, 8-20 words, in English. "
          "Use your own words: do not copy distinctive phrases or the title, "
          "do not say 'this paper' or 'the authors'.")
PROMPT = ("Write one question that a researcher would type into a scientific search engine "
          "and that the following paper answers.\n\nTitle: {title}\nAbstract: {abstract}")


def make_questions(n: int, seed: int = 42, model: str | None = None) -> list[dict]:
    work = get_retriever().work
    with closing(connect_catalog(work)) as db:
        total = db.execute("SELECT count(*) FROM papers").fetchone()[0]
        ids = np.random.default_rng(seed).choice(total, size=n, replace=False).tolist()
        rows = db.execute(f"SELECT row_id, arxiv_id, title, text FROM papers WHERE row_id IN ({','.join('?' * n)})",
                          ids).fetchall()
    questions = []
    for i, (row_id, arxiv_id, title, text) in enumerate(sorted(rows), 1):
        abstract = text[len(title):].lstrip(". ") if text.startswith(title) else text
        q = generate(PROMPT.format(title=title, abstract=abstract[:1500]), system=SYSTEM,
                     model=model, temperature=0.7, max_tokens=60).strip().strip('"')
        questions.append({"question": q, "arxiv_id": arxiv_id})
        print(f"{i:3}/{n} {arxiv_id:>12}  {q}", flush=True)
    return questions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=None)
    p.add_argument("--out", type=Path, default=Path("eval/dev_full.json"))
    args = p.parse_args()
    questions = make_questions(args.n, args.seed, args.model)
    args.out.write_text(json.dumps(questions, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Збережено {len(questions)} питань: {args.out}")


if __name__ == "__main__":
    main()
