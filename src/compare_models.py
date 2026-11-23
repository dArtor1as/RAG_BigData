"""Choose between encoders using labeled questions and exact Flat retrieval."""
import argparse
import csv
import os
from tempfile import NamedTemporaryFile
from contextlib import closing
from pathlib import Path

import faiss
import numpy as np

from .build_index import build
from .common import (DEFAULT_QUESTIONS, DEFAULT_WORK, MODELS, SAMPLE, encode_texts, load_encoder,
                     positive, read_json, sha256, write_json)
from .corpus import catalog_info, connect_catalog, prepare_catalog
from .embed import embed
from .retriever import open_index


def read_questions(path):
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("The questions file must be a nonempty JSON array.")
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Question {i + 1} needs a nonempty 'question' string.")
    return rows


def gold_row_ids(work, questions):
    ids = []
    with closing(connect_catalog(work)) as db:
        for question in questions:
            arxiv_id = question.get("arxiv_id")
            if not isinstance(arxiv_id, str) or not arxiv_id.strip():
                raise ValueError("Each development question needs one arxiv_id encoded as a JSON string.")
            match = db.execute("SELECT row_id FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
            if match is None:
                raise ValueError(f"Gold arxiv_id {arxiv_id!r} is missing from this corpus. Use matching data and questions.")
            ids.append(match[0])
    return ids


def write_csv(path, rows):
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(file.name, path)


def compare(data, work, questions, models=("minilm", "bge"), device="cpu", batch_size=32,
            chunk_size=None, threads=4):
    work = Path(work)
    if len(set(models)) < 2 or len(set(models)) != len(models):
        raise ValueError("Choose at least two distinct encoders.")
    corpus = prepare_catalog(data, work)
    records = read_questions(questions)
    gold = gold_row_ids(work, records)
    texts = [row["question"].strip() for row in records]
    reports = work / "reports"
    reports.mkdir(exist_ok=True)
    summary, details = [], []
    for model in models:
        embed(data, work, model, device, batch_size, chunk_size)
        build(work, model, ["flat"], threads)
        index = open_index(work, model, "flat")
        encoder = load_encoder(model, "cpu")
        queries = encode_texts(encoder, texts, model, "query", batch_size)
        faiss.omp_set_num_threads(positive(threads))
        ranks = []
        for record, target, vector in zip(records, gold, queries):
            # One full exact ranking at a time keeps MRR untruncated without a Q x N matrix.
            _, ids = index.search(vector[None, :], index.ntotal)
            rank = int(np.flatnonzero(ids[0] == target)[0]) + 1
            ranks.append(rank)
            details.append({"model": model, "question": record["question"], "arxiv_id": record["arxiv_id"],
                            "gold_rank": rank, "top_row_id": int(ids[0, 0])})
        ranks = np.asarray(ranks)
        row = {"model": model, "papers": corpus["rows"], "questions": len(records),
               **{f"recall_at_{k}": float(np.mean(ranks <= k)) for k in (1, 5, 10)},
               "mrr": float(np.mean(1.0 / ranks))}
        summary.append(row)
        print(f"{model}: MRR {row['mrr']:.4f}, gold Recall@10 {row['recall_at_10']:.4f}", flush=True)
        del encoder, index
    winner = max(summary, key=lambda r: (r["mrr"], r["recall_at_10"], r["recall_at_1"]))
    write_csv(reports / "model_comparison.csv", summary)
    write_csv(reports / "model_question_ranks.csv", details)
    selection = {**catalog_info(work), "model": winner["model"], "encoder": MODELS[winner["model"]],
                 "questions_file": Path(questions).name, "questions_sha256": sha256(questions),
                 "questions": len(records), "models_compared": list(models),
                 "rule": "Highest full MRR, then Recall@10, then Recall@1, then supplied model order."}
    write_json(work / "selection.json", selection)
    print(f"Selected {winner['model']} on {len(records)} development questions.", flush=True)
    return winner["model"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=SAMPLE)
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--models", nargs="+", choices=MODELS, default=["minilm", "bge"])
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=positive, default=32)
    p.add_argument("--chunk-size", type=positive)
    p.add_argument("--threads", type=positive, default=4)
    compare(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
