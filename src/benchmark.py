"""Measure index construction, RAM, serialized size, latency and agreement with Flat."""
import argparse
import gc
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import faiss
import numpy as np
import psutil

from .common import (DEFAULT_QUESTIONS, DEFAULT_WORK, INDEXES, MODELS, encode_texts, load_encoder,
                     positive, read_json, selected_model, sha256, write_json)
from .compare_models import read_questions, write_csv
from .corpus import catalog_info
from .retriever import index_file

MIB = 1024 ** 2


def measure_index(work, model, kind, queries, threads, repeats):
    """Each call runs in a fresh CPU process, without the query encoder in its heap."""
    faiss.omp_set_num_threads(threads)
    binary, info = index_file(work, model, kind)
    # Repeated small loads reduce the influence of existing allocator slack and page rounding.
    copies = min(32, max(1, (8 * MIB + info["bytes"] - 1) // info["bytes"]))
    process = psutil.Process()
    gc.collect()
    baseline = process.memory_info().rss
    index = faiss.read_index(str(binary))
    extras = [faiss.read_index(str(binary)) for _ in range(copies - 1)]
    loaded = process.memory_info().rss
    del extras
    if index.ntotal != info["rows"] or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("The benchmark loaded an incompatible index.")
    vectors = [vector.reshape(1, -1) for vector in queries]
    k = min(10, index.ntotal)
    # These calls warm the index and give the neighbor IDs for the accuracy calculation.
    neighbors = [index.search(vector, k)[1][0].tolist() for vector in vectors]
    samples = []
    rng = np.random.default_rng(42)
    for _ in range(repeats):
        for i in rng.permutation(len(vectors)):
            vector = vectors[i]
            start = time.perf_counter_ns()
            index.search(vector, k)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return neighbors, {
        "model": model, "index": kind, "papers": index.ntotal, "queries": len(vectors),
        "build_seconds": info["build_seconds"], "ram_mib": (loaded - baseline) / (MIB * copies),
        "file_mib": info["bytes"] / MIB, "mean_search_ms": float(np.mean(samples)),
        "p95_search_ms": float(np.percentile(samples, 95)), "rss_copies": copies,
        "build_threads": info["build_threads"], "search_threads": threads, "repeats": repeats,
    }


def flat_recall(exact, candidate, k):
    scores = []
    for truth, predicted in zip(exact, candidate, strict=True):
        relevant = {row for row in truth if row >= 0}
        if len(relevant) != k:
            raise ValueError("Flat returned incomplete or duplicate neighbor IDs.")
        scores.append(len(relevant.intersection(row for row in predicted if row >= 0)) / k)
    return float(np.mean(scores))



def write_stage4(path, rows, config):
    first = rows[0]
    n, q = first["papers"], first["queries"]
    k = min(10, n)
    hnsw, ivf = config["hnsw"]["params"], config["ivfpq"]["params"]
    lines = ["# Stage 4 · Methods and models", "",
             f"**{n:,} papers · {q} questions · {MODELS[first['model']]['repo']}**", "",
             "Each paper contributes one normalized 384-dimensional embedding of its title and abstract. "
             "FAISS ranks documents by inner product. Flat supplies the exact reference ranking.", "",
             f"HNSW uses M={hnsw['M']}, efConstruction={hnsw['efConstruction']} and efSearch={hnsw['efSearch']}. "
             f"IVF-PQ uses {ivf['nlist']} lists, probes {ivf['nprobe']} and encodes each vector in "
             f"{ivf['pq_m']} groups of {ivf['nbits']} bits. It trained on {ivf['train_rows']:,} vectors.", "",
             "| Index | Build (s) | RAM (MiB) | File (MiB) | Mean search (ms) | Recall@" + str(k) + " vs Flat |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    names = {"flat": "Flat", "hnsw": "HNSW", "ivfpq": "IVF-PQ"}
    for r in rows:
        lines.append(f"| {names[r['index']]} | {r['build_seconds']:.4f} | {r['ram_mib']:.3f} | "
                     f"{r['file_mib']:.3f} | {r['mean_search_ms']:.4f} | {r['recall_at_10_vs_flat']:.1%} |")
    lines += ["", f"Each latency measurement times a single pre-encoded question on CPU. The benchmark uses "
              f"{first['search_threads']} FAISS search thread(s), {first['repeats']} timed passes and one warm-up pass. "
              "RAM estimates the resident memory consumed by loading an index into a fresh process. "
              f"Recall is the mean fraction of Flat's top {k} row IDs that each index recovers.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def benchmark(work, model, questions, threads=1, repeats=10):
    work = Path(work)
    threads, repeats = positive(threads), positive(repeats)
    records = read_questions(questions)
    config = {kind: read_json(work / model / "indexes" / kind / "build.json") for kind in INDEXES}
    if len({info["embedding_sha256"] for info in config.values()}) != 1:
        raise ValueError("The three indexes must use exactly the same embeddings.")
    encoder = load_encoder(model, "cpu")
    queries = encode_texts(encoder, [row["question"].strip() for row in records], model, "query", 32)
    del encoder
    rows = []
    for kind in INDEXES:  # Exact Flat must be first.
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
            neighbors, row = pool.submit(measure_index, work, model, kind, queries, threads, repeats).result()
        if kind == "flat":
            exact = neighbors
        row["recall_at_10_vs_flat"] = flat_recall(exact, neighbors, min(10, row["papers"]))
        rows.append(row)
        print(f"{kind:5}  {row['mean_search_ms']:.3f} ms/query  "
              f"Recall@10 vs Flat {row['recall_at_10_vs_flat']:.3f}", flush=True)
    reports = work / "reports"
    reports.mkdir(exist_ok=True)
    write_csv(reports / "index_benchmark.csv", rows)
    write_stage4(reports / "stage4_index_table.md", rows, config)
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next((s.partition(":")[2].strip() for s in cpuinfo.read_text().splitlines()
                    if s.startswith("model name")), cpu)
    write_json(reports / "index_benchmark_context.json", {
        **catalog_info(work), "model": model, "encoder": MODELS[model], "query_device": "cpu",
        "question_file": Path(questions).name, "questions_sha256": sha256(questions), "questions": len(records),
        "k": min(10, rows[0]["papers"]), "cpu": cpu, "platform": platform.platform(),
        "python": sys.version.split()[0], "versions": {name: version(name) for name in
            ("faiss-cpu", "numpy", "pyarrow", "torch", "sentence-transformers", "transformers", "psutil")},
        "latency_method": "One untimed pass, then shuffled single-query FAISS search calls on pre-encoded questions.",
        "ram_method": "Fresh worker RSS change per loaded index copy, in MiB. The encoder is outside the worker.",
        "index_builds": config,
    })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--threads", type=positive, default=1)
    p.add_argument("--repeats", type=positive, default=10)
    args = p.parse_args()
    benchmark(args.work, args.model or selected_model(args.work), args.questions, args.threads, args.repeats)


if __name__ == "__main__":
    main()
