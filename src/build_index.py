"""Build exact Flat and approximate HNSW/IVF-PQ indexes with inner product."""
import argparse
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np
from filelock import FileLock

from .common import DEFAULT_WORK, INDEXES, MODELS, positive, read_json, selected_model, sha256, write_json
from .corpus import catalog_info


def checked_parts(folder, model, corpus):
    manifest = folder / "manifest.json"
    state = read_json(manifest)
    config = state["config"]
    if (config["source_sha256"] != corpus["source_sha256"] or config["rows"] != corpus["rows"]
            or config["model"] != model or config["encoder"] != MODELS[model]):
        raise ValueError("Embedding checkpoints do not match the corpus or encoder.")
    parts = []
    position = 0
    for name, part in sorted(state["parts"].items(), key=lambda item: item[1]["start"]):
        path = folder / name
        if part["start"] != position or part["stop"] <= position or sha256(path) != part["sha256"]:
            raise ValueError(f"Incomplete or damaged shard {name}. Run embed again.")
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.shape != (part["stop"] - position, config["dimension"]) or matrix.dtype != np.float16:
            raise ValueError(f"Unexpected shape or type in {name}.")
        parts.append((position, part["stop"], path))
        position = part["stop"]
    if position != corpus["rows"]:
        raise ValueError(f"Only {position} of {corpus['rows']} documents are encoded. Resume embed.")
    return parts, config["dimension"], sha256(manifest)


def unit_float32(vectors):
    # FP16 storage introduces rounding. Re-normalize in FP32 before FAISS uses the vectors.
    result = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(result)
    return result


def training_sample(parts, n, d, seed):
    count = min(200_000, n)
    ids = np.sort(np.random.default_rng(seed).choice(n, size=count, replace=False))
    train = np.empty((count, d), dtype="float32")
    for start, stop, path in parts:
        left, right = np.searchsorted(ids, (start, stop))
        if left != right:
            matrix = np.load(path, mmap_mode="r", allow_pickle=False)
            train[left:right] = matrix[ids[left:right] - start]
    faiss.normalize_L2(train)
    return train


def new_index(kind, d, n, parts):
    if kind == "flat":
        return faiss.IndexFlatIP(d), {}
    if kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 128
        return index, {"M": 32, "efConstruction": 200, "efSearch": 128}
    if kind != "ivfpq" or n < 2:
        raise ValueError("IVF-PQ needs at least two vectors and a supported index name.")
    train = training_sample(parts, n, d, seed=42)
    # Supply about 39 training vectors per centroid where the corpus size allows it.
    available = max(1, len(train) // 39)
    nlist = min(4096, available)
    bits = min(8, max(1, available.bit_length() - 1))
    index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 48, bits, faiss.METRIC_INNER_PRODUCT)
    index.cp.seed = index.pq.cp.seed = 42
    index.train(train)
    index.nprobe = min(nlist, 32, max(8, nlist // 8))
    return index, {"nlist": nlist, "nprobe": index.nprobe, "pq_m": 48,
                   "nbits": bits, "train_rows": len(train), "seed": 42}


def build(work, model, indexes=INDEXES, threads=4):
    work = Path(work)
    faiss.omp_set_num_threads(positive(threads))
    folder = work / model / "embeddings"
    corpus = catalog_info(work)
    destination = work / model / "indexes"
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / ".encode.lock")), FileLock(str(destination / ".build.lock")):
        parts, dim, fingerprint = checked_parts(folder, model, corpus)
        for kind in indexes:
            target = destination / kind
            if target.exists():
                info = read_json(target / "build.json")
                if info["embedding_sha256"] != fingerprint or not (target / "index.faiss").is_file():
                    raise ValueError(f"The existing {kind} index is stale or incomplete. Move its folder and rebuild.")
                print(f"{model}: reusing {kind}", flush=True)
                continue
            with TemporaryDirectory(dir=destination) as scratch:
                # Publish the binary and its metadata together after both writes succeed.
                pending = Path(scratch) / kind
                pending.mkdir()
                started = time.perf_counter()
                index, params = new_index(kind, dim, corpus["rows"], parts)
                for _, _, path in parts:
                    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
                    for offset in range(0, len(matrix), 8192):
                        index.add(unit_float32(matrix[offset:offset + 8192]))
                seconds = time.perf_counter() - started
                if index.ntotal != corpus["rows"] or not index.is_trained:
                    raise ValueError(f"Incomplete {kind} index.")
                binary = pending / "index.faiss"
                faiss.write_index(index, str(binary))
                info = {**corpus, "model": model, "encoder": MODELS[model], "index": kind,
                        "dimension": dim, "embedding_sha256": fingerprint, "params": params,
                        "build_threads": threads, "build_seconds": seconds, "bytes": binary.stat().st_size,
                        "faiss_version": faiss.__version__}
                write_json(pending / "build.json", info)
                pending.rename(target)
                del index
                print(f"{model}: built {kind} in {seconds:.3f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--indexes", nargs="+", choices=INDEXES, default=list(INDEXES))
    p.add_argument("--threads", type=positive, default=4)
    args = p.parse_args()
    build(args.work, args.model or selected_model(args.work), args.indexes, args.threads)


if __name__ == "__main__":
    main()
