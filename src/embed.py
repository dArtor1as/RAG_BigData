"""Encode the text column in row_id order and checkpoint normalized .npy shards."""
import argparse
from contextlib import closing
from pathlib import Path

import numpy as np
from filelock import FileLock

from .common import DEFAULT_WORK, MODELS, SAMPLE, encode_texts, load_encoder, positive, read_json, sha256, write_json
from .corpus import connect_catalog, prepare_catalog


def embed(data, work, model, device="cpu", batch_size=32, chunk_size=None):
    batch_size = positive(batch_size)
    corpus = prepare_catalog(data, work)
    folder = Path(work) / model / "embeddings"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    encoder = None
    with FileLock(str(folder / ".encode.lock")), closing(connect_catalog(work)) as db:
        state = read_json(path) if path.is_file() else None
        if chunk_size is None:
            chunk_size = state["config"]["chunk_size"] if state else (250 if corpus["rows"] <= 1000 else 200_000)
        chunk_size = positive(chunk_size)
        config = {**corpus, "model": model, "encoder": MODELS[model], "dimension": 384,
                  "dtype": "float16", "normalized": True, "chunk_size": chunk_size}
        state = state or {"config": config, "parts": {}}
        if state["config"] != config:
            raise ValueError("Checkpoint settings differ. Reuse the original settings or choose a new --work.")
        for start in range(0, corpus["rows"], chunk_size):
            stop = min(start + chunk_size, corpus["rows"])
            name = f"emb_{start // chunk_size:05d}.npy"
            final = folder / name
            old = state["parts"].get(name)
            if old and final.is_file() and sha256(final) == old["sha256"]:
                print(f"{model}: reusing rows {start} to {stop - 1}", flush=True)
                continue
            if encoder is None:
                encoder = load_encoder(model, device)
                if encoder.get_sentence_embedding_dimension() != config["dimension"]:
                    raise ValueError("Unexpected embedding dimension for the configured model.")
            temp = final.with_suffix(".partial")
            vectors = np.lib.format.open_memmap(temp, mode="w+", dtype="float16", shape=(stop - start, 384))
            query = db.execute("SELECT text FROM papers WHERE row_id >= ? AND row_id < ? ORDER BY row_id",
                               (start, stop))
            offset = 0
            while rows := query.fetchmany(min(4096, batch_size * 4)):
                texts = [row[0] for row in rows]
                batch = encode_texts(encoder, texts, model, "passage", batch_size)
                vectors[offset:offset + len(batch)] = batch
                offset += len(batch)
            if offset != stop - start:
                raise ValueError("The corpus changed during encoding.")
            vectors.flush()
            del vectors
            temp.replace(final)
            state["parts"][name] = {"start": start, "stop": stop, "sha256": sha256(final)}
            write_json(path, state)
            print(f"{model}: saved rows {start} to {stop - 1} in {name}", flush=True)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=SAMPLE)
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--model", choices=MODELS, default="minilm")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--batch-size", type=positive, default=32)
    p.add_argument("--chunk-size", type=positive, help="Rows per checkpoint (default: 250 for the sample, 200000 for larger corpora).")
    args = p.parse_args()
    embed(**vars(args))


if __name__ == "__main__":
    main()
