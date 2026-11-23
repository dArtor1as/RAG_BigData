"""Shared paths, fixed encoder revisions, and small file helpers."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_1k.parquet"
DEFAULT_WORK = ROOT / "artifacts" / "sample"
DEFAULT_QUESTIONS = ROOT / "eval" / "sample_dev.json"
INDEXES = ("flat", "hnsw", "ivfpq")
# The same snapshot and prompts must encode both sides of a search.
MODELS = {
    "minilm": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "passage": "", "query": "",
    },
    "bge": {
        "repo": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "passage": "", "query": "Represent this sentence for searching relevant passages: ",
    },
    "e5": {
        "repo": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "passage": "passage: ", "query": "query: ",
    },
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    os.replace(file.name, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive(value):
    value = int(value)
    if value < 1:
        raise ValueError("Expected a positive integer.")
    return value


def selected_model(work):
    selection = Path(work) / "selection.json"
    if selection.is_file():
        return read_json(selection)["model"]
    available = [key for key in MODELS if (Path(work) / key / "embeddings" / "manifest.json").is_file()
                 or any((Path(work) / key / "indexes").glob("*/build.json"))]
    if len(available) > 1:
        raise ValueError("Several encoders are available. Supply a model or run compare_models.")
    return available[0] if available else "minilm"


def load_encoder(key, device):
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    spec = MODELS[key]
    files = ["config.json", "config_sentence_transformers.json", "modules.json", "1_Pooling/config.json",
             "sentence_bert_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "vocab.txt", "sentencepiece.bpe.model"]
    snapshot = snapshot_download(spec["repo"], revision=spec["revision"], allow_patterns=files)
    return SentenceTransformer(snapshot, device=device, local_files_only=True)


def encode_texts(encoder, texts, key, side, batch_size):
    import numpy as np

    vectors = encoder.encode(
        texts, prompt=MODELS[key][side], batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("The encoder returned invalid or unnormalized vectors.")
    return vectors
