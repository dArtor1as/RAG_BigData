"""Профілювання етапів запиту: скільки секунд коштує кожне покращення.

Запуск:
    python -m src.profiling --questions eval/sample_dev.json --limit 10 --models qwen3:0.6b qwen3:4b-instruct

Результат: results/profiling.csv (кожен запуск) і results/profiling.md (середні значення).
"""
import argparse
import csv
import json
import statistics
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

STAGES = ("hyde", "encode", "search", "rerank", "llm", "verify")
CONFIGS = {
    "base": {"hyde": False, "rerank": False},
    "+rerank": {"hyde": False, "rerank": True},
    "+hyde": {"hyde": True, "rerank": False},
    "+hyde+rerank": {"hyde": True, "rerank": True},
}


class Timings(dict):
    """Словник «етап → секунди», який заповнює контекстний менеджер stage()."""

    @contextmanager
    def stage(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            self[name] = self.get(name, 0.0) + perf_counter() - start

    def total(self) -> float:
        return sum(self.values())


def run(questions, models, configs, k, backend, out_dir):
    from .generator import answer
    from .llm import default_backend

    backend = backend or default_backend()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        # Прогрів: перший виклик завантажує енкодер, reranker і модель у пам'ять.
        answer(questions[0], k=k, model=model, backend=backend, hyde=True, rerank=True)
        for name in configs:
            for q in questions:
                result = answer(q, k=k, model=model, backend=backend, **CONFIGS[name])
                t = result["timings"]
                rows.append({"model": model, "config": name, "question": q,
                             **{s: round(t.get(s, 0.0), 4) for s in STAGES},
                             "total": round(sum(t.values()), 4), "refused": result["refused"]})
                print(f"{model:12} {name:13} {rows[-1]['total']:6.2f} с  {q[:60]}", flush=True)

    with (out_dir / "profiling.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# Профілювання етапів запиту ({len(questions)} питань, k={k}, бекенд {backend})", "",
             "Середній час на одне питання, секунди. Прогрівочний запуск не враховано.", "",
             "| Модель | Конфігурація | " + " | ".join(STAGES) + " | Разом |",
             "| --- | --- | " + " | ".join("---:" for _ in STAGES) + " | ---: |"]
    for model in models:
        for name in configs:
            group = [r for r in rows if r["model"] == model and r["config"] == name]
            means = [statistics.mean(r[s] for r in group) for s in (*STAGES, "total")]
            lines.append(f"| {model} | {name} | " + " | ".join(f"{m:.3f}" for m in means) + " |")
    (out_dir / "profiling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", type=Path, default=Path("eval/sample_dev.json"))
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--models", nargs="+", default=["qwen3:0.6b", "qwen3:4b-instruct"])
    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    p.add_argument("--backend", default=None)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()
    data = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [item["question"] if isinstance(item, dict) else item for item in data][:args.limit]
    run(questions, args.models, args.configs, args.k, args.backend or None, args.out)


if __name__ == "__main__":
    main()
