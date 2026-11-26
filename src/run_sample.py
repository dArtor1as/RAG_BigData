"""Run the Student 2 sample workflow from raw Parquet to a live search."""
import argparse
from pathlib import Path

from .benchmark import benchmark
from .build_index import build
from .common import DEFAULT_QUESTIONS, DEFAULT_WORK, SAMPLE
from .compare_models import compare
from .retriever import Retriever


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=DEFAULT_WORK)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = p.parse_args()
    winner = compare(SAMPLE, args.work, args.questions, device=args.device, batch_size=16)
    build(args.work, winner)
    benchmark(args.work, winner, args.questions)
    hits = Retriever(args.work, winner).search("Can reader activity help a library recommend related preprints?", 3)
    print("\nExample search")
    for i, paper in enumerate(hits, 1):
        print(f"{i}. {paper['arxiv_id']}  {paper['score']:.4f}  {paper['title']}")
    print(f"Results: {args.work / 'reports'}")


if __name__ == "__main__":
    main()
