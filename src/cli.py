"""Консольний інтерфейс RAG-асистента.

    python -m src.cli "How can sensor networks save battery power?" -k 5 --rerank
    python -m src.cli "Як розпізнавати арабське мовлення?" --hyde --model qwen3:0.6b
    python -m src.cli "What is BERT?" --no-rag
    python -m src.cli "..." --json        # повний результат answer() у JSON
"""
import argparse
import json

from .citations import STATUS_LABELS
from .generator import answer
from .llm import BACKENDS


def print_result(result: dict) -> None:
    cfg = result["config"]
    print(f"\n== Відповідь ({cfg['model']}, {cfg['backend']}, RAG={'так' if cfg['use_rag'] else 'ні'}, "
          f"HyDE={'так' if cfg['hyde'] else 'ні'}, reranker={'так' if cfg['rerank'] else 'ні'})\n")
    if result["confidence"] == "uncertain" and not result["refused"]:
        print("[!] Джерела можуть не відповідати питанню — перевірте їх.\n")
    print(result["answer"])
    if result["relevance"] is not None:
        t = cfg["thresholds"]
        print(f"\n[релевантність {result['relevance']:.3f}; пороги: відмова < {t['low']:.3f}, "
              f"попередження < {t['high']:.3f}]")
    if result["refused"]:
        reason = {"low_score": "схожість нижча за нижній поріг", "llm": "LLM не знайшла відповіді в джерелах"}
        print(f"[відмова: {reason[result['refusal_reason']]}]")

    unsupported = [c for c in result["claims"] if c["status"] not in ("ok", "short")]
    if result["claims"]:
        print(f"\n== Перевірка цитат: підтверджено {result['supported_ratio']:.0%} речень")
        for c in unsupported:
            sim = f", схожість {c['similarity']:.2f}" if c["similarity"] is not None else ""
            print(f"  ! {STATUS_LABELS[c['status']]}{sim}: {c['sentence']}")

    if result["hyde_passage"]:
        print(f"\n== Гіпотетична анотація (HyDE)\n{result['hyde_passage']}")

    if result["sources"]:
        print("\n== Джерела")
        for s in result["sources"]:
            extra = f", rerank {s['rerank_score']:.2f}" if "rerank_score" in s else ""
            print(f"[{s['n']}] {s['title']}\n    {s['url']}  (score {s['score']:.3f}{extra})")

    timings = result["timings"]
    print("\n== Час, с: " + ", ".join(f"{k} {v:.2f}" for k, v in timings.items())
          + f" | разом {sum(timings.values()):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5, help="кількість джерел для LLM")
    p.add_argument("--model", help="модель LLM (за замовчуванням qwen3:4b-instruct для ollama)")
    p.add_argument("--backend", choices=BACKENDS)
    p.add_argument("--no-rag", action="store_true", help="питати LLM без пошуку")
    p.add_argument("--hyde", action="store_true", help="пошук за гіпотетичною анотацією")
    p.add_argument("--rerank", action="store_true", help="переоцінити топ-50 cross-encoder'ом")
    p.add_argument("--no-refuse", action="store_true", help="не застосовувати поріг відмови")
    p.add_argument("--json", action="store_true", help="вивести повний результат у JSON")
    args = p.parse_args()

    result = answer(args.query, k=args.k, model=args.model, use_rag=not args.no_rag, hyde=args.hyde,
                    rerank=args.rerank, backend=args.backend, refuse=not args.no_refuse)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
