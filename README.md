# RAG_BigData

RAG-система інтелектуального пошуку по анотаціях arXiv (категорії `cs.*`): Spark-обробка корпусу, ембедінги BGE, FAISS-індекс, генерація відповіді LLM з посиланнями на джерела.

Розділи нижче пише автор відповідного модуля.

- [Студент 1 — дані та Spark](#студент-1--дані-та-spark)
- [Student 2 — embeddings and index](#student-2--embeddings-and-index)

Великі файли (повний Parquet, `.npy`, індекси, повний `catalog.sqlite`) у git не комітимо, вони лежать на Google Диску:

- корпус після Spark: [rag-arxiv/data](https://drive.google.com/drive/folders/1UIKPv4BlmjIRcoqzPf2z3tS_hA48XqxL?usp=sharing);
- ембедінги та індекси повного корпусу: [arxiv_student2_full](https://drive.google.com/drive/folders/1QDzrS4a2GRpZ4bLF4rbdsyWp31pT6vpH?usp=sharing).

## Студент 1 — дані та Spark

### Підготовка даних (`src/prepare_data.py`)
PySpark-пайплайн для фільтрації статей категорії `cs.*`, очищення тексту, дедуплікації (за `arxiv_id` та SHA-256) і нумерації `row_id` (`0..N-1`).
- **Запуск:** `python src/prepare_data.py --input <path_to_json> --output data/arxiv_cs.parquet --stats results/prep_stats.json`
- **Вхід:** `arxiv-metadata-oai-snapshot.json` (Kaggle).
- **Вихід:** очищений корпус `arxiv_cs.parquet` (1 058 983 статті, 12 файлів) та метрики відсіву `prep_stats.json`.
- **Повний корпус (Google Drive):** [<посилання на папку або zip>](https://drive.google.com/drive/folders/1UIKPv4BlmjIRcoqzPf2z3tS_hA48XqxL?usp=sharing)

## Student 2 — embeddings and index

This package implements the embedding and search layer of the team's RAG assistant. It includes the supplied sample, both English encoders' sample embeddings, three ready-to-query indexes for the selected model, and the measurements for Stage 4. The query encoder runs on the laptop. The Kaggle notebook handles GPU encoding when the full clean corpus arrives.

### Run a search on your Mac

Extract the ZIP. In Terminal, open the extracted `rag-arxiv` folder. Use Python 3.11 or 3.12 to create the environment. The first query downloads the pinned BGE encoder from Hugging Face.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.retriever "Which open-source system was adapted to recognize continuous Arabic speech?" -k 3
```

The default search uses BGE and HNSW on the bundled sample. To reproduce the whole experiment in a new directory on your own hardware, run the command below. It encodes all 1,000 papers with each model, compares the encoders, builds the three indexes for the winner, benchmarks them and prints a live search result. You can add `--device mps` for corpus encoding on Apple Silicon. The final index benchmark uses CPU.

```bash
python -m src.run_sample --work artifacts/my_run
```

For a notebook workflow, open `notebooks/student2_kaggle.ipynb`. Its default sample mode runs end to end. On Kaggle, attach the project ZIP as an input and import the notebook.

### What the sample showed

The supplied Parquet contains 1,000 unique arXiv papers, all dated 2007, with contiguous IDs from 0 to 999. The 18 English questions in `eval/sample_dev.json` were written for this package by paraphrasing individual abstracts. Each has one source paper. Use these questions for development and reserve Student 4's held-out questions for the final evaluation.

| Exact Flat search | Target ranked first | Recall@5 | Recall@10 | Full MRR |
| --- | ---: | ---: | ---: | ---: |
| all-MiniLM-L6-v2 | 14 / 18 | 100.0% | 100.0% | 0.8889 |
| bge-small-en-v1.5 | 15 / 18 | 100.0% | 100.0% | 0.9167 |

BGE wins the provisional comparison because it places the source paper higher on average. For the question about color assignments in a random network, for example, BGE ranked arXiv 0704.1269 first and MiniLM ranked it second. The decision uses full mean reciprocal rank, followed by Recall@10, Recall@1 and the supplied model order to resolve ties.

The index table uses the same BGE query vectors for every index. Its Recall@10 measures overlap with Flat's exact neighbors. The model table above measures whether the question's labeled source paper appears in the results.

| BGE index on 1,000 papers | Build (s) | Loaded RAM (MiB) | File (MiB) | Mean search (ms) | Recall@10 vs Flat |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat | 0.0035 | 1.587 | 1.465 | 0.0439 | 100.0% |
| HNSW | 0.1327 | 1.882 | 1.724 | 0.0667 | 100.0% |
| IVF-PQ | 0.0601 | 0.105 | 0.091 | 0.0446 | 61.1% |

HNSW returned the same ten paper IDs as Flat for every sample query. Flat had the lowest mean latency in this small run. IVF-PQ produced the smallest file and recovered 110 of the 180 exact neighbors. Use the full-corpus benchmark on the demo laptop for the final presentation.

`results/sample/` contains the measured CSVs, individual question ranks, model decision, hardware and parameter metadata, and the ready-to-copy `stage4_index_table.md`. New runs write their reports in `<work>/reports/`.

### Team README contribution

- `src/embed.py` reads the prepared Parquet and saves 384-dimensional, unit-normalized embeddings in resumable FP16 NumPy shards aligned by `row_id`.
- `src/build_index.py` reads these shards on CPU and builds FAISS Flat, HNSW and IVF-PQ indexes, recording training parameters, build times and file sizes.
- `src/retriever.py` exposes `search(query: str, k: int) -> list[dict]`. Each hit contains `row_id`, `arxiv_id`, `title`, `text` and `score`.
- `python -m src.run_sample --work artifacts/my_run` reproduces the model comparison and index measurements using `data/sample_1k.parquet` and `eval/sample_dev.json`.

### From Student 4's questions to the full corpus

1. **Select the encoder.** Ask Student 4 for an initial development subset. Put it in `eval/questions_dev.json` as a JSON array of objects with `question` and `arxiv_id` strings. The corpus used for the comparison must contain every labeled paper. If those papers come from the full corpus, use a development corpus that includes them and realistic distractors, or use the full clean corpus. Each input corpus needs dense `row_id` values from 0 through N−1.
2. **Encode on Kaggle.** In `notebooks/student2_kaggle.ipynb`, set `MODE = "full"`, point `FULL_PARQUET` to Student 1's Parquet file or Spark directory, and set `SELECTION_FILE` to the model decision from step 1. Enable a GPU. The notebook writes the metadata and embedding checkpoints under `/kaggle/working/arxiv_student2_full/`.
3. **Build and measure on the laptop.** Download the work folder and place its contents directly in `artifacts/full/`. Keep `catalog.sqlite` and the selected model's entire `embeddings/` directory together. Run the CPU build, benchmark and query commands below. Share the resulting Stage 4 table with Student 4.
4. **Share the retrieval layer.** Give Student 3 the source, requirements, catalog, and the selected index folder with its `build.json` and `index.faiss`. Give Student 4 the model decision, full index measurements and access to each index used for relevance evaluation. Preserve the directory layout under `artifacts/full/` when sharing these files. The supplied `.gitignore` keeps generated embeddings, indexes and full data out of git. The 1,000-paper sample and small measurement tables can go in the repository.

An example development record preserves leading zeros in the arXiv identifier.

```json
[{"question": "Which study adapted a speech recognizer for Arabic?", "arxiv_id": "0704.2083"}]
```

**Encoder selection.** The following command uses the packaged sample if the labels come from it. Change `--data` to the corresponding development Parquet file or directory when needed. A new experiment has its own `--work` directory.

```bash
python -m src.compare_models --data data/sample_1k.parquet --work artifacts/selection --questions eval/questions_dev.json --models minilm bge
```

**GPU encoding.** The notebook runs the equivalent of the command below. In this example BGE won model selection. Use the model recorded in the team's `selection.json`.

```bash
python -m src.embed --data /path/to/clean_parquet --work artifacts/full --model bge --device cuda --batch-size 128 --chunk-size 200000
```

**Full CPU indexes and measurements.** The builder infers the encoder when a work folder contains one model. If it contains several, pass `--model` explicitly or use the saved selection.

```bash
python -m src.build_index --work artifacts/full --threads 4
python -m src.benchmark --work artifacts/full --questions eval/questions.json --threads 1 --repeats 10
python -m src.retriever --work artifacts/full "How can neighboring sensors conserve battery power?" -k 5
```

### How the implementation works

**Data alignment.** The input schema is `row_id, arxiv_id, title, abstract, categories, year, text`. The code embeds Student 1's `text` field, which combines title and abstract. SQLite stores the lookup fields on disk. Its integer primary key orders rows across out-of-order Spark parts. Duplicate IDs, empty search text, gaps and mismatched corpus fingerprints raise an error before new indexes are published.

**Embeddings and checkpoints.** Both English models have 384 output dimensions. BGE adds its retrieval instruction to questions. MiniLM uses plain questions. The optional `e5` model supports a multilingual demo and applies its own query and passage prefixes. Each model uses a fixed revision and its native tokenizer and sequence limit. The encoder normalizes in FP32 and writes FP16 for storage. FAISS receives re-normalized FP32 vectors after reading each shard. [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5) · [Multilingual E5 model card](https://huggingface.co/intfloat/multilingual-e5-small)

**Recovery.** Each shard records its row interval and SHA-256. A retry verifies completed shards and regenerates missing or damaged ones. Keep the same source Parquet, model revision and shard size. You can lower the inference batch size after a GPU memory error. Save or download the work folder before a Kaggle session ends. A later session can restore it using the notebook's `RESUME_FROM` setting.

**Index parameters.** Flat searches every vector. HNSW uses 32 graph neighbors, `efConstruction=200` and `efSearch=128`. IVF-PQ samples up to 200,000 training vectors across all shards with seed 42. At sufficient scale it uses 4,096 lists, 32 probes and 48 eight-bit subquantizers. On the 1,000-paper sample it uses 25 lists, eight probes and 48 four-bit subquantizers. This gives each centroid enough training examples. [FAISS indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) · [FAISS clustering guidance](https://github.com/facebookresearch/faiss/wiki/FAQ#can-i-ignore-warning-clustering-xxx-points-to-yyy-centroids)

**Benchmark.** Build time covers creating, training and filling each index and stops before serialization. RAM measures the change in resident memory in a fresh worker. For small indexes, the worker loads several copies and divides the change by the number of copies. Serialized size comes directly from the file. Mean latency and p95 cover one warm CPU FAISS search at a time, after query encoding and before metadata lookup. Each question gets one warm-up and ten timed searches by default. Approximate Recall@10 is the mean of the size of the intersection of the two top-ten ID sets divided by ten. At corpus sizes below ten, both the cutoff and denominator are the corpus size.

### Python interface for the team

Run the team's scripts as package modules, for example `python -m src.cli` once Student 3 adds that file. The ordinary import is available from the project root.

```python
from src.retriever import search

hits = search("Can a digital library recommend papers from reading activity?", k=3)
for hit in hits:
    print(hit["arxiv_id"], hit["title"], round(hit["score"], 4))
```

The return value is a ranked list of plain dictionaries. IDs such as `0704.2963` remain strings. Higher scores indicate stronger matches. IVF-PQ reports approximate inner products. A positive `k` is capped at the corpus size. The default function caches its encoder and index between calls. Configure the full demo before starting Student 3's app.

```bash
export RAG_WORK="$PWD/artifacts/full"
export RAG_MODEL=bge
export RAG_INDEX=hnsw
```

Student 4 can select an index directly for per-model evaluation.

```python
from src.retriever import Retriever

retriever = Retriever(work="artifacts/full", model="bge", index="ivfpq")
hits = retriever.search("How do sensor networks extend battery lifetime?", k=10)
```

To create all three index variants for an additional encoder, first encode it and run `python -m src.build_index --work artifacts/full --model minilm`. To check the supplied row-alignment regressions, run `python -m unittest discover -s tests -v`.
