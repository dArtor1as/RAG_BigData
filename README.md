# RAG_BigData

RAG-система інтелектуального пошуку по анотаціях arXiv (категорії `cs.*`): Spark-обробка корпусу, ембедінги BGE, FAISS-індекс, генерація відповіді LLM з посиланнями на джерела.

Розділи нижче пише автор відповідного модуля.

- [Студент 1 — дані та Spark](#студент-1--дані-та-spark)
- [Student 2 — embeddings and index](#student-2--embeddings-and-index)
- [Студент 3 — запит і генерація](#студент-3--запит-і-генерація)

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

## Студент 3 — запит і генерація

Усе між питанням і відповіддю: покращення пошуку на рівні запиту, збирання контексту, генерація з посиланнями `[1]`, `[2]` і перевірка, чи підтверджують їх джерела. Кожне покращення вмикається прапорцем, щоб його внесок можна було виміряти окремо.

### Запуск

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen3:4b-instruct
ollama pull qwen3:0.6b
python -m src.cli "How can neighboring sensors conserve battery power?" --rerank
streamlit run src/app.py
```

За замовчуванням працює вибірка `artifacts/sample`. Для повного корпусу покладіть з Google Диску `catalog.sqlite` і `bge/indexes/hnsw/` в `artifacts/full/` (з тією ж структурою) і задайте змінні:

```powershell
$env:RAG_WORK = "artifacts/full"; $env:RAG_MODEL = "bge"; $env:RAG_INDEX = "hnsw"
Copy-Item results/full/thresholds.json artifacts/full/thresholds.json
```

Ембедінги (`bge/embeddings/*.npy`) для запитів не потрібні: HNSW-індекс уже містить усі вектори. Індекс займає ~2 ГБ RAM і завантажується ~11 с при першому запиті. Пороги відмови для повного корпусу лежать у `results/full/thresholds.json`; без цього файлу беруться значення за замовчуванням (0.60 / 0.70), а для повного корпусу вони занизькі.

Gemini замість Ollama: `$env:GEMINI_API_KEY = "..."` і `--backend gemini` (або перемикач у Streamlit). Ключ не комітимо.

### Модулі

- **`src/generator.py`** — `answer(query, k, model, use_rag, hyde, rerank, backend)` повертає `{"answer", "sources", ...}`. Джерела нумеруються `[1]…[k]`; LLM відповідає лише за ними, мовою питання, з посиланнями, або повертає `NOT_FOUND`. Додаткові ключі: `refused`, `refusal_reason`, `claims`, `supported_ratio`, `timings`, `hyde_passage`. Це спільна точка входу для CLI, Streamlit і веб-сервісу.
- **`src/llm.py`** — `generate(prompt, system, model, backend)`: Ollama (`qwen3:4b-instruct`, `qwen3:0.6b` з вимкненим режимом міркувань) або Gemini. Ту саму функцію можна використати для LLM-судді.
- **`src/query.py`** — `retrieve(query, k, hyde, use_rerank)`:
  - **HyDE** (Gao et al., 2022): LLM пише гіпотетичну анотацію, і пошук іде за середнім вектором анотації та питання;
  - **reranker** `cross-encoder/ms-marco-MiniLM-L-6-v2` переоцінює топ-50 і лишає топ-k;
  - **дворівневий поріг** за косинусом питання з найкращою анотацією:
    - нижче `low` — відмова без виклику LLM;
    - між `low` і `high` — відповідь із попередженням (`confidence="uncertain"`);
    - вище `high` — звичайна відповідь.

    Оцінку reranker'а для порогу не використовуємо: на наукових перефразуваннях вона буває дуже низькою (−9…−10) навіть для правильної статті.

  Пороги підбирає `python -m src.query --calibrate` (для повного корпусу додайте `--inside eval/sample_dev.json eval/dev_full.json`). `low` — найменша схожість серед питань з корпусу мінус 0.03, тобто жодне таке питання не відкидається. `high` — 95-й перцентиль питань поза корпусом (`eval/offtopic.json`, 49 штук). Результат зберігається в `<RAG_WORK>/thresholds.json`.
- **`src/dev_questions.py`** — `python -m src.dev_questions --n 100` бере випадкові статті корпусу, і LLM пише до кожної пошукове питання своїми словами (`eval/dev_full.json`). Це набір для калібрування; фінальне оцінювання — на питаннях студента 4.
- **`src/eval_modes.py`** — `python -m src.eval_modes --questions eval/sample_dev.json eval/dev_full.json --out results/full` для кожного питання знаходить місце цільової статті серед 50 кандидатів у чотирьох режимах пошуку. Пише `eval_modes.csv` (по питаннях) і `eval_modes.md` (підсумок).
- **`src/citations.py`** — `verify_citations()`: ділить відповідь на речення і рахує косинус кожного речення з процитованою анотацією (енкодер BGE). Статуси: `ok`, `no_citation`, `bad_index`, `low_similarity`.
- **`src/profiling.py`** — час кожного етапу (`hyde`, `encode`, `search`, `rerank`, `llm`, `verify`). `python -m src.profiling --limit 10` записує `results/profiling.csv` і `results/profiling.md`.
- **`src/cli.py`**, **`src/app.py`** — консоль і Streamlit:
  - перемикачі моделі, k, RAG, HyDE, reranker і порогу;
  - джерела зі score і посиланням на arXiv;
  - непідтверджені речення підсвічені.

Тести без LLM та індексу: `python -m unittest discover -s tests -v`.

### Результати на вибірці (1000 статей)

**Пороги відмови** на вибірці (18 питань з корпусу, 49 поза ним): `low` = `high` = 0.600. Найслабше питання з корпусу має схожість 0.630, тож 96% чужих питань відсікаються ще до LLM, а сірої зони немає.

**Перевірка цитат** на 36 відповідях (18 питань, дві моделі, 119 речень з посиланнями), поріг 0.60:

| Анотація, з якою порівнюємо речення | Медіана схожості | Позначено «не підтверджено» |
| --- | ---: | ---: |
| процитована | 0.70 | 32% |
| інша з того ж топ-5 (та сама тема) | 0.62 | 43% |
| випадкова стаття корпусу | 0.52 | 82% |

Перевірка надійно ловить посилання на статтю з іншої теми, але погано відрізняє сусідню статтю з тієї ж теми.

**Моделі.** `qwen3:4b-instruct` ставить посилання після кожного твердження. `qwen3:0.6b` часто ставить одне посилання в кінці або пише «the answer is [1]», тож частка підтверджених речень у неї нижча.

**Профілювання** (10 питань, k=5, RTX 3050 Laptop 4 ГБ, середні секунди на питання; повна таблиця — `results/profiling.md`):

| Модель | Конфігурація | HyDE | Reranker | LLM | Разом |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen3:0.6b | base | — | — | 0.69 | 0.89 |
| qwen3:0.6b | +hyde+rerank | 0.43 | 2.20 | 0.79 | 3.69 |
| qwen3:4b-instruct | base | — | — | 4.69 | 4.89 |
| qwen3:4b-instruct | +rerank | — | 2.25 | 4.76 | 7.21 |
| qwen3:4b-instruct | +hyde | 5.01 | — | 4.73 | 10.05 |
| qwen3:4b-instruct | +hyde+rerank | 5.74 | 3.01 | 5.46 | 14.67 |

Векторний пошук і кодування питання на вибірці займають соті частки секунди. Майже весь час іде на LLM. HyDE з моделлю 4B — це ще один виклик LLM, тож час запиту приблизно подвоюється. Reranker на CPU додає 2–3 с.

### Результати на повному корпусі (1 058 983 статті, HNSW)

Попередня оцінка пошуку (офіційні метрики — у студента 4): місце цільової статті для 18 ручних питань студента 2 і 100 синтетичних (`src/dev_questions.py`). Місця по кожному питанню — у [`results/full/eval_modes.csv`](results/full/eval_modes.csv); порожня клітинка означає, що статті немає серед 50 кандидатів.

| Пошук | Ручні 18: R@1 | Ручні 18: R@5 | Синтетичні 100: R@1 | Синтетичні 100: R@5 | Разом MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 6 | 10 | 73 | 83 | 0.713 |
| +rerank | 7 | 14 | 86 | 96 | 0.842 |
| +hyde | 8 | 11 | 70 | 85 | 0.720 |
| +hyde+rerank | 8 | 14 | 87 | 94 | 0.850 |

Reranker допомагає на обох наборах. HyDE помітний лише на ручних питаннях: вони навмисно перефразовані, і вигадана анотація повертає їм лексику статей. Синтетичні питання LLM писала просто з анотації, тож вони й так близькі до неї, і HyDE може навіть трохи зсунути правильну статтю вниз. На вибірці всі 18 ручних питань знаходились у топ-5; серед мільйона статей простий пошук знаходить лише 10.

Питання для демонстрації (спершу без покращень, потім з указаним режимом; перевірено повними відповідями qwen3:4b-instruct):

| Режим | Питання | Що видно |
| --- | --- | --- |
| +rerank | What happens to the set of valid vertex color assignments when more edges are added to a sparse random network? | без reranker'а стаття на 36-му місці і LLM відмовляє; з ним — перша |
| +rerank | How do large language models strategically lie, and can current safety tools detect such deception? | без reranker'а відповідь правдоподібна, але з дотичних статей; з ним головна стаття перша |
| +hyde | Is ambiguity in derivatives equivalent to Carlet-Ding's NB_F parameter? | без HyDE статті немає в топ-50 і LLM відмовляє; з HyDE — перша (3 з 3 запусків) |
| +hyde | How does Minimum Message Length compare to AIC, BIC, and HQ in ARMA model selection? | те саме, плюс попередження сірої зони (0.71) |
| +hyde+rerank | How can discontinuities be distinguished from measurement noise when the available information is a truncated spectrum? | лише з reranker'ом стаття третя, але LLM відмовляє; разом із HyDE — відповідь |

Питання «How can neighboring sensors conserve battery power?» для демо на повному корпусі не підходить: стаття, під яку його писали на вибірці, тут на 520-му місці серед сотень схожих робіт про енергію сенсорів.

**Відмова: поріг + LLM.** Серед мільйона CS-статей майже на будь-яку тему знаходиться дотична стаття: на питання про мРНК-вакцини — стаття про дизайн мРНК (0.75), про правила крикету — щось про крикет (0.75). Тому одного порогу мало, і рішення приймається у два кроки.

Пороги для повного корпусу (`results/full/thresholds.json`) підібрані на 118 питаннях з корпусу (18 ручних і 100 синтетичних) і 49 поза ним: `low` = 0.666, `high` = 0.741.

| | Нижче `low` | Між порогами | Вище `high` |
| --- | ---: | ---: | ---: |
| питання з корпусу | 0% | 3% | 97% |
| питання поза корпусом | 67% | 27% | 6% |

Що питання, які пройшли поріг, отримують у підсумку (4B, без reranker'а):

| | Відмова: поріг | Відмова: LLM | Відповідь з попередженням | Відповідь |
| --- | ---: | ---: | ---: | ---: |
| 49 питань поза корпусом | 33 | 12 | 3 | 1 |
| 50 питань з корпусу | 0 | 11 | 0 | 39 |

З 11 відмов LLM на питаннях з корпусу 10 трапились, коли пошук не приніс потрібну статтю в топ-5. Модель тоді каже «немає відповіді», а не відповідає за чужими статтями. Хибна відмова, коли потрібна стаття була серед джерел, одна. Модель відмовляє або словом `NOT_FOUND`, або фразою «The provided sources do not…»; обидва варіанти розпізнаються як відмова.

Окремо, з вимкненим порогом і reranker'ом на 7 питаннях з корпусу і 6 поза ним:

| Модель | Хибні відмови (з корпусу) | Відмови на питаннях поза корпусом | Підтверджено речень |
| --- | ---: | ---: | ---: |
| qwen3:4b-instruct | 0/7 | 3/6 | 86% |
| qwen3:0.6b | 1/7 | 4/6 | 50% |

Слово `NOT_FOUND` модель 0.6B не пише ніколи, а відмовляє лише фразами на кшталт «The sources do not…». У відповідях 0.6B підтверджено лише половину речень проти 86% у 4B.

**Формат контексту.** Джерела подаються як `Source [n] / Title / Abstract`, а промпт закінчується вимогою писати своїми словами з посиланнями `[n]` або відповісти `NOT_FOUND`. Зі старим форматом `[n] (arXiv:id) текст` модель 0.6B часто просто копіювала шапки джерел у відповідь.

**Профілювання** 4B на повному корпусі (`results/full/profiling.md`): пошук у HNSW займає 5–40 мс, решта етапів — як на вибірці: base 3.5 с, +rerank 8.3 с, +hyde 10.0 с, +hyde+rerank 12.4 с на питання.

### Обмеження

- `ms-marco` reranker і енкодер BGE навчені англійською. Для питань українською пошук і перевірка цитат працюють гірше; варіант — модель `e5` (multilingual), яку вже підтримує `src/common.py`.
- Перевірка цитат міряє схожість тексту, а не логічне слідування, тому може пропустити перекручений факт зі «схожими» словами.
- Тег `qwen3:4b` в Ollama — це Qwen3-4B-Thinking-2507: вона завжди міркує і ігнорує `think=False`, тому беремо `qwen3:4b-instruct`.
