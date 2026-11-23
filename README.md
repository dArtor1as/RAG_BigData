# RAG_BigData

### Підготовка даних (`src/prepare_data.py`)
PySpark-пайплайн для фільтрації статей категорії `cs.*`, очищення тексту, дедуплікації (за `arxiv_id` та SHA-256) і нумерації `row_id` (`0..N-1`).
- **Запуск:** `python src/prepare_data.py --input <path_to_json> --output data/arxiv_cs.parquet --stats results/prep_stats.json`
- **Вхід:** `arxiv-metadata-oai-snapshot.json` (Kaggle).
- **Вихід:** очищений корпус `arxiv_cs.parquet` (1 058 983 статті, 12 файлів) та метрики відсіву `prep_stats.json`.
- **Повний корпус (Google Drive):** [<посилання на папку або zip>](https://drive.google.com/drive/folders/1UIKPv4BlmjIRcoqzPf2z3tS_hA48XqxL?usp=sharing)
