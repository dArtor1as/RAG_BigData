import argparse
import json
import os
import time
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, StructField, StructType


def build_spark(driver_memory: str, shuffle_partitions: int) -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName("arXiv-RAG-PrepareData")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


def get_input_schema() -> StructType:
    """Читаємо лише потрібні поля, щоб економити RAM і час парсингу."""
    return StructType([
        StructField("id", StringType(), True),
        StructField("title", StringType(), True),
        StructField("abstract", StringType(), True),
        StructField("categories", StringType(), True),
        StructField("update_date", StringType(), True),
        StructField(
            "versions",
            ArrayType(StructType([StructField("created", StringType(), True)])),
            True,
        ),
    ])


def assign_contiguous_row_id(df, num_partitions: int):
    """
    Призначає суворо послідовний row_id (0 .. N-1) без злиття в одну партицію
    та без вивантаження даних у Python-воркери.
    """
    df_part = (
        df.repartitionByRange(num_partitions, "arxiv_id")
        .sortWithinPartitions("arxiv_id")
        .withColumn("_pid", F.spark_partition_id())
        .withColumn("_mid", F.monotonically_increasing_id())
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    part_counts = (
        df_part.groupBy("_pid").count().orderBy("_pid").collect()
    )

    offsets = {}
    running_total = 0
    for row in part_counts:
        offsets[int(row["_pid"])] = running_total
        running_total += int(row["count"])

    if not offsets:
        df_part.unpersist()
        return df.withColumn("row_id", F.lit(0).cast("long")).limit(0)

    map_args = []
    for pid, offset in offsets.items():
        map_args.extend([F.lit(pid), F.lit(offset).cast("long")])
    offset_map = F.create_map(*map_args)

    w_part = Window.partitionBy("_pid").orderBy("_mid")
    df_indexed = (
        df_part.withColumn(
            "row_id",
            (offset_map[F.col("_pid")] + F.row_number().over(w_part) - F.lit(1)).cast("long"),
        )
        .select("row_id", "arxiv_id", "title", "abstract", "categories", "year", "text")
        .repartitionByRange(num_partitions, "row_id")
        .sortWithinPartitions("row_id")
    )
    return df_indexed, df_part


def main():
    parser = argparse.ArgumentParser(description="Підготовка корпусу arXiv cs.* на PySpark")
    parser.add_argument("--input", required=True, help="Шлях до arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--output", required=True, help="Шлях до вихідної папки Parquet")
    parser.add_argument("--stats", required=True, help="Шлях до вихідного JSON із метриками")
    parser.add_argument("--num-partitions", type=int, default=12, help="Кількість Parquet-файлів")
    parser.add_argument("--driver-memory", default="6g", help="Пам'ять драйвера Spark (напр., 6g або 8g)")
    parser.add_argument("--shuffle-partitions", type=int, default=32, help="spark.sql.shuffle.partitions")
    parser.add_argument("--min-abstract-len", type=int, default=100, help="Мінімальна довжина анотації")
    args = parser.parse_args()

    total_start = time.perf_counter()
    spark = build_spark(args.driver_memory, args.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")

    funnel = []
    timings = {}

    def record_step(step_id: str, label: str, count_val: int, step_sec: float):
        prev_count = funnel[-1]["count"] if funnel else count_val
        raw_total = funnel[0]["count"] if funnel else count_val
        dropped = prev_count - count_val
        funnel.append({
            "step": step_id,
            "label": label,
            "count": int(count_val),
            "dropped": int(dropped),
            "kept_pct_of_raw": round(100.0 * count_val / raw_total, 2) if raw_total else 0.0,
            "step_time_sec": round(step_sec, 2),
        })
        timings[step_id] = round(step_sec, 2)
        print(f"[{step_id}] залишилось: {count_val:,} | відсіяно: {dropped:,} | час: {step_sec:.2f} с")

    try:
        # 1. Читання сирого JSON
        t0 = time.perf_counter()
        df_raw = spark.read.schema(get_input_schema()).json(args.input)
        raw_count = df_raw.count()
        record_step("raw_json", "Сирий датасет arXiv", raw_count, time.perf_counter() - t0)

        # 2. Фільтр категорій cs.*
        t0 = time.perf_counter()
        df_cs = (
            df_raw.withColumn("categories", F.trim(F.coalesce(F.col("categories"), F.lit(""))))
            .filter(F.col("categories").rlike(r"(^|\s)cs\."))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        cs_count = df_cs.count()
        record_step("filter_cs", "Фільтр категорій cs.*", cs_count, time.perf_counter() - t0)

        # 3. Очищення тексту, фільтр коротких/порожніх анотацій та витяг року
        t0 = time.perf_counter()
        first_created = F.when(
            F.size(F.col("versions")) > 0,
            F.col("versions").getItem(0).getField("created"),
        ).otherwise(F.lit(""))

        year_from_ver = F.regexp_extract(
            F.coalesce(first_created, F.lit("")), r"\b((?:19|20)\d{2})\b", 1
        )
        year_from_upd = F.regexp_extract(
            F.coalesce(F.col("update_date"), F.lit("")), r"\b((?:19|20)\d{2})\b", 1
        )
        year_expr = (
            F.when(year_from_ver != "", year_from_ver)
            .when(year_from_upd != "", year_from_upd)
            .cast("long")
        )

        df_cleaned = (
            df_cs.select(
                F.trim(F.coalesce(F.col("id"), F.lit(""))).alias("arxiv_id"),
                F.trim(F.regexp_replace(F.coalesce(F.col("title"), F.lit("")), r"\s+", " ")).alias("title"),
                F.trim(F.regexp_replace(F.coalesce(F.col("abstract"), F.lit("")), r"\s+", " ")).alias("abstract"),
                F.col("categories"),
                year_expr.alias("year"),
            )
            .filter(
                (F.col("arxiv_id") != "")
                & (F.col("title") != "")
                & (F.length(F.col("abstract")) >= args.min_abstract_len)
                & F.col("year").isNotNull()
            )
            .withColumn("text", F.concat_ws(". ", F.col("title"), F.col("abstract")))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        cleaned_count = df_cleaned.count()
        df_cs.unpersist()
        record_step(
            "clean_and_validate",
            f"Очищення тексту, len(abstract) >= {args.min_abstract_len}, валідний рік",
            cleaned_count,
            time.perf_counter() - t0,
        )

        # 4. Дедуплікація за arxiv_id
        t0 = time.perf_counter()
        df_dedup_id = df_cleaned.dropDuplicates(["arxiv_id"]).persist(StorageLevel.MEMORY_AND_DISK)
        dedup_id_count = df_dedup_id.count()
        df_cleaned.unpersist()
        record_step("dedup_arxiv_id", "Дедуплікація за arxiv_id", dedup_id_count, time.perf_counter() - t0)

        # 5. Дедуплікація за SHA-256 хешем тексту
        t0 = time.perf_counter()
        df_dedup_hash = (
            df_dedup_id.withColumn("_text_hash", F.sha2(F.lower(F.col("text")), 256))
            .dropDuplicates(["_text_hash"])
            .drop("_text_hash")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        dedup_hash_count = df_dedup_hash.count()
        df_dedup_id.unpersist()
        record_step("dedup_text_hash", "Дедуплікація за хешем тексту (SHA-256)", dedup_hash_count, time.perf_counter() - t0)

        # 6. Призначення row_id (0 .. N-1) та запис у кілька Parquet-файлів
        t0 = time.perf_counter()
        df_final, df_part_ref = assign_contiguous_row_id(df_dedup_hash, args.num_partitions)
        df_final.write.mode("overwrite").parquet(args.output)
        df_part_ref.unpersist()
        df_dedup_hash.unpersist()
        write_sec = round(time.perf_counter() - t0, 2)
        timings["assign_row_id_and_write_parquet"] = write_sec
        print(f"[write_parquet] збережено у {args.output} за {write_sec:.2f} с")

        total_sec = round(time.perf_counter() - total_start, 2)
        timings["total_pipeline_sec"] = total_sec

        # Підрахунок розміру вихідних Parquet-файлів
        parquet_files = sorted(Path(args.output).glob("*.parquet"))
        total_size_mb = round(sum(p.stat().st_size for p in parquet_files) / (1024**2), 2)

        stats_payload = {
            "config": {
                "input_path": os.path.basename(args.input),
                "output_dir": os.path.basename(args.output),
                "spark_version": spark.version,
                "driver_memory": args.driver_memory,
                "shuffle_partitions": args.shuffle_partitions,
                "num_partitions": args.num_partitions,
                "min_abstract_len": args.min_abstract_len,
            },
            "funnel": funnel,
            "timings_sec": timings,
            "output_info": {
                "final_rows": int(dedup_hash_count),
                "num_parquet_files": len(parquet_files),
                "total_size_mb": total_size_mb,
                "columns": ["row_id", "arxiv_id", "title", "abstract", "categories", "year", "text"],
            },
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.stats)), exist_ok=True)
        with open(args.stats, "w", encoding="utf-8") as f:
            json.dump(stats_payload, f, ensure_ascii=False, indent=2)

        print(f"\nГотово! Загальний час: {total_sec:.2f} с")
        print(f"Файлів Parquet: {len(parquet_files)}, розмір: {total_size_mb} МБ")
        print(f"Метрики збережено у: {args.stats}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
