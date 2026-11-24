"""Regression checks for the row order shared by Spark, FAISS and the metadata lookup."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.common import DEFAULT_WORK, SAMPLE, read_json
from src.corpus import connect_catalog, prepare_catalog
from src.retriever import open_index


class AlignmentTests(unittest.TestCase):
    def test_shuffled_spark_parts(self):
        source = pq.read_table(SAMPLE)
        shuffled = source.take(np.random.default_rng(42).permutation(len(source)))
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            parts = Path(directory) / "spark"
            parts.mkdir()
            pq.write_table(shuffled.slice(0, 400), parts / "part-a.parquet")
            pq.write_table(shuffled.slice(400), parts / "part-b.parquet")
            prepare_catalog(parts, work)
            with closing(connect_catalog(work)) as db:
                actual = db.execute("SELECT row_id, arxiv_id, title, text FROM papers ORDER BY row_id").fetchall()
            cols = ("row_id", "arxiv_id", "title", "text")
            expected = list(zip(*(source[col].to_pylist() for col in cols)))
            self.assertEqual(actual, expected)

    def test_year_stored_as_hive_partition(self):
        sample = pq.read_table(SAMPLE).slice(0, 3)
        with tempfile.TemporaryDirectory() as directory:
            parts = Path(directory) / "spark"
            partition = parts / "year=2007"
            partition.mkdir(parents=True)
            pq.write_table(sample.drop(["year"]), partition / "part-0.parquet")
            self.assertEqual(prepare_catalog(parts, Path(directory) / "work")["rows"], 3)

    def test_rejects_duplicate_and_missing_ids(self):
        sample = pq.read_table(SAMPLE).slice(0, 2)
        cases = [(pa.concat_tables([sample, sample]), sqlite3.IntegrityError), (sample.slice(1), ValueError)]
        for table, error in cases:
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                data, work = Path(directory) / "bad.parquet", Path(directory) / "work"
                pq.write_table(table, data)
                with self.assertRaises(error):
                    prepare_catalog(data, work)
                self.assertFalse((work / "catalog.sqlite").exists())

    def test_flat_ids_match_shard_boundaries(self):
        work = DEFAULT_WORK
        if not (work / "selection.json").is_file():
            self.skipTest("Build the sample artifacts to check encoded row alignment.")
        key = read_json(work / "selection.json")["model"]
        index = open_index(work, key, "flat")
        folder = work / key / "embeddings"
        for filename, part in read_json(folder / "manifest.json")["parts"].items():
            vectors = np.load(folder / filename, mmap_mode="r")
            for row_id in [part["start"], part["stop"] - 1]:
                vector = vectors[row_id - part["start"]].astype("float32")
                np.testing.assert_allclose(index.reconstruct(row_id),
                                           vector / np.linalg.norm(vector), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
