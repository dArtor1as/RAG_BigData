"""Build one disk-backed row_id -> paper lookup from Parquet or Spark part files."""
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

from .common import sha256

REQUIRED = {"row_id", "arxiv_id", "title", "abstract", "categories", "year", "text"}
USED = ["row_id", "arxiv_id", "title", "text"]


def connect_catalog(work):
    path = (Path(work) / "catalog.sqlite").resolve()
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def catalog_info(work):
    with closing(connect_catalog(work)) as db:
        digest, count = db.execute("SELECT source_sha256, rows FROM corpus").fetchone()
    return {"source_sha256": digest, "rows": count}


def prepare_catalog(source, work):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source, work = Path(source), Path(work)
    if source.is_file():
        files = [source]
    else:
        files = sorted(p for p in source.rglob("*.parquet") if p.is_file() and
                       not any(s.startswith((".", "_")) for s in p.relative_to(source).parts))
    if not files:
        raise FileNotFoundError(f"No Parquet data found at {source}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(bytes.fromhex(sha256(file)))
    source_hash = digest.hexdigest()
    work.mkdir(parents=True, exist_ok=True)
    target = work / "catalog.sqlite"
    with FileLock(str(work / ".catalog.lock")):
        if target.exists():
            info = catalog_info(work)
            if info["source_sha256"] != source_hash:
                raise ValueError("This work directory belongs to different Parquet data. Choose another --work.")
            return info
        with TemporaryDirectory(dir=work) as temp:
            pending = Path(temp) / "catalog.sqlite"
            with closing(sqlite3.connect(pending)) as db:
                db.execute("""CREATE TABLE papers (
                    row_id INTEGER PRIMARY KEY, arxiv_id TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL, text TEXT NOT NULL)""")
                for file in files:
                    parquet = pq.ParquetFile(file)
                    # Spark may keep non-retrieval fields such as year in Hive directory names.
                    partitions = {part.split("=", 1)[0] for part in file.relative_to(source).parts[:-1]
                                  if "=" in part} if source.is_dir() else set()
                    stored = set(parquet.schema_arrow.names)
                    missing = (REQUIRED - stored - partitions) | (set(USED) - stored)
                    if missing:
                        raise ValueError(f"{file.name} is missing {sorted(missing)}")
                    if not pa.types.is_integer(parquet.schema_arrow.field("row_id").type):
                        raise ValueError(f"{file.name}: row_id must be an integer.")
                    for batch in parquet.iter_batches(batch_size=4096, columns=USED):
                        columns = batch.to_pydict()
                        rows = list(zip(*(columns[name] for name in USED)))
                        for row_id, arxiv_id, title, text in rows:
                            if row_id is None or not all(isinstance(v, str) and v.strip() for v in (arxiv_id, title, text)):
                                raise ValueError(f"{file.name}: invalid metadata at row_id {row_id!r}.")
                        db.executemany("INSERT INTO papers VALUES (?, ?, ?, ?)", rows)
                n, first, last = db.execute("SELECT count(*), min(row_id), max(row_id) FROM papers").fetchone()
                if n == 0 or first != 0 or last != n - 1:
                    raise ValueError("Expected unique row_id values covering 0 through N-1.")
                db.execute("CREATE TABLE corpus (source_sha256 TEXT, rows INTEGER)")
                db.execute("INSERT INTO corpus VALUES (?, ?)", (source_hash, n))
                db.commit()
            pending.replace(target)
    return {"source_sha256": source_hash, "rows": n}
