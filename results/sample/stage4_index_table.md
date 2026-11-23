# Stage 4 · Methods and models

**1,000 papers · 18 questions · BAAI/bge-small-en-v1.5**

Each paper contributes one normalized 384-dimensional embedding of its title and abstract. FAISS ranks documents by inner product. Flat supplies the exact reference ranking.

HNSW uses M=32, efConstruction=200 and efSearch=128. IVF-PQ uses 25 lists, probes 8 and encodes each vector in 48 groups of 4 bits. It trained on 1,000 vectors.

| Index | Build (s) | RAM (MiB) | File (MiB) | Mean search (ms) | Recall@10 vs Flat |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat | 0.0035 | 1.587 | 1.465 | 0.0439 | 100.0% |
| HNSW | 0.1327 | 1.882 | 1.724 | 0.0667 | 100.0% |
| IVF-PQ | 0.0601 | 0.105 | 0.091 | 0.0446 | 61.1% |

Each latency measurement times a single pre-encoded question on CPU. The benchmark uses 1 FAISS search thread(s), 10 timed passes and one warm-up pass. RAM estimates the resident memory consumed by loading an index into a fresh process. Recall is the mean fraction of Flat's top 10 row IDs that each index recovers.
