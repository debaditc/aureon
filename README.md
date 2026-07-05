# aureon
Aureon — the Adaptive Unified Retrieval Engine — is a small, open-source Python package. It runs keyword search (BM25, "sparse") and vector search (embeddings, "dense") in parallel, then lets a per-query router decide how much to trust each one, at runtime. Every ranking decision comes back with a full Explain-Mode breakdown.
