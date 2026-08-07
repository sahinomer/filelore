# Benchmarks

Generated JSON results under `benchmarks/results/` are ignored by Git.

## Test system

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 7 5800H |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU (8 GB) |

## CLIP embedding

```sh
uv run --extra embedding python -m benchmarks.clip_embedding
```

Use `--no-use-fast-processor` to benchmark with `use_fast=False`. Add
`--output benchmarks/results/clip.json` to save the full result locally.

Configuration: `openai/clip-vit-base-patch32`, 64 synthetic 512x512 images, 64
text prompts, batch sizes 1/8/16/32, 2 warm-up runs, and 5 measured runs. Only
`use_fast` changes between runs.

| Processor option | Single image median | Best batch | Best median/item | Best throughput |
| --- | ---: | ---: | ---: | ---: |
| `use_fast=True` | 10.903 ms | 16 | 4.637 ms | 215.678 items/s |
| `use_fast=False` | 13.822 ms | 16 | 7.352 ms | 136.026 items/s |
| Difference | 21.1% lower | - | 36.9% lower | 58.6% higher |

## CLAP embedding

```sh
uv run --extra embedding python -m benchmarks.clap_embedding
```

Add `--output benchmarks/results/clap.json` to save the full result locally.
The benchmark does not require a comparison option: its baseline score is the
best median audio-embedding throughput across the configured batch sizes.

Configuration: `laion/larger_clap_general`, 64 deterministic synthetic mono
waveforms at the model's sampling rate and maximum duration, 64 text prompts,
batch sizes 1/4/8/16, 2 warm-up runs, and 5 measured runs. Use
`--audio-duration` to run a shorter waveform workload on memory-constrained
systems.

| Model | Single audio median | Best batch | Best median/item | Best throughput |
| --- | ---: | ---: | ---: | ---: |
| `laion/larger_clap_general` | 44.887 ms | 16 | 29.205 ms | 34.240 items/s |

## Qdrant storage

Run the same workload in Python Local Mode and against a Qdrant service:

```sh
# Python Local Mode
uv run python -m benchmarks.qdrant_storage

# Qdrant service
uv run python -m benchmarks.qdrant_storage --url http://127.0.0.1:6333
```

Use `--payload-indexes` to create the benchmark's keyword and integer indexes.
Add `--output benchmarks/results/qdrant.json` to save a full result locally.

Configuration: 10,000 records, 128 dimensions, batch size 500, no payload
indexes, 10 warm-up queries, and 100 measured queries.

| Metric | Local | Service | Service comparison |
| --- | ---: | ---: | ---: |
| Ingestion | 45.728 s | 1.618 s | 28.26x faster |
| Metadata: extension median | 75.693 ms | 10.038 ms | 7.54x faster |
| Metadata: exact size median | 77.285 ms | 5.032 ms | 15.36x faster |
| Vector: unfiltered median | 4.835 ms | 5.209 ms | 7.7% slower |
| Vector: extension filter median | 74.163 ms | 14.836 ms | 5.00x faster |

Qdrant service is recommended for performance; local mode remains the default
so FileLore works without a separate service.

These are directional results from a normal development environment with
background tasks running; use them only for relative comparison.
