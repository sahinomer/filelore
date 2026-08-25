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

| Workload | Processor option | Single-item median | Sequential median/item | Sequential throughput | Best batch | Best median/item | Best throughput |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image | `use_fast=True` | 10.903 ms | 12.448 ms | 80.333 items/s | 16 | 4.637 ms | 215.678 items/s |
| Image | `use_fast=False` | 13.822 ms | 14.533 ms | 68.807 items/s | 16 | 7.352 ms | 136.026 items/s |
| Text | `use_fast=True` | 10.052 ms | 8.843 ms | 113.082 items/s | 32 | 0.531 ms | 1882.453 items/s |
| Text | `use_fast=False` | 9.954 ms | 10.414 ms | 96.022 items/s | 32 | 0.842 ms | 1187.726 items/s |

For the recorded Transformers 4.57.6 runs, `use_fast` is forwarded to both
the CLIP image processor and `AutoTokenizer`, so it is applicable to both
workloads. Compared with the legacy option, the fast image processor reduced
single-image latency by 21.1% and best-batch latency per item by 36.9%, while
increasing best image throughput by 58.6%.

`Single-item median` represents one direct prediction. The sequential columns
measure sustained processing of all 64 inputs with model batch size 1 and are
included because FileLore currently embeds one search text query per request.

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

| Workload | Single-item median | Sequential median/item | Sequential throughput | Best batch | Best median/item | Best throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Audio | 44.887 ms | 49.812 ms | 20.075 items/s | 16 | 29.205 ms | 34.240 items/s |
| Text | 11.129 ms | 11.244 ms | 88.933 items/s | 16 | 0.914 ms | 1094.335 items/s |

## Harrier embedding

```sh
uv run --extra embedding python -m benchmarks.harrier_embedding
```

Add `--output benchmarks/results/harrier.json` to save the full result locally.
The baseline score is the best median document-embedding throughput among the
configured batch sizes.

Configuration: `microsoft/harrier-oss-v1-270m`, 64 deterministic synthetic
multilingual document chunks targeting the chunker's 1,600-character maximum,
64 multilingual queries using the `web_search_query` prompt, batch sizes
1/8/16/32, 2 warm-up runs, and 5 measured runs.

| Workload | Single-item median | Sequential median/item | Sequential throughput | Best batch | Best median/item | Best throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Document chunk | 50.248 ms | 47.824 ms | 20.910 items/s | 32 | 7.759 ms | 128.875 items/s |
| Query | 49.555 ms | 46.584 ms | 21.467 items/s | 32 | 1.697 ms | 589.188 items/s |

Batch size 32 was the best tested size for both workloads. Compared with model
batch size 1, it increased document throughput by 6.16x and query throughput by
27.45x. Peak allocated GPU memory was 739.382 MB for the document workload at
that batch size.

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
