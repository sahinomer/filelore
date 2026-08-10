# Retrieval evaluation

Evaluate an existing FileLore index with COCO image captions:

```sh
uv run --extra embedding python -m evaluation.retrieval \
  /path/to/coco_captions_val2017.json \
  --target image
```

Evaluate it with Clotho audio captions:

```sh
uv run --extra embedding python -m evaluation.retrieval \
  /path/to/clotho_captions_evaluation.csv \
  --target audio
```

The evaluator checks the parent records in the Qdrant `files` collection before
loading a model. Missing or ambiguous filenames are excluded and reported. Each
caption is one query with one relevant file. The default cutoffs are 1, 5, and
10; use `--k` to change them. Audio results are ranked by the highest-scoring
segment per parent file.

Each run writes an ignored JSON summary to
`evaluation/results/<timestamp>-<target>.json`. Use `--output
/path/to/result.json` to select another destination. The evaluator accepts the
same `--index-path`, `--qdrant-url`, and `FILELORE_QDRANT_URL` storage options as
the main CLI.

## Recorded retrieval results

The evaluator is intended for reproducible relative comparisons when changing
embedding models while keeping the dataset, indexing, and retrieval settings
fixed. The results below are an end-to-end sanity check of the evaluation
workflow. Published results are included only as approximate reference points,
not as controlled model benchmarks.

All annotated files were present uniquely in the index. JSON result files retain
all calculated metrics; the tables below focus on Recall and MRR.

### Setup

| Component | Image evaluation | Audio evaluation |
| --- | --- | --- |
| Dataset | COCO 2017 validation | Clotho evaluation |
| Indexed files | 5,000 | 1,045 |
| Caption queries | 25,014 | 5,225 |
| Model | `openai/clip-vit-base-patch32` | `laion/larger_clap_general` |
| Vector | `image_clip_openai_vit_b32` | `audio_clap_laion_larger_general` |
| Raw candidate limit | 10 | 100 |

### COCO text-to-image retrieval

| Metric | @1 | @5 | @10 |
| --- | ---: | ---: | ---: |
| Recall | 30.45% | 54.82% | 66.24% |
| MRR | 30.45% | 39.39% | 40.92% |

| ViT-B/32 reference | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| FileLore, all 25,014 captions | 30.45% | 54.82% | 66.24% |
| Community reproduction, first caption per image | 28.42% | 53.10% | 64.16% |

Baseline reference: [OpenAI CLIP issue #115](https://github.com/openai/CLIP/issues/115).
The caption protocol differs, so these values provide context rather than a
direct comparison.

### Clotho text-to-audio retrieval

| Metric | @1 | @5 | @10 |
| --- | ---: | ---: | ---: |
| Recall | 15.54% | 39.02% | 52.63% |
| MRR | 15.54% | 23.85% | 25.67% |

| LAION-CLAP reference | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| FileLore, highest-scoring segment per file | 15.54% | 39.02% | 52.63% |
| Best non-fusion row in Table 3 (`CLAP-HTSAT`) | 16.70% | 41.10% | 54.10% |

Baseline reference: [LAION-CLAP paper, Table 3](https://arxiv.org/pdf/2211.06687).
FileLore indexes audio as 10-second chunks, so the published non-fusion result
is used as an approximate reference. The table reports `CLAP-HTSAT` trained on
AudioCaps + Clotho + WT5K, while this evaluation uses LAION's improved
[`larger_clap_general`](https://huggingface.co/laion/larger_clap_general)
checkpoint for general audio, music, and speech.

## Execution times

All timings below used Qdrant Python Local Mode; a Qdrant service was not
measured. The full metric pass embeds captions in batches of 32 and retrieves
their vectors sequentially. The full-pass average is the total metric-pass time
divided by its query count and is not single-query latency.

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 7 5800H |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU (8 GB) |

| Target | Model load (excluded) | Metric pass | Embedding | Retrieval | Throughput | Full-pass average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image | 25.303 s | 360.512 s | 59.636 s | 300.053 s | 69.38 queries/s | 14.41 ms/query |
| Audio | 23.185 s | 74.455 s | 13.084 s | 61.204 s | 70.18 queries/s | 14.25 ms/query |

Single-query latency is measured separately on 1,000 deterministically selected
queries after 10 unmeasured warm-up searches. Queries run one at a time, starting
immediately before text embedding and ending when the vector-database response
completes. Parsing, index scanning, model loading, warm-up, deduplication, and
metric calculation are excluded.

| Target | Average | P95 | P99 |
| --- | ---: | ---: | ---: |
| Image | 23.67 ms | 29.16 ms | 30.84 ms |
| Audio | 24.97 ms | 31.91 ms | 33.86 ms |
