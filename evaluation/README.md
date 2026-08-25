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

Evaluate text-to-document retrieval with the RAG-Multi-Corpus supporting-facts
CSV after its documents have been indexed:

```sh
uv run --extra embedding python -m evaluation.retrieval \
  /path/to/queries_01122025.csv \
  --target document
```

The evaluator checks the parent records in the Qdrant `files` collection before
loading a model. Missing or ambiguous filenames are excluded and reported. Each
media caption is one query with one relevant file. Audio results are ranked by
the highest-scoring segment per parent file.

For documents, rows with the same normalized enterprise and query are combined,
and all cited supporting files become the query's relevant set. Equivalent
DOCX, HTML, Markdown, PDF, and PPTX files share one logical document identity.
The evaluator reports a mixed-format view and one view per format. A query with
no indexed relevant document is skipped; a multi-document query with at least
one indexed citation remains eligible and is reported as partially covered.
The default six document views run concurrently, and query embeddings are
calculated once and reused by every view. Use `--document-workers` to change the
view concurrency.

The default cutoffs are 1, 5, and 10; use `--k` to change them.

Each run writes an ignored JSON summary to
`evaluation/results/<timestamp>-<target>.json`. Use `--output
/path/to/result.json` to select another destination. The evaluator accepts the
same `--index-path`, `--qdrant-url`, and `FILELORE_QDRANT_URL` storage options as
the main CLI.

## Recorded retrieval results

This is a relative, end-to-end evaluation of FileLore's parsing, indexing, and
retrieval flow. Published results validate the general range rather than exact
scores, which remain dependent on the selected model and evaluation protocol.

All COCO and Clotho annotated files were present uniquely in the index. The
document run skipped five queries whose only citation was the one unavailable
logical document described below. JSON result files retain every calculated
metric and per-query outcome.

### Setup

| Component | Image evaluation | Audio evaluation | Document evaluation |
| --- | --- | --- | --- |
| Dataset | COCO 2017 validation | Clotho evaluation | RAG-Multi-Corpus |
| Indexed files | 5,000 | 1,045 | 1,180 physical; 236 logical |
| Evaluated queries | 25,014 captions | 5,225 captions | 1,066 consolidated queries |
| Model | `openai/clip-vit-base-patch32` | `laion/larger_clap_general` | `microsoft/harrier-oss-v1-270m` |
| Vector | `image_clip_openai_vit_b32` | `audio_clap_laion_larger_general` | `text_harrier_microsoft_oss_v1_270m` |
| Raw candidate limit | 10 | 100 | 200 |

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

### RAG-Multi-Corpus text-to-document retrieval

The index contained 1,180 physical files: 236 logical documents represented in
all five formats. This file count is separate from the query count. The CSV's
1,252 source rows consolidated to 1,071 unique enterprise/query pairs because
some queries appeared in multiple rows with additional supporting files. Five
of those 1,071 queries were skipped because their only cited logical document,
`ZX Bank/Account Close Guide.md`, was unavailable, leaving 1,066 evaluated
queries. All evaluated queries were fully covered, with no partial, ambiguous,
format-missing, or candidate-underfilled cases.

The mixed view represents the deployed index: equivalent format variants can
occupy separate physical ranks, while each relevant logical document receives
credit only once.

| Mixed-view metric | @1 | @5 | @10 |
| --- | ---: | ---: | ---: |
| Hit rate | 76.27% | 87.80% | 93.06% |
| Recall | 76.27% | 87.76% | 92.95% |
| MRR | 76.27% | 80.52% | 81.25% |
| nDCG | 76.27% | 82.31% | 84.02% |
| MAP | 76.27% | 80.50% | 81.22% |
| Complete-relevance rate | 76.27% | 87.71% | 92.87% |

The per-format views isolate parsing and chunking differences without duplicate
format variants competing for ranks.

| Retrieval view | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed | 76.27% | 87.76% | 92.95% | 81.25% | 84.02% |
| DOCX | 78.66% | 94.67% | 97.19% | 85.52% | 88.38% |
| HTML | 79.03% | 94.67% | 97.19% | 85.74% | 88.54% |
| Markdown | 78.85% | 94.70% | 97.47% | 85.71% | 88.60% |
| PDF | 78.38% | 95.61% | 98.22% | 85.84% | 88.90% |
| PPTX | 70.40% | 88.20% | 93.01% | 78.25% | 81.82% |

PDF had the strongest Recall@10 at 98.22%, while PPTX was the weakest format at
93.01%. Mixed Recall@10 was lower because its ranking contains all five
equivalent physical variants. This is expected mixed-index behavior rather than
an average parser score.

| Mixed-view query type | Queries | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Analytical | 157 | 94.90% | 83.84% | 86.52% |
| Boolean | 154 | 92.86% | 75.93% | 79.91% |
| Comparative | 185 | 92.97% | 82.31% | 84.88% |
| Descriptive | 189 | 93.03% | 85.00% | 86.80% |
| Open-Ended | 100 | 90.00% | 79.23% | 81.81% |
| Procedural | 246 | 93.90% | 81.11% | 84.13% |
| Temporal | 31 | 83.87% | 73.06% | 75.70% |
| Other | 4 | 100.00% | 81.25% | 85.77% |
| Overall micro average | 1,066 | 92.95% | 81.25% | 84.02% |
| Official-type macro average | 7 types | 91.65% | 80.07% | 82.82% |

The macro average gives each of the seven official query types equal weight and
excludes `Other`. Only three eligible queries cited multiple logical documents,
so their separate metrics are retained in the JSON but are too small to support
a useful conclusion here.

#### Published validation reference

The [dataset repository](https://github.com/udayallu/RAG-Multi-Corpus) proposes
retrieval and cross-format benchmark tasks but does not publish a fixed score.
The closest dataset-author reference is the
[W-RAC paper](https://arxiv.org/abs/2604.04936), which reports the following
aggregate retrieval results on the original 786-query release:

| Published method | Recall@3 | Recall@6 | MRR | nDCG@3 | nDCG@6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Agentic chunking baseline | 88% | 93% | 87% | 88% | 89% |
| W-RAC | 84% | 91% | 83% | 83% | 85% |

The paper evaluates retrieved chunks from 786 queries at cutoffs 3 and 6.
FileLore evaluates logical source-document recall from a larger local CSV with
1,071 consolidated queries, of which 1,066 were eligible, at cutoffs 1, 5, and
10. FileLore also reports five isolated format views plus a mixed view in which
duplicate physical variants compete for rank.

## Execution times

All timings below used Qdrant Python Local Mode; a Qdrant service was not
measured. The image and audio metric passes embed captions in batches of 32 and
retrieve their vectors sequentially. The document pass embeds each query once
and retrieves the mixed and five per-format views concurrently with six workers.
The full-pass average is the total metric-pass wall time divided by its unique
query count and is not single-query latency.

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 7 5800H |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU (8 GB) |

| Target | Model load (excluded) | Metric pass | Embedding | Retrieval | Throughput | Full-pass average |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image | 25.303 s | 360.512 s | 59.636 s | 300.053 s | 69.38 queries/s | 14.41 ms/query |
| Audio | 23.185 s | 74.455 s | 13.084 s | 61.204 s | 70.18 queries/s | 14.25 ms/query |
| Document, six views | 12.449 s | 1,269.234 s | 2.055 s | 1,267.161 s | 0.84 full queries/s | 1,190.65 ms/full query |

The document pass completed 6,396 query-view evaluations at 5.04 view
evaluations/s. Its retrieval value is parallel wall time for all six views, not
the sum of their individual durations.

Single-query latency is measured separately on 1,000 deterministically selected
queries after 10 unmeasured warm-up searches. Queries run one at a time, starting
immediately before text embedding and ending when the vector-database response
completes. Parsing, index scanning, model loading, warm-up, deduplication, and
metric calculation are excluded.

| Target | Average | P95 | P99 |
| --- | ---: | ---: | ---: |
| Image | 23.67 ms | 29.16 ms | 30.84 ms |
| Audio | 24.97 ms | 31.91 ms | 33.86 ms |
| Document, mixed view | 107.98 ms | 135.65 ms | 153.31 ms |
