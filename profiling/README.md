# Indexing profiler

This profiler diagnoses where FileLore indexing spends time. It is not a
benchmark and does not define a baseline score.

The implementation lives outside `filelore/`. It invokes the real CLI indexing
pipeline and temporarily wraps selected runtime boundaries for observation.
Discovery, hashing, metadata parsing, decoding, embedding, batching, and
storage behavior are never reimplemented. Target validation and tests fail
when a pipeline refactor invalidates profiler coverage.

## Run

Install the optional model and profiler dependencies:

```sh
uv sync --extra embedding --group profiling
```

Profile one or more datasets:

```sh
uv run --extra embedding --group profiling python -m profiling.index_pipeline \
  --image-directory /path/to/image \
  --audio-directory /path/to/audio
```

Profile a mixed-format document corpus:

```sh
uv run --extra embedding --group profiling python -m profiling.index_pipeline \
  --document-directory /path/to/documents
```

Profile the same workload against a running Qdrant service:

```sh
uv run --extra embedding --group profiling python -m profiling.index_pipeline \
  --image-directory /path/to/image \
  --audio-directory /path/to/audio \
  --qdrant-url http://127.0.0.1:6333
```

The default uses an isolated temporary local Qdrant index, the production
index batch size of 100, a 200 ms resource-sampling interval, and the standard
CLIP, CLAP, and Harrier models. Models must already be downloaded if the
machine is offline.

Use `--index-path` to retain the diagnostic index. For safety, an explicit
index path must be absent or empty; the profiler never clears an existing
index. `--qdrant-url` and `--index-path` are mutually exclusive. A service
profile requires the `files` and `files_segments` collections to be absent or
empty; the profiler checks them but never clears them. Use `--cprofile` to add
a Python call profile.

When Docker publishes Qdrant only on IPv4, use `127.0.0.1` rather than
`localhost`. An IPv6-first `localhost` resolution can add a connection fallback
delay to every Qdrant request.

Stage timings include time waiting for a Qdrant service. Process CPU, RAM, and
I/O samples cover FileLore only, not the separate Qdrant process or container.
System CPU and device-wide NVIDIA GPU samples retain their original scope.

## Metrics

The semantic timeline covers:

- discovery, existing-record lookup, and full-file SHA-256 hashing;
- model loading and cleanup;
- image metadata, decode/convert, model preprocessing, transfers, CUDA
  forward execution, and vector postprocessing;
- audio metadata, segment planning, decode/downmix/resample, CLAP feature
  extraction, transfers, CUDA forward execution, and vector postprocessing;
- document parsing, structure-aware chunking, SentenceTransformer
  preprocessing, CUDA forward execution, and vector postprocessing;
- Qdrant collection setup, segment deletion, record preparation, and upserts;
- per-modality queues and the complete indexing run.

Resource sampling reports process and system CPU, resident memory, process I/O
bytes, NVIDIA GPU utilization, GPU memory, and GPU power when available. GPU
forward stages also use CUDA events for synchronized device time.

Stage durations are hierarchical and inclusive. Parent and child rows must not
be added together.

## Results

Each run writes an ignored directory under `profiling/results/<timestamp>/`:

| File | Contents |
| --- | --- |
| `summary.md` | Human-readable stage and resource tables |
| `summary.json` | Structured environment, configuration, and aggregates |
| `events.csv` | Every nested timing event |
| `resources.csv` | Time-series CPU, memory, disk-I/O, and GPU samples |
| `cprofile.prof` | Optional Python call profile |

Raw output remains ignored because profiles are environment-specific and can
be large. Concise reviewed observations can be committed under
`profiling/reports/`.

## Recorded profiles

These results provide a relative view of the image, audio, and document
indexing pipelines on the test system. Image and audio were profiled together;
documents were profiled in a separate mixed-format run.

### Test system

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 7 5800H |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU (8 GB) |

Shared configuration: Windows 11, Python 3.12.3, PyTorch 2.13.0 with CUDA
13.0, and index batch size 100. The image/audio run used Transformers 4.57.6,
`openai/clip-vit-base-patch32`, and `laion/larger_clap_general`. Its Local Mode
profile used an isolated temporary index; its service profile used Qdrant
1.18.3 in Docker through `http://127.0.0.1:6333`.

The document run used SentenceTransformers 5.7.0,
`microsoft/harrier-oss-v1-270m`, and an isolated temporary Qdrant Local Mode
index. All document formats ran together so model lifetime, batching, storage,
and resource use represent a normal mixed indexing session.

| Dataset | Workload | Items |
| --- | --- | ---: |
| [COCO 2017 validation images](https://cocodataset.org/#download) | Image | 5,000 |
| [Clotho evaluation split](https://zenodo.org/records/3490684) | Audio | 1,045 |
| [RAG-Multi-Corpus](https://github.com/udayallu/RAG-Multi-Corpus) | Documents | 1,180 |

| Workload | End-to-end time | Source throughput | Queue time | Queue throughput | Model inputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image | 85.47 s | 58.50 images/s | 70.59 s | 70.83 images/s | 5,000 images |
| Audio | 167.71 s | 6.23 files/s | 160.93 s | 6.49 files/s | 4,160 segments |
| Documents | 161.17 s | 7.32 files/s | 147.90 s | 7.98 files/s | 11,841 chunks |

Audio throughput is reported per source file; the 1,045 audio files produced
4,160 overlapping segments (3.98 model inputs per file), so chunking contributes
substantially to the lower files/s figure and corresponds to 24.80 segments/s
end-to-end and 25.85 segments/s within the indexing queue.

All 1,180 documents indexed successfully and produced 52,437 structural blocks,
11,841 chunks, and 4,761,905 embedding-input characters. The corpus contained
236 files of each supported format, so no format dominated the mixed result by
file count. Document throughput corresponds to 73.47 chunks/s end-to-end and
80.06 chunks/s within the indexing queue.

### Qdrant mode comparison

This comparison applies to the image/audio workload. Both runs used the same
datasets, models, segmentation settings, and index batch size. Documents were
profiled only in Local Mode because the image/audio comparison already isolates
the storage-mode effect.

| Metric | Local Mode | Qdrant service | Change |
| --- | ---: | ---: | ---: |
| Complete run | 253.19 s | 222.57 s | 12.1% faster |
| Image phase | 85.47 s | 68.45 s | 19.9% faster |
| Audio phase | 167.71 s | 154.13 s | 8.1% faster |
| Storage preparation and write | 52.75 s | 11.40 s | 78.4% faster |
| Actual upserts | 48.69 s | 6.70 s | 86.2% faster |
| Upsert throughput | 209.61 points/s | 1,522.66 points/s | 7.26x higher |

### End-to-end phase steps

Each percentage is relative to that workload's complete end-to-end time. These
rows are non-overlapping and include work outside the indexing queues.

| Pipeline step | Image | Audio | Documents |
| --- | ---: | ---: | ---: |
| Discovery | 0.15 s (0.18%) | 0.03 s (0.02%) | 0.03 s (0.02%) |
| Collection setup and existing-record lookup | 0.72 s (0.84%) | 0.16 s (0.10%) | 0.19 s (0.12%) |
| Full-file hashing | 3.08 s (3.61%) | 1.69 s (1.01%) | 0.37 s (0.23%) |
| Model loading | 10.29 s (12.03%) | 4.23 s (2.52%) | 12.27 s (7.61%) |
| Indexing queue | 70.59 s (82.59%) | 160.93 s (95.95%) | 147.90 s (91.76%) |
| Model cleanup | 0.16 s (0.19%) | 0.20 s (0.12%) | 0.26 s (0.16%) |
| Unassigned phase/control-flow overhead | 0.48 s (0.56%) | 0.47 s (0.28%) | 0.15 s (0.09%) |

### Pipeline order

The profiler exercises the production pipeline in this order. Bracketed work
is repeated for each indexing batch; audio embedding work is additionally
repeated for each planned segment batch.

```text
Images
discover -> collection/lookup -> hash -> load CLIP
  -> [metadata -> decode -> CLIP preprocessing -> host-to-device
      -> GPU forward -> device-to-host -> vector postprocessing
      -> prepare records and write to Qdrant]
  -> model cleanup

Audio
discover -> collections/lookup -> hash -> load CLAP
  -> [metadata -> plan segments
      -> [decode/downmix/resample -> CLAP feature preprocessing
          -> host-to-device -> GPU forward -> device-to-host
          -> vector postprocessing]
      -> prepare file and segment records and write to Qdrant]
  -> model cleanup

Documents
discover -> collections/lookup -> hash -> load Harrier
  -> [parse -> structure-aware chunking
      -> SentenceTransformer preprocessing -> GPU forward
      -> vector postprocessing
      -> prepare file and chunk records and write to Qdrant]
  -> model cleanup
```

### Image queue

These are non-overlapping leaf stages relative to the 70.59-second image
queue. The table is shown in pipeline order and sums to approximately 100%
after rounding.

| Pipeline step | Time | Share of image queue |
| --- | ---: | ---: |
| Metadata | 1.58 s | 2.23% |
| Decode and RGB conversion | 12.25 s | 17.36% |
| CLIP model preprocessing | 13.23 s | 18.74% |
| Host-to-device transfer | 0.54 s | 0.76% |
| GPU forward | 11.02 s | 15.60% |
| Device-to-host transfer | 0.09 s | 0.13% |
| Vector postprocessing | 0.58 s | 0.82% |
| Record preparation and Qdrant write | 27.26 s | 38.61% |
| Unassigned queue/control-flow overhead | 4.05 s | 5.74% |

### Audio queue

The 1,045 audio files produced 4,160 overlapping segments. These are
non-overlapping leaf stages relative to the 160.93-second audio queue and sum
to approximately 100% after rounding.

| Pipeline step | Time | Share of audio queue |
| --- | ---: | ---: |
| Metadata | 0.49 s | 0.30% |
| Segment planning | 0.01 s | 0.01% |
| Decode, downmix, and resample | 18.00 s | 11.18% |
| CLAP feature preprocessing | 85.22 s | 52.95% |
| Host-to-device transfer | 0.27 s | 0.17% |
| GPU forward | 29.92 s | 18.59% |
| Device-to-host transfer | 0.06 s | 0.04% |
| Vector postprocessing | 0.38 s | 0.24% |
| Record preparation and Qdrant write | 25.49 s | 15.84% |
| Unassigned queue/control-flow overhead | 1.10 s | 0.68% |

### Document queue

These are non-overlapping leaf stages relative to the 147.90-second document
queue. Small framework, deletion, serialization, and control-flow costs are
combined under other overhead.

| Pipeline step | Time | Share of document queue |
| --- | ---: | ---: |
| Document parsing | 27.69 s | 18.72% |
| Structure-aware chunking | 0.77 s | 0.52% |
| SentenceTransformer preprocessing | 1.47 s | 1.00% |
| GPU forward | 48.04 s | 32.48% |
| Vector postprocessing | 0.77 s | 0.52% |
| Qdrant Local Mode upserts | 64.45 s | 43.58% |
| Other overhead | 4.71 s | 3.19% |

Local Qdrant upserts were the largest queue cost, followed by Harrier GPU
forward execution and document parsing. Chunking consumed only 0.52% of queue
time. Stage timings are inclusive in the generated summary; this table uses
leaf stages to avoid double counting.

### Document parser comparison

The main document result remains the mixed workload. These parser rows are
derived from per-file events within that run; no format-specific reruns were
needed.

| Format | Files | Input | Parse time | Median/file | p95/file | Blocks | Chunks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DOCX | 236 | 2.790 MiB | 9.325 s | 29.205 ms | 93.613 ms | 11,784 | 2,841 |
| HTML | 236 | 1.363 MiB | 1.814 s | 6.101 ms | 17.305 ms | 11,774 | 2,837 |
| Markdown | 236 | 1.015 MiB | 0.843 s | 2.732 ms | 7.849 ms | 12,182 | 2,841 |
| PDF | 236 | 6.278 MiB | 10.689 s | 32.665 ms | 115.161 ms | 4,913 | 851 |
| PPTX | 236 | 8.612 MiB | 5.013 s | 17.533 ms | 46.102 ms | 11,784 | 2,471 |

PDF and DOCX accounted for 72.29% of parsing time. The formats contain
equivalent corpus material, but their extracted structural boundaries differ;
that is reflected in the chunk totals and should not be interpreted as a
parser-quality score.

### Resource consumption

The image/audio and document profiles are separate runs, so combining their
averages would not describe either workload. The comparison therefore uses
the maximum observed demand from each run. The profiler collected 1,233
image/audio samples and 777 document samples at a target interval of 200 ms.

| Resource | Image/audio peak | Document peak |
| --- | ---: | ---: |
| Process CPU, normalized across 16 logical processors | 63.02% | 9.62% |
| Process RAM | 1,892.04 MiB | 2,028.38 MiB |
| GPU utilization | 100.00% | 100.00% |
| GPU memory used | 3,091 MiB | 2,999 MiB |
| GPU power | 76.78 W | 114.91 W |

GPU metrics are device-wide readings from `nvidia-smi`; GPU memory can
therefore include allocations from the driver and other processes. The raw
document process-CPU peak was 153.90% under psutil's multicore semantics,
equivalent to 9.62% of the test system's total logical-CPU capacity. Its
system-wide CPU peak was 18.20%, and GPU activity was nonzero in 80.93% of
samples.

| Cumulative FileLore process I/O | Image/audio | Documents |
| --- | ---: | ---: |
| Read | 7,059.21 MiB | 246.27 MiB |
| Written | 337.21 MiB | 462.00 MiB |
