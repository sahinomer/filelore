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

Profile either or both datasets:

```sh
uv run --extra embedding --group profiling python -m profiling.index_pipeline \
  --image-directory /path/to/image \
  --audio-directory /path/to/audio
```

The default uses an isolated temporary local Qdrant index, the production
index batch size of 100, a 200 ms resource-sampling interval, and the standard
CLIP and CLAP models. Models must already be downloaded if the machine is
offline.

Use `--index-path` to retain the diagnostic index. For safety, an explicit
index path must be absent or empty; the profiler never clears an existing
index. Use `--cprofile` to add a Python call profile.

## Metrics

The semantic timeline covers:

- discovery, existing-record lookup, and full-file SHA-256 hashing;
- model loading and cleanup;
- image metadata, decode/convert, model preprocessing, transfers, CUDA
  forward execution, and vector postprocessing;
- audio metadata, segment planning, decode/downmix/resample, CLAP feature
  extraction, transfers, CUDA forward execution, and vector postprocessing;
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

## Recorded full-pipeline profile

These results provide a relative view of the indexing pipeline on the test
system.

### Test system

| Component | Value |
| --- | --- |
| CPU | AMD Ryzen 7 5800H |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU (8 GB) |

Configuration: Windows 11, Python 3.12.3, PyTorch 2.13.0 with CUDA
13.0, Transformers 4.57.6, an isolated temporary Qdrant local-mode index,
index batch size 100, `openai/clip-vit-base-patch32`, and
`laion/larger_clap_general`.

| Dataset | Workload | Items |
| --- | --- | ---: |
| [COCO 2017 validation images](https://cocodataset.org/#download) | Image | 5,000 |
| [Clotho evaluation split](https://zenodo.org/records/3490684) | Audio | 1,045 |

Both modality runs completed successfully with no recorded stage errors. They
ran sequentially and indexed 6,045 source files in 253.19 seconds.

| Phase | End-to-end time | Share of run | End-to-end throughput | Queue time | Queue throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Image | 85.47 s | 33.76% | 58.50 images/s | 70.59 s | 70.83 images/s |
| Audio | 167.71 s | 66.24% | 6.23 files/s | 160.93 s | 6.49 files/s |
| Complete run | 253.19 s | 100.00% | - | - | - |

### End-to-end phase steps

Each percentage is relative to its modality's complete end-to-end time. These
rows are non-overlapping and include work outside the indexing queues.

| Pipeline step | Image time | Image share | Audio time | Audio share |
| --- | ---: | ---: | ---: | ---: |
| Discovery | 0.15 s | 0.18% | 0.03 s | 0.02% |
| Collection setup and existing-record lookup | 0.72 s | 0.84% | 0.16 s | 0.10% |
| Full-file hashing | 3.08 s | 3.61% | 1.69 s | 1.01% |
| Model loading | 10.29 s | 12.03% | 4.23 s | 2.52% |
| Indexing queue | 70.59 s | 82.59% | 160.93 s | 95.95% |
| Model cleanup | 0.16 s | 0.19% | 0.20 s | 0.12% |
| Unassigned phase/control-flow overhead | 0.48 s | 0.56% | 0.47 s | 0.28% |

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

### Resource consumption

The profiler collected 1,233 samples at a target interval of 200 ms. Process
CPU is normalized across the test system's 16 logical processors. GPU metrics
are device-wide readings from `nvidia-smi`; GPU memory can therefore include
allocations from the driver and other processes.

| Resource | Average | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Process CPU | 9.82% | 35.67% | 63.02% |
| Process RAM | 1,427.27 MiB | 1,720.02 MiB | 1,892.04 MiB |
| GPU utilization | 16.25% | 100.00% | 100.00% |
| GPU memory used | 2,526.29 MiB | 3,089 MiB | 3,091 MiB |
| GPU power | 49.05 W | 70.43 W | 76.78 W |

The cumulative FileLore process I/O counters recorded 7,059.21 MiB read and
337.21 MiB written during the run.
