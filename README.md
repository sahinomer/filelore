# FileLore

FileLore is a local-first indexing and semantic-search project. It provides a
small foundation for building searchable file indexes while keeping storage
and retrieval under the user's control.

## Capabilities

- Discover and index files from local directories.
- Extract structured file and media metadata.
- Create embeddings for semantic retrieval.
- Combine semantic search with metadata filters.
- Store indexes locally or in a configured Qdrant service.

The current implementation supports image, audio, and text-document indexing.
The CLI and TUI provide text-to-image, image-to-image, text-to-audio,
audio-to-audio, and text-to-document search. Audio and document content is
stored as searchable child chunks: the CLI shows raw chunk matches, while the
TUI groups matching chunks under their source files.

Text documents currently include PDF, HTML, DOCX, PPTX, and Markdown files.
Extraction preserves available structural context such as pages, slides,
headings, sections, and source lines. Text handling is Unicode-safe and supports
multilingual documents and queries.

## Setup

FileLore requires Python 3.12 or later and
[uv](https://docs.astral.sh/uv/).

Clone the repository, enter its directory, and synchronize the environment:

```sh
uv sync --extra embedding
```

Each embedding model is downloaded when it is used for the first time.

## Tests

Run the test suite with the test dependency group:

```sh
uv run --group test python -m pytest
```

## Benchmarks

See the [benchmark guide](benchmarks/README.md) for repeatable CLIP and CLAP
embedding and Qdrant storage workloads.

## Retrieval evaluation

See the [retrieval evaluation guide](evaluation/README.md) for COCO and Clotho
MRR, Recall, NDCG, and query-latency measurements against an existing index.

## Profiling

See the [indexing profiler guide](profiling/README.md) for external stage,
resource, disk-I/O, and GPU observation of the real indexing pipeline.

## Index

Index the supported files under a directory:

```sh
uv run python -m filelore --index /path/to/files
```

FileLore scans directories recursively and stores its local index under
`~/.filelore/qdrant` by default. Discovery groups images, audio, and text
documents into separate queues. It hashes discovered files before loading a
model, skips unchanged content, and processes only new or changed files. Each
non-empty queue is confirmed separately, and only one required embedding model
is loaded at a time. Use `-y` / `--yes` to accept every queue without
prompting:

```sh
uv run python -m filelore --index /path/to/files --yes
```

Limit indexing to a specific type when desired:

```sh
uv run python -m filelore --index /path/to/files --index-type audio
```

Index only supported text documents:

```sh
uv run python -m filelore --index /path/to/documents --index-type text
```

Repeat `--index-type` to select multiple types. When it is omitted, all
recognized types are indexed. Run `uv run python -m filelore --help` to see
storage and indexing options.

## Search

Open the full-screen search interface by running FileLore without arguments:

```sh
uv run python -m filelore
```

`-i` / `--interactive` is the explicit equivalent. Select Image, Audio, or
Text beside the search field. FileLore loads that target's model only when a
search is submitted, retains it for later searches, and releases it before
switching targets. Supplying `--target image`, `--target audio`, or
`--target text` when the TUI is launched constrains the selector to that
target.

Find media similar to a reference image or audio file using any of these
methods:

- Type or paste a file path. Relative paths start from the terminal's working
  directory.
- Start typing a path, then press Tab or Right to accept a suggestion.
- Select Browse or press Ctrl+O to choose a file with the keyboard or mouse.

Autocomplete and Browse follow the selected Image or Audio target. Text search
uses a natural-language query rather than a reference document. Once an image
or audio reference is chosen, it appears as an attachment and the same field
accepts optional result filters:

```text
samples/rain.wav format:wav longer-than:1 shorter-than:30
```

Interactive queries may combine semantic text with the metadata currently
stored in the index. Image queries support:

```text
cat name:holiday format:jpg min-res:1280x720 after:2025 before:2026
```

Audio queries support `sample-rate`, `bitrate`, `longer-than`, and
`shorter-than` instead of the resolution filters:

```text
glass breaking format:wav sample-rate:48000 longer-than:1 shorter-than:30
```

Text-document queries search the extracted document chunks. They can use any
supported document format as a filter:

```text
regional rail connections format:pdf after:2025
```

All targets support `name`, `format`, `after`, and `before`. Press F1 for help
tailored to the currently selected target. Date boundaries accept
`YYYY`, `YYYY-MM`, `YYYY-MM-DD`, or a full ISO datetime. `after` is inclusive
and `before` is exclusive, so
`after:2025 before:2026` selects files last modified during 2025.

The default result limit is 20. Audio and document chunk matches are grouped by
parent file in the TUI, and each file can be expanded to show its matching
chunks ordered by similarity. Document results include a text preview and any
available page, slide, heading, section, or source-line context.

The existing one-shot interface remains available.

Find images similar to a reference file. The image target is inferred from the
query file extension:

```sh
uv run python -m filelore --query-file /path/to/reference.jpg
```

Metadata flags still filter the indexed results rather than the query file:

```sh
uv run python -m filelore --query-file /path/to/reference.png \
  --format jpeg --min-resolution 1920x1080
```

Audio file queries are divided into the same overlapping model-sized chunks
used during indexing. Each query chunk searches the indexed audio segments,
and duplicate segment matches retain their best score:

```sh
uv run python -m filelore --query-file /path/to/reference.wav \
  --format wav --longer-than 1 --shorter-than 30
```

Search the index with a natural-language description:

```sh
uv run python -m filelore "a red sports car" --target image
```

One-shot search requires `--target image`, `--target audio`, or `--target text`
so FileLore loads only the required model. A recognized `--format` can imply
the target, so the image target is inferred in this example:

```sh
uv run python -m filelore "a mountain landscape" \
  --format jpeg \
  --min-resolution 1920x1080 \
  --limit 20
```

Audio search currently returns raw chunk matches with their timestamps. Audio
metadata filters can narrow those results:

```sh
uv run python -m filelore "glass breaking" \
  --target audio \
  --sample-rate 48000 \
  --bitrate 192000 \
  --longer-than 1 \
  --shorter-than 30
```

Search text documents by their extracted content:

```sh
uv run python -m filelore "regional rail connections" --target text
```

A supported result format can infer the text target:

```sh
uv run python -m filelore "quarterly sustainability goals" --format pdf
```

Text indexing uses `microsoft/harrier-oss-v1-270m` by default through
SentenceTransformers. Document embedding model details are isolated from the
parser, chunker, and index, so another compatible SentenceTransformer model can
be configured in code without changing the rest of the text pipeline.

## Areas of exploration

- Semantic and experimental document chunking strategies.
- Text retrieval benchmarks, profiling, and evaluation.
- Optional grouping or deduplication for raw CLI audio results.
- Speech transcription and search.
- PDF OCR, reranking, and hybrid lexical/vector retrieval.

## License

FileLore is available under the [MIT License](LICENSE).
