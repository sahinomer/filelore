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

The current implementation starts with image indexing and text-to-image
semantic search.

## Setup

FileLore requires Python 3.12 or later and
[uv](https://docs.astral.sh/uv/).

Clone the repository, enter its directory, and synchronize the environment:

```sh
uv sync --extra embedding
```

The embedding model is downloaded when it is used for the first time.

## Tests

Run the test suite with the test dependency group:

```sh
uv run --group test python -m pytest
```

## Index

Index the supported files under a directory:

```sh
uv run python -m filelore --index /path/to/files
```

FileLore scans directories recursively and stores its local index under
`~/.filelore/qdrant` by default. Run `uv run python -m filelore --help` to see
storage and indexing options.

## Search

Search the index with a natural-language description:

```sh
uv run python -m filelore "a red sports car"
```

Metadata filters can narrow the results:

```sh
uv run python -m filelore "a mountain landscape" \
  --format jpeg \
  --min-resolution 1920x1080 \
  --limit 20
```

## Areas of exploration

- Text-document indexing and content search.
- Audio indexing and semantic retrieval.
- Speech transcription and search.

## License

FileLore is available under the [MIT License](LICENSE).
