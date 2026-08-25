"""Runtime-only instrumentation around the real FileLore indexing pipeline."""

from __future__ import annotations

from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar
from unittest.mock import patch

import filelore.cli as cli_module
import filelore.index.pipeline as pipeline_module
from filelore.audio import AudioChunkVectorizer, SoundFileAudioDecoder
from filelore.documents import DocumentParserRegistry, ParagraphChunker
from filelore.embedding import BaseEmbedding
from filelore.embedding.audio import ClapAudioEmbedding
from filelore.embedding.document import SentenceTransformerTextEmbedding
from filelore.embedding.image import ClipImageEmbedding
from filelore.index import FileIndexRepository, IndexCoordinator
from filelore.metadata import AudioMetadataParser, ImageMetadataParser
from filelore.processors import AudioProcessor, DocumentProcessor, ImageProcessor
from filelore.storage import QdrantVectorDatabase
from profiling.metrics import StageRecorder


EmbeddingType = TypeVar("EmbeddingType", bound=BaseEmbedding[Any])


PIPELINE_TARGETS: tuple[tuple[object, str], ...] = (
    (IndexCoordinator, "discover"),
    (FileIndexRepository, "get_by_paths"),
    (pipeline_module, "calculate_file_hash"),
    (ImageMetadataParser, "parse"),
    (AudioMetadataParser, "parse"),
    (DocumentParserRegistry, "parse"),
    (ParagraphChunker, "chunks"),
    (ImageProcessor, "process_batch"),
    (AudioProcessor, "process_batch"),
    (DocumentProcessor, "process_batch"),
    (AudioChunkVectorizer, "_plan_segments"),
    (SoundFileAudioDecoder, "decode"),
    (FileIndexRepository, "store_prepared_many"),
    (QdrantVectorDatabase, "ensure_collection"),
    (QdrantVectorDatabase, "upsert"),
    (QdrantVectorDatabase, "delete_by_filter"),
    (cli_module, "_index_queue"),
)


def validate_instrumentation_targets() -> None:
    """Fail clearly when a pipeline refactor invalidates profiler coverage."""
    missing = [
        f"{getattr(target, '__name__', type(target).__name__)}.{name}"
        for target, name in PIPELINE_TARGETS
        if not callable(getattr(target, name, None))
    ]
    if missing:
        raise RuntimeError(
            "Profiler targets no longer match the FileLore pipeline: "
            + ", ".join(missing)
        )


class _ProcessorProxy:
    def __init__(
        self,
        processor: Any,
        recorder: StageRecorder,
        stage: str,
        input_name: str,
    ) -> None:
        self._processor = processor
        self._recorder = recorder
        self._stage = stage
        self._input_name = input_name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get(self._input_name, ())
        try:
            items = len(inputs)
        except TypeError:
            items = None
        with self._recorder.span(self._stage, items=items):
            return self._processor(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)


class ExternalInstrumentation:
    """Patch semantic boundaries only for the duration of a profile run."""

    def __init__(self, recorder: StageRecorder) -> None:
        self.recorder = recorder
        self._stack = ExitStack()

    def __enter__(self) -> ExternalInstrumentation:
        validate_instrumentation_targets()
        self._patch_pipeline()
        return self

    def __exit__(self, *exception: object) -> None:
        self._stack.__exit__(*exception)

    def profiled_factory(
        self,
        factory: Callable[[], EmbeddingType],
        modality: str,
    ) -> Callable[[], EmbeddingType]:
        def create() -> EmbeddingType:
            with self.recorder.span(f"{modality}.model_load"):
                embedding = factory()
            self._instrument_embedding(embedding, modality)
            return embedding

        return create

    def _patch_pipeline(self) -> None:
        recorder = self.recorder

        original_discover = IndexCoordinator.discover

        @wraps(original_discover)
        def discover(instance: Any, *args: Any, **kwargs: Any) -> Any:
            with recorder.span("planning.discovery") as measured:
                result = original_discover(instance, *args, **kwargs)
                measured.items = result.total_files
                measured.details["file_counts"] = {
                    queue.file_type: len(queue.paths) for queue in result.queues
                }
                extension_counts: dict[str, int] = {}
                for queue in result.queues:
                    for path in queue.paths:
                        extension = path.suffix.casefold() or "<none>"
                        extension_counts[extension] = (
                            extension_counts.get(extension, 0) + 1
                        )
                measured.details["extension_counts"] = extension_counts
                return result

        self._stack.enter_context(
            patch.object(IndexCoordinator, "discover", discover)
        )

        original_lookup = FileIndexRepository.get_by_paths

        @wraps(original_lookup)
        def get_by_paths(
            instance: Any, paths: Sequence[Any], *args: Any, **kwargs: Any
        ) -> Any:
            with recorder.span("planning.existing_lookup", items=len(paths)):
                return original_lookup(instance, paths, *args, **kwargs)

        self._stack.enter_context(
            patch.object(FileIndexRepository, "get_by_paths", get_by_paths)
        )

        original_hash = pipeline_module.calculate_file_hash

        @wraps(original_hash)
        def calculate_file_hash(path: Any, *args: Any, **kwargs: Any) -> str:
            file_path = Path(path)
            try:
                input_bytes = file_path.stat().st_size
            except OSError:
                input_bytes = None
            with recorder.span(
                "planning.hash",
                items=1,
                input_bytes=input_bytes,
                details={"path": str(file_path)},
            ):
                return original_hash(path, *args, **kwargs)

        self._stack.enter_context(
            patch.object(
                pipeline_module, "calculate_file_hash", calculate_file_hash
            )
        )

        self._patch_metadata(ImageMetadataParser, "image")
        self._patch_metadata(AudioMetadataParser, "audio")
        self._patch_processing(ImageProcessor, "image")
        self._patch_processing(AudioProcessor, "audio")
        self._patch_document_parsing()
        self._patch_document_chunking()
        self._patch_processing(DocumentProcessor, "text")

        original_plan = AudioChunkVectorizer._plan_segments

        @wraps(original_plan)
        def plan_segments(instance: Any, sources: Sequence[Any]) -> Any:
            with recorder.span(
                "audio.segment_planning",
                items=len(sources),
            ) as measured:
                result = original_plan(instance, sources)
                measured.details["segments"] = len(result[0])
                return result

        self._stack.enter_context(
            patch.object(AudioChunkVectorizer, "_plan_segments", plan_segments)
        )

        original_decode = SoundFileAudioDecoder.decode

        @wraps(original_decode)
        def decode(instance: Any, path: Any, audio_range: Any, **kwargs: Any) -> Any:
            details = {
                "path": str(path),
                "start_seconds": audio_range.start_seconds,
                "end_seconds": audio_range.end_seconds,
                "duration_seconds": audio_range.duration_seconds,
            }
            with recorder.span(
                "audio.decode_downmix_resample", items=1, details=details
            ) as measured:
                result = original_decode(
                    instance, path, audio_range, **kwargs
                )
                measured.details["output_samples"] = len(result.samples)
                measured.details["output_bytes"] = len(result.samples) * 4
                return result

        self._stack.enter_context(
            patch.object(SoundFileAudioDecoder, "decode", decode)
        )

        original_store = FileIndexRepository.store_prepared_many

        @wraps(original_store)
        def store_prepared_many(
            instance: Any, prepared_files: Sequence[Any], *args: Any, **kwargs: Any
        ) -> Any:
            segment_count = sum(len(item.segments) for item in prepared_files)
            with recorder.span(
                "storage.prepare_and_write",
                items=len(prepared_files),
                details={"segments": segment_count},
            ):
                return original_store(
                    instance, prepared_files, *args, **kwargs
                )

        self._stack.enter_context(
            patch.object(
                FileIndexRepository,
                "store_prepared_many",
                store_prepared_many,
            )
        )

        self._patch_storage()

        original_queue = cli_module._index_queue

        @wraps(original_queue)
        def index_queue(
            file_indexer: Any, queue: Any, *args: Any, **kwargs: Any
        ) -> Any:
            with recorder.span(
                f"{queue.file_type}.queue",
                items=queue.work_count,
            ):
                return original_queue(file_indexer, queue, *args, **kwargs)

        self._stack.enter_context(
            patch.object(cli_module, "_index_queue", index_queue)
        )

    def _patch_metadata(self, parser_type: type[Any], modality: str) -> None:
        recorder = self.recorder
        original = parser_type.parse

        @wraps(original)
        def parse(instance: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
            with recorder.span(
                f"{modality}.metadata",
                items=1,
                details={"path": str(path)},
            ):
                return original(instance, path, *args, **kwargs)

        self._stack.enter_context(patch.object(parser_type, "parse", parse))

    def _patch_processing(self, processor_type: type[Any], modality: str) -> None:
        recorder = self.recorder
        original = processor_type.process_batch

        @wraps(original)
        def process_batch(
            instance: Any, paths: Sequence[Any], *args: Any, **kwargs: Any
        ) -> Any:
            with recorder.span(
                f"{modality}.processing", items=len(paths)
            ):
                return original(instance, paths, *args, **kwargs)

        self._stack.enter_context(
            patch.object(processor_type, "process_batch", process_batch)
        )

    def _patch_document_parsing(self) -> None:
        recorder = self.recorder
        original = DocumentParserRegistry.parse

        @wraps(original)
        def parse(instance: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
            document_path = Path(path)
            try:
                input_bytes = document_path.stat().st_size
            except OSError:
                input_bytes = None
            with recorder.span(
                "text.parse",
                items=1,
                input_bytes=input_bytes,
                details={
                    "path": str(document_path),
                    "extension": document_path.suffix.casefold(),
                },
            ) as measured:
                result = original(instance, path, *args, **kwargs)
                measured.details["blocks"] = len(result.blocks)
                return result

        self._stack.enter_context(
            patch.object(DocumentParserRegistry, "parse", parse)
        )

    def _patch_document_chunking(self) -> None:
        recorder = self.recorder
        original = ParagraphChunker.chunks

        @wraps(original)
        def chunks(instance: Any, document: Any) -> Any:
            with recorder.span(
                "text.chunking",
                items=len(document.blocks),
                details={"extension": document.metadata.extension},
            ) as measured:
                result = original(instance, document)
                measured.details["chunks"] = len(result)
                measured.details["characters"] = sum(
                    len(chunk.embedding_text) for chunk in result
                )
                return result

        self._stack.enter_context(
            patch.object(ParagraphChunker, "chunks", chunks)
        )

    def _patch_storage(self) -> None:
        recorder = self.recorder
        original_ensure = QdrantVectorDatabase.ensure_collection

        @wraps(original_ensure)
        def ensure_collection(instance: Any, config: Any) -> Any:
            with recorder.span(
                "storage.ensure_collection",
                items=1,
                details={"collection": config.name},
            ):
                return original_ensure(instance, config)

        self._stack.enter_context(
            patch.object(
                QdrantVectorDatabase, "ensure_collection", ensure_collection
            )
        )

        original_upsert = QdrantVectorDatabase.upsert

        @wraps(original_upsert)
        def upsert(
            instance: Any, collection: str, records: Sequence[Any]
        ) -> Any:
            with recorder.span(
                "storage.upsert",
                items=len(records),
                details={"collection": collection},
            ):
                return original_upsert(instance, collection, records)

        self._stack.enter_context(
            patch.object(QdrantVectorDatabase, "upsert", upsert)
        )

        original_delete = QdrantVectorDatabase.delete_by_filter

        @wraps(original_delete)
        def delete_by_filter(
            instance: Any, collection: str, metadata_filter: Any
        ) -> Any:
            with recorder.span(
                "storage.delete_segments",
                details={"collection": collection},
            ):
                return original_delete(instance, collection, metadata_filter)

        self._stack.enter_context(
            patch.object(
                QdrantVectorDatabase, "delete_by_filter", delete_by_filter
            )
        )

    def _instrument_embedding(
        self, embedding: EmbeddingType, modality: str
    ) -> None:
        if isinstance(embedding, ClipImageEmbedding):
            self._require_embedding_methods(
                embedding,
                (
                    "_prepare_image",
                    "_move_inputs",
                    "_image_features",
                    "_feature_rows",
                    "_prepare_vectors",
                ),
            )
            embedding._processor = _ProcessorProxy(
                embedding._processor,
                self.recorder,
                "image.model_preprocessing",
                "images",
            )
            self._wrap_instance_method(
                embedding, "_prepare_image", "image.decode_convert", path_arg=0
            )
            self._wrap_instance_method(
                embedding, "_move_inputs", "image.host_to_device"
            )
            self._wrap_gpu_method(
                embedding, "_image_features", "image.gpu_forward"
            )
            self._wrap_instance_method(
                embedding, "_feature_rows", "image.device_to_host"
            )
            self._wrap_instance_method(
                embedding, "_prepare_vectors", "image.vector_postprocessing"
            )
        elif isinstance(embedding, ClapAudioEmbedding):
            self._require_embedding_methods(
                embedding,
                (
                    "_move_inputs",
                    "_audio_features",
                    "_feature_rows",
                    "_prepare_vectors",
                ),
            )
            embedding._processor = _ProcessorProxy(
                embedding._processor,
                self.recorder,
                "audio.model_preprocessing",
                "audio",
            )
            self._wrap_instance_method(
                embedding, "_move_inputs", "audio.host_to_device"
            )
            self._wrap_gpu_method(
                embedding, "_audio_features", "audio.gpu_forward"
            )
            self._wrap_instance_method(
                embedding, "_feature_rows", "audio.device_to_host"
            )
            self._wrap_instance_method(
                embedding, "_prepare_vectors", "audio.vector_postprocessing"
            )
        elif isinstance(embedding, SentenceTransformerTextEmbedding):
            self._require_embedding_methods(
                embedding,
                ("_prepare_vectors",),
            )
            self._require_embedding_methods(
                embedding._model,
                ("encode", "forward"),
            )
            preprocessing_method = (
                "preprocess"
                if callable(getattr(embedding._model, "preprocess", None))
                else "tokenize"
            )
            self._require_embedding_methods(
                embedding._model,
                (preprocessing_method,),
            )
            self._wrap_instance_method(
                embedding._model,
                "encode",
                "text.model_encode",
                items_arg=0,
            )
            self._wrap_instance_method(
                embedding._model,
                preprocessing_method,
                "text.model_preprocessing",
                items_arg=0,
            )
            self._wrap_gpu_method(
                embedding._model,
                "forward",
                "text.gpu_forward",
                torch_module=embedding._torch,
                device=embedding.device,
            )
            self._wrap_instance_method(
                embedding,
                "_prepare_vectors",
                "text.vector_postprocessing",
            )

        self._wrap_instance_method(
            embedding,
            "predict_batch",
            f"{modality}.embedding",
            items_arg=0,
        )
        self._wrap_instance_method(
            embedding, "close", f"{modality}.model_cleanup"
        )

    @staticmethod
    def _require_embedding_methods(
        embedding: Any, names: Sequence[str]
    ) -> None:
        missing = [
            name
            for name in names
            if not callable(getattr(embedding, name, None))
        ]
        if missing:
            raise RuntimeError(
                f"Profiler no longer matches {type(embedding).__name__}: "
                + ", ".join(missing)
            )

    def _wrap_instance_method(
        self,
        instance: Any,
        name: str,
        stage: str,
        *,
        items_arg: int | None = None,
        path_arg: int | None = None,
    ) -> None:
        original = getattr(instance, name)
        recorder = self.recorder

        @wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            items: int | None = None
            if items_arg is not None and len(args) > items_arg:
                items = self._infer_items(args[items_arg])
            elif "expected_count" in kwargs:
                items = int(kwargs["expected_count"])
            input_bytes: int | None = None
            details: dict[str, Any] = {}
            if path_arg is not None and len(args) > path_arg:
                candidate = args[path_arg]
                items = 1
                if isinstance(candidate, (str, Path)):
                    path = Path(candidate)
                    details["path"] = str(path)
                    try:
                        input_bytes = path.stat().st_size
                    except OSError:
                        pass
            elif items is None and args:
                items = self._infer_items(args[0])
            with recorder.span(
                stage,
                items=items,
                input_bytes=input_bytes,
                details=details,
            ):
                return original(*args, **kwargs)

        setattr(instance, name, measured)

    def _wrap_gpu_method(
        self,
        instance: Any,
        name: str,
        stage: str,
        *,
        torch_module: Any | None = None,
        device: str | None = None,
    ) -> None:
        original = getattr(instance, name)
        recorder = self.recorder

        @wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            active_torch = torch_module or instance._torch
            active_device = device or instance.device
            using_cuda = str(active_device).startswith("cuda")
            items = self._infer_items(args[0]) if args else None
            if using_cuda:
                active_torch.cuda.synchronize()
                started = active_torch.cuda.Event(enable_timing=True)
                finished = active_torch.cuda.Event(enable_timing=True)
                started.record()
            with recorder.span(stage, items=items) as span:
                result = original(*args, **kwargs)
                if using_cuda:
                    finished.record()
                    active_torch.cuda.synchronize()
                    span.cuda_ms = float(started.elapsed_time(finished))
                return result

        setattr(instance, name, measured)

    @staticmethod
    def _infer_items(value: Any) -> int | None:
        if isinstance(value, dict):
            for nested in value.values():
                inferred = ExternalInstrumentation._infer_items(nested)
                if inferred is not None:
                    return inferred
            return None
        if value is None or isinstance(value, (str, bytes, Path)):
            return None
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
        rows = getattr(value, "rows", None)
        if rows is not None:
            return len(rows)
        try:
            return len(value)
        except TypeError:
            return None
