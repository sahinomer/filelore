"""Translation from index query models to storage metadata filters."""

from __future__ import annotations

from filelore.index.models import FileMetadataQuery
from filelore.storage import (
    ConditionOperator,
    MetadataCondition,
    MetadataFilter,
)


def normalize_file_format(value: str) -> str:
    format_key = value.strip().removeprefix(".").casefold()
    return {
        "jpg": "jpeg",
        "jfif": "jpeg",
        "tif": "tiff",
    }.get(format_key, format_key)


def file_type_filter(file_type: str) -> MetadataFilter:
    return MetadataFilter(
        all_of=(
            MetadataCondition(
                "file_type", file_type, operator=ConditionOperator.EQUAL
            ),
        )
    )


def file_metadata_filter(query: FileMetadataQuery) -> MetadataFilter | None:
    """Translate optional CLI and API search fields into a storage filter."""
    conditions: list[MetadataCondition] = []
    if query.name_contains:
        conditions.append(
            MetadataCondition(
                "file_name_search",
                query.name_contains.casefold(),
                operator=ConditionOperator.TEXT_CONTAINS,
            )
        )
    if query.file_format:
        conditions.append(
            MetadataCondition(
                "format_key", normalize_file_format(query.file_format)
            )
        )
    numeric_fields = (
        (
            "metadata.width",
            query.min_width,
            ConditionOperator.GREATER_THAN_OR_EQUAL,
        ),
        (
            "metadata.height",
            query.min_height,
            ConditionOperator.GREATER_THAN_OR_EQUAL,
        ),
        ("metadata.width", query.max_width, ConditionOperator.LESS_THAN_OR_EQUAL),
        (
            "metadata.height",
            query.max_height,
            ConditionOperator.LESS_THAN_OR_EQUAL,
        ),
    )
    for field, value, operator in numeric_fields:
        if value is not None:
            conditions.append(MetadataCondition(field, value, operator=operator))
    if query.modified_after is not None:
        conditions.append(
            MetadataCondition(
                "modified_at",
                query.modified_after,
                operator=ConditionOperator.GREATER_THAN_OR_EQUAL,
            )
        )
    if query.modified_before is not None:
        conditions.append(
            MetadataCondition(
                "modified_at",
                query.modified_before,
                operator=ConditionOperator.LESS_THAN_OR_EQUAL,
            )
        )
    return MetadataFilter(all_of=tuple(conditions)) if conditions else None
