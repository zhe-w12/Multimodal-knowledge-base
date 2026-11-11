"""Copy a legacy Milvus collection into the workspace-filtered schema.

The source collection is read-only throughout this migration. Existing source
vectors are retained so the application can be rolled back if needed.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Mapping

from manufacturing_rag.adapters.embedding import HashEmbeddingAdapter
from manufacturing_rag.adapters.vector_store import MilvusVectorStore, parse_metadata, vector_index_text
from manufacturing_rag.config import DEFAULT_CONFIG
from manufacturing_rag.schemas import DEFAULT_WORKSPACE_ID, DocumentChunk


def build_workspace_row(
    row: Mapping[str, Any],
    *,
    workspace_id: str,
    has_sparse_vector: bool,
    sparse_vector_builder: Callable[[DocumentChunk], dict[int, float]],
) -> dict[str, Any]:
    """Add an explicit workspace boundary while preserving the stored vectors."""

    try:
        metadata_value = json.loads(str(row.get("metadata_json") or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Milvus row {row.get('id')} has invalid metadata_json") from exc
    if not isinstance(metadata_value, dict):
        raise ValueError(f"Milvus row {row.get('id')} metadata_json is not an object")

    existing_workspace_id = str(metadata_value.get("workspace_id") or "").strip()
    if existing_workspace_id and existing_workspace_id != workspace_id:
        raise ValueError(
            f"Milvus row {row.get('id')} belongs to {existing_workspace_id}, not {workspace_id}"
        )
    metadata_value["workspace_id"] = workspace_id

    chunk = DocumentChunk.from_mapping(
        {
            "id": str(row.get("id") or ""),
            "text": str(row.get("text") or ""),
            "source": str(row.get("source") or "unknown"),
            "page": row.get("page"),
            "block_id": row.get("block_id"),
            "modality": row.get("modality") or "ocr_text",
            "metadata": metadata_value,
            "image_path": row.get("image_path"),
        }
    )

    sparse_vector = row.get("sparse_vector") if has_sparse_vector else None
    if not isinstance(sparse_vector, dict) or not sparse_vector:
        sparse_vector = sparse_vector_builder(chunk)

    vector = row.get("vector")
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"Milvus row {row.get('id')} has no dense vector")

    try:
        page = int(row.get("page", -1))
    except (TypeError, ValueError):
        page = -1

    return {
        "id": chunk.id[:256],
        "text": chunk.text[:8192],
        "source": chunk.source[:1024],
        "workspace_id": workspace_id,
        "page": page,
        "block_id": str(row.get("block_id") or "")[:256],
        "modality": str(row.get("modality") or "ocr_text")[:64],
        "metadata_json": json.dumps(metadata_value, ensure_ascii=False, separators=(",", ":"))[:8192],
        "image_path": str(row.get("image_path") or "")[:1024],
        "vector": vector,
        "sparse_vector": sparse_vector,
    }


def migrate_collection(
    *,
    target_name: str,
    workspace_id: str,
    expected_rows: int,
) -> dict[str, Any]:
    source_name = DEFAULT_CONFIG.zilliz_collection_name
    if not target_name or len(target_name) > 255:
        raise ValueError("target collection name must contain 1–255 characters")
    if target_name == source_name:
        raise ValueError("target collection must differ from the source collection")
    if not workspace_id or len(workspace_id) > 128:
        raise ValueError("workspace ID must contain 1–128 characters")
    if expected_rows < 0 or expected_rows > 16_384:
        raise ValueError("expected row count must be between 0 and 16384")

    store = MilvusVectorStore(config=DEFAULT_CONFIG, embedding=HashEmbeddingAdapter(DEFAULT_CONFIG))
    try:
        utility, Collection = store._connect_and_import()
        if not utility.has_collection(source_name, using=DEFAULT_CONFIG.zilliz_alias):
            raise RuntimeError(f"source collection does not exist: {source_name}")

        source = Collection(source_name, using=DEFAULT_CONFIG.zilliz_alias)
        source_fields = {field.name for field in source.schema.fields}
        if "workspace_id" in source_fields:
            raise RuntimeError("source collection already has workspace_id; use a per-workspace migration")
        if "vector" not in source_fields:
            raise RuntimeError("source collection has no dense vector field")
        if int(source.num_entities) != expected_rows:
            raise RuntimeError(
                f"source row count changed: expected {expected_rows}, found {source.num_entities}"
            )

        vector_schema = next(field for field in source.schema.fields if field.name == "vector")
        source_dimensions = int(vector_schema.params.get("dim", 0))
        if source_dimensions != DEFAULT_CONFIG.embedding_dimensions:
            raise RuntimeError(
                f"source vector dimension {source_dimensions} does not match configured dimension "
                f"{DEFAULT_CONFIG.embedding_dimensions}"
            )

        output_fields = [
            field
            for field in (
                "id",
                "text",
                "source",
                "page",
                "block_id",
                "modality",
                "metadata_json",
                "image_path",
                "vector",
                "sparse_vector",
            )
            if field in source_fields
        ]
        source_rows = (
            source.query(
                expr='id != ""',
                output_fields=output_fields,
                limit=max(expected_rows, 1),
                consistency_level="Strong",
            )
            if expected_rows
            else []
        )
        if len(source_rows) != expected_rows:
            raise RuntimeError(
                f"source query returned {len(source_rows)} rows; expected {expected_rows}"
            )
        source_ids = [str(row.get("id") or "") for row in source_rows]
        if any(not chunk_id for chunk_id in source_ids) or len(set(source_ids)) != expected_rows:
            raise RuntimeError("source collection contains missing or duplicate chunk IDs")

        store.collection_name = target_name
        store.ensure_collection(drop_existing=False)
        target = Collection(target_name, using=DEFAULT_CONFIG.zilliz_alias)
        target_fields = {field.name for field in target.schema.fields}
        required_target_fields = set(output_fields) | {"workspace_id", "sparse_vector"}
        missing_fields = required_target_fields - target_fields
        if missing_fields:
            raise RuntimeError(
                "target collection schema is missing required fields: " + ", ".join(sorted(missing_fields))
            )

        target_count = int(target.num_entities)
        existing_rows = (
            target.query(
                expr='id != ""',
                output_fields=["id", "workspace_id"],
                limit=max(target_count, 1),
                consistency_level="Strong",
            )
            if target_count
            else []
        )
        source_id_set = set(source_ids)
        for existing in existing_rows:
            if str(existing.get("workspace_id") or "") != workspace_id:
                raise RuntimeError("target collection already contains rows from another workspace")
            if str(existing.get("id") or "") not in source_id_set:
                raise RuntimeError("target collection contains rows outside the source inventory")

        migrated_rows = [
            build_workspace_row(
                row,
                workspace_id=workspace_id,
                has_sparse_vector="sparse_vector" in source_fields,
                sparse_vector_builder=lambda chunk: store._sparse_vector(vector_index_text(chunk)),
            )
            for row in source_rows
        ]
        for offset in range(0, len(migrated_rows), 500):
            target.upsert(migrated_rows[offset : offset + 500])
        target.flush()
        target.load()

        actual_rows = target.query(
            expr=f'workspace_id == {json.dumps(workspace_id, ensure_ascii=False)}',
            output_fields=["id", "workspace_id"],
            limit=max(expected_rows, 1),
            consistency_level="Strong",
        ) if expected_rows else []
        actual_ids = {str(row.get("id") or "") for row in actual_rows}
        if len(actual_rows) != expected_rows or actual_ids != source_id_set:
            raise RuntimeError(
                f"target verification failed: expected {expected_rows} workspace rows, found {len(actual_rows)}"
            )
        if int(target.num_entities) != expected_rows:
            raise RuntimeError(
                f"target collection contains {target.num_entities} rows after migration; expected {expected_rows}"
            )

        return {
            "source_collection": source_name,
            "target_collection": target_name,
            "workspace_id": workspace_id,
            "source_rows": expected_rows,
            "target_rows": len(actual_rows),
            "source_preserved": True,
        }
    finally:
        try:
            from pymilvus import connections

            connections.disconnect(DEFAULT_CONFIG.zilliz_alias)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-collection", required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    arguments = parser.parse_args()
    result = migrate_collection(
        target_name=arguments.target_collection,
        workspace_id=arguments.workspace_id,
        expected_rows=arguments.expected_rows,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
