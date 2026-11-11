from __future__ import annotations

import json

import pytest

from scripts.migrate_workspace_vectors import build_workspace_row


def test_build_workspace_row_adds_scope_and_preserves_existing_vectors():
    dense = [0.25, -0.5]
    sparse = {7: 0.75}
    row = {
        "id": "chunk-1",
        "text": "campus information",
        "source": "guide.pdf",
        "page": 2,
        "block_id": "p2-b1",
        "modality": "ocr_text",
        "metadata_json": json.dumps({"quality": "high"}),
        "image_path": "",
        "vector": dense,
        "sparse_vector": sparse,
    }

    migrated = build_workspace_row(
        row,
        workspace_id="campus-default",
        has_sparse_vector=True,
        sparse_vector_builder=lambda _chunk: pytest.fail("existing sparse vectors should be preserved"),
    )

    assert migrated["workspace_id"] == "campus-default"
    assert json.loads(migrated["metadata_json"]) == {
        "quality": "high",
        "workspace_id": "campus-default",
    }
    assert migrated["vector"] is dense
    assert migrated["sparse_vector"] is sparse
    assert migrated["page"] == 2


def test_build_workspace_row_rejects_conflicting_workspace_metadata():
    row = {
        "id": "chunk-1",
        "text": "campus information",
        "source": "guide.pdf",
        "metadata_json": json.dumps({"workspace_id": "department-private"}),
        "vector": [0.25, -0.5],
    }

    with pytest.raises(ValueError, match="belongs to department-private"):
        build_workspace_row(
            row,
            workspace_id="campus-default",
            has_sparse_vector=False,
            sparse_vector_builder=lambda _chunk: {1: 1.0},
        )
