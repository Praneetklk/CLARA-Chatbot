# scripts/opensearch_index_chunk.py

import uuid
from typing import Dict, Any

from core.config import settings
from core.logger import logger
from models.bedrock_models import bedrock_models
from integrations.opensearch_client import opensearch_client


def index_chunk_to_opensearch(
    text: str,
    metadata: Dict[str, Any],
    doc_id: str | None = None,
) -> str:
    """
    Generate an embedding for a chunk and index it into OpenSearch.

    Args:
        text: Chunk content (plain text)
        metadata: Dict of metadata (program, domain_code, chapter_code, etc.)
        doc_id: Optional explicit document ID. If None, a UUID will be generated.

    Returns:
        The document ID used in OpenSearch.
    """
    if not text or not text.strip():
        raise ValueError("Chunk text is empty; cannot index")

    # 1) Generate embedding via Bedrock
    logger.info(
        "Generating embedding for chunk",
        extra={"text_preview": text[:80]},
    )
    embedding = bedrock_models.generate_embedding(text)

    if not embedding:
        raise RuntimeError("Embedding generation returned empty vector")

    # 2) Choose or create document ID
    if doc_id is None:
        # Try to reuse any chunk_id in metadata, otherwise use UUID
        doc_id = metadata.get("chunk_id") or str(uuid.uuid4())

    # 3) Build document body compatible with vector_search.py expectations
    body: Dict[str, Any] = {
        "content": text,
        "metadata": metadata or {},
        settings.OPENSEARCH_EMBEDDING_FIELD: embedding,
    }

    logger.info(
        "Indexing chunk into OpenSearch",
        extra={
            "index": settings.OPENSEARCH_INDEX_NAME,
            "doc_id": doc_id,
            "program": metadata.get("program"),
            "chapter_code": metadata.get("chapter_code"),
            "domain_code": metadata.get("domain_code"),
        },
    )

    # 4) Index into OpenSearch
    opensearch_client.index_document(doc_id=doc_id, body=body)

    logger.info(
        "Successfully indexed chunk into OpenSearch",
        extra={"doc_id": doc_id},
    )
    return doc_id


if __name__ == "__main__":
    """
    Simple manual test entrypoint.

    You can run this directly to verify connectivity and mapping:

        python -m scripts.opensearch_index_chunk

    Make sure your .env has:
      - OPENSEARCH_ENDPOINT
      - OPENSEARCH_INDEX_NAME
      - OPENSEARCH_EMBEDDING_FIELD
      - BEDROCK_EMBEDDING_MODEL_ID
      - AWS_* creds / profile

    And that the OpenSearch index exists with:
      - a knn_vector field matching the embedding dimension.
    """
    sample_text = "Sample anesthesia services requirement text for testing."
    sample_metadata = {
        "program": "NIAHO-HOSP",
        "domain_code": "AS",
        "chapter_code": "AS.1",
        "chunk_id": "TEST-CHUNK-001",
        "source": "manual-test",
    }

    doc_id = index_chunk_to_opensearch(sample_text, sample_metadata)
    print(f"Indexed test chunk with doc_id={doc_id}")
