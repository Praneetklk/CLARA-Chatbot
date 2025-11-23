# services/vector_search.py

from typing import List, Dict, Any, Optional

from models.bedrock_models import bedrock_models
from schemas.request_models import MetadataFilter, ConfidenceMetrics
from core.config import settings
from core.logger import logger
from integrations.opensearch_client import opensearch_client


class VectorSearchService:
    """
    Handle retrieval from OpenSearch using Bedrock embeddings.

    Enhanced with:
    - Direct OpenSearch kNN search
    - Metadata filtering for program isolation
    - Confidence metrics calculation
    """

    def _build_opensearch_filter(
        self,
        metadata_filter: Optional[MetadataFilter],
    ) -> Optional[Dict[str, Any]]:
        """
        Convert MetadataFilter into an OpenSearch bool filter.

        Assumes your index stores metadata under flat fields:
          - program
          - domain_code
          - chapter_code
        """
        if not metadata_filter:
            return None

        must: List[Dict[str, Any]] = []

        if metadata_filter.program:
            must.append({"term": {"program": metadata_filter.program}})

        if metadata_filter.chapter_code:
            must.append({"term": {"chapter_code": metadata_filter.chapter_code}})

        if metadata_filter.domain_code:
            must.append({"term": {"domain_code": metadata_filter.domain_code}})

        if not must:
            return None

        return {"bool": {"filter": must}}

    def search(
        self,
        query: str,
        top_k: int = None,
        include_metadata: bool = False,
        use_rerank: bool = True,
        metadata_filter: Optional[MetadataFilter] = None,
    ) -> Dict[str, Any]:
        """
        Search using direct OpenSearch kNN with Bedrock embeddings.

        Process:
        1. Generate a query embedding via Bedrock embedding model.
        2. Perform kNN search on OpenSearch with optional metadata filter.
        3. Optionally rerank the results via Bedrock Rerank.
        4. Calculate confidence metrics and format for LLM consumption.
        """
        top_k = top_k or settings.KB_RETRIEVE_TOP_K

        try:
            # Empty query → short-circuit with low confidence
            if not query:
                logger.warning("Empty query passed to vector search")
                return {
                    "context": "",
                    "sources": None,
                    "confidence": ConfidenceMetrics(
                        avg_score=0.0,
                        max_score=0.0,
                        min_score=0.0,
                        num_results=0,
                        is_low_confidence=True,
                    ),
                    "num_results": 0,
                }

            # Over-fetch to give reranker more candidates if needed
            initial_k = max(top_k * settings.INITIAL_RETRIEVAL_MULTIPLIER, top_k)

            logger.info(
                "Starting vector search via OpenSearch",
                extra={
                    "query_preview": query[:100],
                    "top_k": top_k,
                    "initial_k": initial_k,
                },
            )

            # 1) Generate query embedding via Bedrock
            query_embedding = bedrock_models.generate_embedding(query)

            # 2) Build OpenSearch filter from metadata (program, chapter, domain)
            os_filter = self._build_opensearch_filter(metadata_filter)

            # 3) Perform kNN search against OpenSearch
            os_response = opensearch_client.knn_search(
                vector=query_embedding,
                k=initial_k,
                filter_query=os_filter,
            )

            hits = os_response.get("hits", {}).get("hits", [])
            if not hits:
                logger.warning(
                    "No results from OpenSearch",
                    extra={
                        "query": query[:100],
                        "filter": os_filter,
                        "index": opensearch_client.index,
                    },
                )
                return {
                    "context": "",
                    "sources": None,
                    "confidence": ConfidenceMetrics(
                        avg_score=0.0,
                        max_score=0.0,
                        min_score=0.0,
                        num_results=0,
                        is_low_confidence=True,
                    ),
                    "num_results": 0,
                }

            # 4) Normalize OpenSearch hits into internal 'chunk' format
            retrieved_chunks: List[Dict[str, Any]] = []
            for hit in hits:
                source = hit.get("_source", {}) or {}
                score = float(hit.get("_score", 0.0))

                metadata = {
                    "program": source.get("program"),
                    "domain_code": source.get("domain_code"),
                    "chapter_code": source.get("chapter_code"),
                    "edition": source.get("edition"),
                    "effective_date": source.get("effective_date"),
                    "program_family": source.get("program_family"),
                    "scope": source.get("scope"),
                    "page_range": source.get("page_range"),
                    "page_start": source.get("page_start"),
                    "page_end": source.get("page_end"),
                    "chunk_id": source.get("chunk_id"),
                    "chapter_title": source.get("chapter_title"),
                }

                chunk = {
                    "content": {
                        "text": source.get("content", ""),
                    },
                    "score": score,
                    "metadata": {k: v for k, v in metadata.items() if v is not None},
                    "location": {
                        "type": "opensearch",
                        "index": opensearch_client.index,
                        "id": hit.get("_id"),
                    },
                }
                retrieved_chunks.append(chunk)

            # 5) Optional reranking via Bedrock Rerank
            if use_rerank and len(retrieved_chunks) > top_k:
                try:
                    logger.info(
                        "Applying Bedrock Rerank to OpenSearch results",
                        extra={
                            "initial_k": len(retrieved_chunks),
                            "top_k": top_k,
                        },
                    )
                    reranked_chunks = bedrock_models.rerank_documents(
                        query=query,
                        documents=retrieved_chunks,
                        top_k=top_k,
                    )
                    if reranked_chunks:
                        logger.info(
                            f"Re-ranked {len(reranked_chunks)} documents "
                            f"from {len(retrieved_chunks)} initial OpenSearch results",
                        )
                        retrieved_chunks = reranked_chunks
                    else:
                        retrieved_chunks = retrieved_chunks[:top_k]
                except Exception as e:
                    logger.error(
                        f"Reranking failed, falling back to raw OpenSearch scores: {str(e)}"
                    )
                    retrieved_chunks = sorted(
                        retrieved_chunks,
                        key=lambda c: c.get("score", 0.0),
                        reverse=True,
                    )[:top_k]
            else:
                # No rerank → just take top_k by score
                retrieved_chunks = sorted(
                    retrieved_chunks,
                    key=lambda c: c.get("score", 0.0),
                    reverse=True,
                )[:top_k]

            # 6) Calculate confidence + build context and sources
            confidence = self._calculate_confidence_metrics(retrieved_chunks)
            context = self._format_context(retrieved_chunks)

            sources = None
            if include_metadata:
                sources = self._format_sources(retrieved_chunks)

            return {
                "context": context,
                "sources": sources,
                "confidence": confidence,
                "num_results": len(retrieved_chunks),
            }

        except Exception as e:
            logger.error(f"Vector search via OpenSearch failed: {str(e)}")

            return {
                "context": "",
                "sources": None,
                "confidence": ConfidenceMetrics(
                    avg_score=0.0,
                    max_score=0.0,
                    min_score=0.0,
                    num_results=0,
                    is_low_confidence=True,
                ),
                "num_results": 0,
            }

    def _calculate_confidence_metrics(self, chunks: List[Dict]) -> ConfidenceMetrics:
        """
        Calculate confidence metrics from retrieval scores.

        Low confidence indicates query may be too vague or results not relevant.
        """
        if not chunks:
            return ConfidenceMetrics(
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                num_results=0,
                is_low_confidence=True,
            )

        scores = [chunk.get("score", 0.0) for chunk in chunks]
        scores = [s for s in scores if s > 0]

        if not scores:
            return ConfidenceMetrics(
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                num_results=len(chunks),
                is_low_confidence=True,
            )

        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)

        is_low_confidence = avg_score < settings.LOW_CONFIDENCE_THRESHOLD

        logger.info(
            "Confidence metrics calculated",
            extra={
                "avg_score": round(avg_score, 3),
                "max_score": round(max_score, 3),
                "min_score": round(min_score, 3),
                "num_results": len(chunks),
                "is_low_confidence": is_low_confidence,
            },
        )

        return ConfidenceMetrics(
            avg_score=round(avg_score, 3),
            max_score=round(max_score, 3),
            min_score=round(min_score, 3),
            num_results=len(chunks),
            is_low_confidence=is_low_confidence,
        )

    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string for LLM."""
        if not chunks:
            return ""

        context_parts = []
        for chunk in chunks:
            text = chunk.get("content", {}).get("text", "")
            if not text:
                continue

            metadata = chunk.get("metadata", {})
            prefix_parts = []

            if "program" in metadata:
                prefix_parts.append(f"[Program: {metadata['program']}]")
            if "chapter_code" in metadata:
                prefix_parts.append(f"[Chapter: {metadata['chapter_code']}]")

            prefix = " ".join(prefix_parts)
            if prefix:
                context_parts.append(f"{prefix} {text}")
            else:
                context_parts.append(text)

        return "\n\n---\n\n".join(context_parts)

    def _format_sources(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Format source chunks for response."""
        sources = []
        for chunk in chunks[:5]:
            sources.append(
                {
                    "content": chunk.get("content", {}).get("text", "")[:500],
                    "score": chunk.get("score", 0.0),
                    "metadata": chunk.get("metadata", {}),
                    "location": chunk.get("location", {}),
                }
            )
        return sources


vector_search_service = VectorSearchService()
