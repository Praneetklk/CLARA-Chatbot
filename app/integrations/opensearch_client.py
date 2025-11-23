# integrations/opensearch_client.py

import json
from typing import Any, Dict, List, Optional

import boto3
import requests
from requests_aws4auth import AWS4Auth

from core.config import settings
from core.logger import logger


def _detect_service_for_endpoint(endpoint: str) -> str:
    """
    Detect the correct AWS service name based on the endpoint.

    - OpenSearch Serverless (AOSS): usually 'aoss' and endpoint ends with '.aoss.amazonaws.com'
    - Classic OpenSearch / ES domains: usually 'es'
    """
    endpoint = (endpoint or "").lower()
    if "aoss.amazonaws.com" in endpoint:
        return "aoss"
    return "es"


def _build_awsauth() -> AWS4Auth:
    """
    Build AWS4Auth for OpenSearch using current AWS credentials.

    Priority:
    1. Explicit credentials from settings / environment (AWS_ACCESS_KEY_ID, etc.)
    2. boto3 Session (optionally with AWS_PROFILE / AWS_PROFILE_NAME)
    """
    service = _detect_service_for_endpoint(settings.OPENSEARCH_ENDPOINT)

    # 1) Try explicit credentials from settings / env first
    access_key = getattr(settings, "AWS_ACCESS_KEY_ID", None) or None
    secret_key = getattr(settings, "AWS_SECRET_ACCESS_KEY", None) or None
    session_token = getattr(settings, "AWS_SESSION_TOKEN", None) or None

    if access_key and secret_key:
        return AWS4Auth(
            access_key,
            secret_key,
            settings.AWS_REGION,
            service,
            session_token=session_token,
        )

    # 2) Fallback to boto3 Session (respect AWS_PROFILE / AWS_PROFILE_NAME)
    profile_name = (
        getattr(settings, "AWS_PROFILE", None)
        or getattr(settings, "AWS_PROFILE_NAME", None)
        or None
    )

    if profile_name:
        session = boto3.Session(profile_name=profile_name, region_name=settings.AWS_REGION)
    else:
        session = boto3.Session(region_name=settings.AWS_REGION)

    creds = session.get_credentials()
    if not creds:
        # Give a clear error instead of AttributeError: 'NoneType'...
        raise RuntimeError(
            "No AWS credentials found for OpenSearch. "
            "Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and optionally AWS_SESSION_TOKEN) "
            "or configure an AWS_PROFILE that has access to Bedrock and OpenSearch."
        )

    frozen = creds.get_frozen_credentials()

    return AWS4Auth(
        frozen.access_key,
        frozen.secret_key,
        settings.AWS_REGION,
        service,
        session_token=frozen.token,
    )



class OpenSearchClient:
    """
    Thin wrapper around the OpenSearch HTTP API for:
    - Indexing chunk documents with embeddings
    - Running kNN vector search with optional filters
    """

    def __init__(self) -> None:
        if not settings.OPENSEARCH_ENDPOINT:
            raise ValueError("OPENSEARCH_ENDPOINT is not configured")

        self.endpoint = settings.OPENSEARCH_ENDPOINT.rstrip("/")
        self.index = settings.OPENSEARCH_INDEX_NAME
        self.embedding_field = settings.OPENSEARCH_EMBEDDING_FIELD
        self.auth = _build_awsauth()
        self._headers = {"Content-Type": "application/json"}

    # ------------------------------------------------------------------
    # Ingestion / Indexing
    # ------------------------------------------------------------------

    def index_document(self, doc_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Index or update a single document in OpenSearch.

        Args:
            doc_id: Unique ID for the document (e.g., chunk_id, hash)
            body: Dict that should include:
                  - "content": chunk text
                  - "metadata": metadata dict
                  - embedding field: settings.OPENSEARCH_EMBEDDING_FIELD (List[float])

        Returns:
            OpenSearch index API response as dict.
        """
        url = f"{self.endpoint}/{self.index}/_doc/{doc_id}"
        logger.info(
            "Indexing document into OpenSearch",
            extra={"index": self.index, "doc_id": doc_id},
        )

        resp = requests.put(
            url,
            auth=self.auth,
            headers=self._headers,
            data=json.dumps(body),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Retrieval / kNN Search
    # ------------------------------------------------------------------

    def knn_search(
        self,
        vector: List[float],
        k: int,
        filter_query: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a kNN search on the vector field, with optional structured filters.

        Args:
            vector: Query embedding vector
            k: Number of results to retrieve
            filter_query: Optional OpenSearch bool filter, e.g.:
                {
                    "bool": {
                        "filter": [
                            {"term": {"metadata.program.keyword": "NIAHO-HOSP"}},
                            {"term": {"metadata.chapter_code.keyword": "QM.1"}},
                        ]
                    }
                }

        Returns:
            Raw OpenSearch search response as dict.
        """
        body: Dict[str, Any] = {
            "size": k,
            "query": {
                "knn": {
                    self.embedding_field: {
                        "vector": vector,
                        "k": k,
                    }
                }
            },
        }

        # Use post_filter so scoring is driven by kNN, but we still restrict results
        if filter_query:
            body["post_filter"] = filter_query

        url = f"{self.endpoint}/{self.index}/_search"
        logger.info(
            "Executing OpenSearch kNN search",
            extra={"index": self.index, "k": k},
        )

        resp = requests.post(
            url,
            auth=self.auth,
            headers=self._headers,
            data=json.dumps(body),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# Global singleton instance
opensearch_client = OpenSearchClient()
