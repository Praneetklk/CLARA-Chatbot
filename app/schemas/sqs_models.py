# app/schemas/sqs_models.py
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class ResultRef(BaseModel):
    inline: Optional[Dict[str, Any]] = None  # small results
    s3_uri: Optional[str] = None             # large results

class JobEnvelope(BaseModel):
    job_id: str
    status: str = Field(..., pattern="^(success|error)$")
    result_ref: ResultRef
    metadata: Dict[str, Any] = {}
    published_at_ms: int
    version: int = 1
