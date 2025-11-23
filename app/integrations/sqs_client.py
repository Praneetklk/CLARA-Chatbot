# app/integrations/sqs_client.py
import json
import hashlib
from botocore.config import Config
from typing import Dict, Any, Optional
from core.config import settings
from core.logger import logger
from core.aws_client import get_sqs_client

_sqs = get_sqs_client()

def publish_json(
    envelope: Dict[str, Any],
    *,
    fifo: bool = False,
    group_id: Optional[str] = None
) -> str:
    """
    Publish a JSON message to SQS.
    Assumes body <= 256KB. If larger, send a small body with a pointer to S3 instead.
    """
    body = json.dumps(envelope, separators=(",", ":"), ensure_ascii=False)
    params = {
        "QueueUrl": settings.SQS_QUEUE_URL,
        "MessageBody": body,
        "MessageAttributes": {
            "job_id": {"DataType": "String", "StringValue": envelope.get("job_id", "")},
            "status": {"DataType": "String", "StringValue": envelope.get("status", "unknown")},
            "content_type": {"DataType": "String", "StringValue": "application/json"},
        },
    }
    if fifo:
        params["MessageGroupId"] = group_id or envelope.get("job_id", "default")
        params["MessageDeduplicationId"] = hashlib.sha256(body.encode("utf-8")).hexdigest()

    print(f"🔍 Debug - SQS Queue URL: {settings.SQS_QUEUE_URL}")
    print(f"🔍 Debug - SQS Region: {settings.SQS_REGION}")
    print(f"🔍 Debug - Message body size: {len(body)} bytes")
    
    resp = _sqs.send_message(**params)
    msg_id = resp.get("MessageId", "")
    print(f"🔍 Debug - SQS Response: {resp}")
    logger.info("SQS publish ok job_id=%s msg_id=%s", envelope.get("job_id"), msg_id)
    return msg_id
