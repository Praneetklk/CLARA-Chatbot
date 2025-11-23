import json
from datetime import datetime
from core.logger import logger

def log_response(
    query: str,
    response: str,
    model_used: str,
    context: str = None,
    query_id: str = None,
    user_id: str = None
) -> str:
    """
    Enhanced logging for chat responses with model tracking.
    """
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": "chat_response",
        "query_id": query_id,
        "user_id": user_id,
        "model_used": model_used,
        "query": query[:500],  # Truncate long queries
        "response_preview": response[:500],  # Truncate long responses
        "response_length": len(response),
        "context_length": len(context) if context else 0
    }
    
    # Check if response seems like a failure
    failure_indicators = [
        "unable to answer",
        "don't know",
        "contact DNV",
        "insufficient information"
    ]
    
    if any(indicator in response.lower() for indicator in failure_indicators):
        log_data["event"] = "chat_response_partial"
        logger.warning(json.dumps(log_data))
    else:
        logger.info(json.dumps(log_data))
    
    return response