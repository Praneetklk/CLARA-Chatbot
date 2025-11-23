from services.llm_service import llm_service
from schemas.request_models import QueryRequest, ModelType, UserTier
from core.logger import logger
from typing import Optional

def get_qa_response(query: str, model: Optional[str] = None) -> str:
    """
    Backward compatibility wrapper for the chat functionality.
    """
    try:
        # Convert string model to model_override if provided
        model_override = None
        if model:
            # Map legacy model names to current format
            if model.lower() in ["nova-micro", "micro"]:
                model_override = "nova-micro"
            elif model.lower() in ["nova-lite", "lite"]:
                model_override = "nova-lite"
            elif model.lower() in ["nova-pro", "pro"]:
                model_override = "nova-pro"
            else:
                model_override = model
        
        request = QueryRequest(
            query=query,
            user_tier=UserTier.FREE,  # Default to free tier for legacy compatibility
            model_override=model_override
        )
        
        response = llm_service.process_query(request)
        return response.response
        
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        fallback = (
            "Unable to process your question at this time. "
            "Please contact DNV representative (healthcare@dnv.com) for assistance."
        )
        return fallback