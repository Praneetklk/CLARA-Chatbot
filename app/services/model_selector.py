# services/model_selector.py
from schemas.request_models import UserTier, QueryComplexity
from typing import Optional, Tuple
from core.config import settings
from core.logger import logger

class ModelSelector:
    """Model selection and routing based on user tier"""
    
    def __init__(self):
        # Complexity indicators for metadata tracking
        self.simple_indicators = [
            "what is", "define", "list", "name", "when", "where",
            "yes or no", "true or false", "how many", "is it"
        ]
        
        self.complex_indicators = [
            "analyze", "compare", "evaluate", "assess", "develop",
            "strategy", "comprehensive", "detailed", "plan", "report",
            "create", "design", "implement", "optimize", "review"
        ]
        
        self.moderate_indicators = [
            "explain", "describe", "how", "why", "summarize",
            "what are the requirements", "provide", "outline",
            "compliance", "regulation", "standard", "requirement",
            "accreditation", "certification", "audit", "assessment"
        ]
    
    def select_model(
        self, 
        query: str, 
        user_tier: Optional[UserTier] = None,
        model_override: Optional[str] = None
    ) -> Tuple[str, QueryComplexity]:
        """Select model/router based on user tier"""
        # Default to free tier if not specified
        user_tier = user_tier or UserTier.FREE
        
        # Analyze complexity for metadata (not for routing decisions)
        complexity = self.analyze_complexity(query)
        
        # Handle explicit override
        if model_override:
            if "nova-micro" in model_override.lower():
                logger.info(f"Override: Using Nova Micro")
                return settings.NOVA_MICRO_MODEL_ID, QueryComplexity.SIMPLE
            elif "nova-router" in model_override.lower() or "prompt-router" in model_override.lower():
                logger.info(f"Override: Using Nova Prompt Router")
                return settings.NOVA_PROMPT_ROUTER_ARN, complexity
        
        if user_tier == UserTier.FREE:
            logger.info(f"Free tier: Routing to Nova Micro (complexity: {complexity.value})")
            return settings.NOVA_MICRO_MODEL_ID, complexity
        
        elif user_tier == UserTier.PAID:
            logger.info(f"{user_tier.value} tier: Routing to Nova Prompt Router (complexity: {complexity.value})")
            return settings.NOVA_PROMPT_ROUTER_ARN, complexity
        
        logger.warning(f"Unknown tier {user_tier}, defaulting to Nova Micro")
        return settings.NOVA_MICRO_MODEL_ID, complexity
    
    def analyze_complexity(self, query: str) -> QueryComplexity:
        """Analyze query complexity for metadata tracking"""
        query_lower = query.lower()
        word_count = len(query.split())
        question_marks = query.count('?')
        
        # Calculate indicator scores
        simple_score = sum(1 for ind in self.simple_indicators if ind in query_lower)
        complex_score = sum(1 for ind in self.complex_indicators if ind in query_lower)
        moderate_score = sum(1 for ind in self.moderate_indicators if ind in query_lower)
        
        # Length-based scoring
        if word_count < 10:
            simple_score += 2
        elif word_count > 50:
            complex_score += 2
        elif word_count > 30:
            moderate_score += 1
        
        # Multiple questions indicate complexity
        if question_marks > 1:
            complex_score += question_marks - 1
        
        # Determine complexity
        if complex_score >= 3 or (complex_score > simple_score and complex_score > moderate_score):
            return QueryComplexity.COMPLEX
        elif simple_score > moderate_score and word_count < 20:
            return QueryComplexity.SIMPLE
        else:
            return QueryComplexity.MODERATE
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def get_model_description(self, model_id: str) -> str:
        """Get human-readable model description"""
        if "nova-micro" in model_id.lower():
            return "Amazon Nova Micro"
        elif "nova-lite" in model_id.lower():
            return "Amazon Nova Lite"
        elif "nova-pro" in model_id.lower():
            return "Amazon Nova Pro"
        elif "prompt-router" in model_id.lower():
            return "Nova Prompt Router"
        return model_id

model_selector = ModelSelector()