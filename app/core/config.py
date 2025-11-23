# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional, Dict, List


class Settings(BaseSettings):
    """
    Centralized application configuration.
    Grouped logically for readability; values unchanged.
    """
    
    # ------------------------------------------------------------
    # Project / Runtime
    # ------------------------------------------------------------
    PROJECT_NAME: str = "MedLaunch AI/ML Portal"
    DEBUG: bool = True
    ENABLE_CORS: bool = True
    
    # HTTP / API
    FRONTEND_ENDPOINT: str = ""
    BACKEND_ENDPOINT: str = ""
    RATE_LIMIT_MIN: str = "10"
    
    # Model generation defaults
    DEFAULT_MAX_TOKENS: int = 2000
    DEFAULT_TEMPERATURE: float = 0.2
    DEFAULT_TOP_P: float = 0.9
    
    # ------------------------------------------------------------
    # AWS Core
    # ------------------------------------------------------------
    AWS_REGION: str = "us-east-1"
    AWS_ACCOUNT_ID: str = "102282314102"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None
    
    # ------------------------------------------------------------
    # Knowledge Bases - OpenSearch Serverless "VXI60ESHGR"
    # ------------------------------------------------------------
    OPENSEARCH_KB_ID: str = "WDBAL7LGHY"
    KB_SEARCH_TYPE: str = "HYBRID"
    KB_RETRIEVE_TOP_K: int = 50
    INITIAL_RETRIEVAL_MULTIPLIER: int = 2
        # ------------------------------------------------------------
    # Direct OpenSearch Vector Index (bypassing Bedrock KB)
    # ------------------------------------------------------------
    OPENSEARCH_ENDPOINT: str = Field(
        "",
        description=(
            "OpenSearch endpoint for direct vector search, e.g. "
            "'https://your-domain.region.aoss.amazonaws.com'"
        ),
        env="OPENSEARCH_ENDPOINT",
    )
    OPENSEARCH_INDEX_NAME: str = Field(
        "dnv-chunks",
        description="OpenSearch index used to store chunk documents and vectors",
        env="OPENSEARCH_INDEX_NAME",
    )
    OPENSEARCH_EMBEDDING_FIELD: str = Field(
        "embedding",
        description="Field name in OpenSearch mapping that stores the knn_vector",
        env="OPENSEARCH_EMBEDDING_FIELD",
    )

    # Bedrock embedding model used for both ingestion & query vectors
    BEDROCK_EMBEDDING_MODEL_ID: str = Field(
        "amazon.titan-embed-text-v2:0",
        description="Bedrock embedding model ID for generating OpenSearch vectors",
        env="BEDROCK_EMBEDDING_MODEL_ID",
    )

    # Legacy KB (deprecated, kept for reference)
    BEDROCK_KB_ID: str = "VALTHOXJFF"
    
    # Router Classification to find the intent
    ENABLE_FIRST_TURN_ROUTER: bool = True
    ROUTER_MODEL_ID: str = "amazon.nova-micro-v1:0"
    ROUTER_MAX_TOKENS: int = 128
    ROUTER_TEMPERATURE: float = 0.1
    ROUTER_TOP_P: float = 0.9
    ROUTER_MAX_GREETING_WORDS: int = 25

    # ------------------------------------------------------------
    # Metadata Extraction & Filtering
    # ------------------------------------------------------------
    

    """
    Program aliases for smart natural language mapping
    Maps common user phrases to canonical program identifiers
    """
    PROGRAM_ALIASES: Dict[str, str] = {
        # NIAHO variations
        "hospital": "NIAHO-HOSP",
        "acute care": "NIAHO-HOSP",
        "acute care hospital": "NIAHO-HOSP",
        "hosp": "NIAHO-HOSP",
        "niaho hospital": "NIAHO-HOSP",
        
        "cah": "NIAHO-CAH",
        "critical access": "NIAHO-CAH",
        "critical access hospital": "NIAHO-CAH",
        "niaho cah": "NIAHO-CAH",
        
        "psychiatric": "NIAHO-PSY",
        "psy": "NIAHO-PSY",
        "behavioral health": "NIAHO-PSY",
        "niaho psychiatric": "NIAHO-PSY",
        
        "procedural": "NIAHO-PROC",
        "proc": "NIAHO-PROC",
        "niaho procedural": "NIAHO-PROC",
        
        # Specialty programs
        "cardiac": "CARDIAC_VASCULAR",
        "vascular": "CARDIAC_VASCULAR",
        "heart": "CARDIAC_VASCULAR",
        "cardiovascular": "CARDIAC_VASCULAR",
        
        "stroke": "STROKE",
        "stroke center": "STROKE",
        "stroke certification": "STROKE",
        
        "infection": "INFECTION",
        "infection control": "INFECTION",
        "infection prevention": "INFECTION",
        
        "cybersecurity": "CYBERSECURITY",
        "cyber": "CYBERSECURITY",
        "information security": "CYBERSECURITY",
        
        "glycemic": "GLYCEMIC",
        "glucose": "GLYCEMIC",
        "diabetes": "GLYCEMIC",
        
        "ortho": "ORTHO_SPINE",
        "orthopedic": "ORTHO_SPINE",
        "spine": "ORTHO_SPINE",
        
        "palliative": "PALLIATIVE",
        "palliative care": "PALLIATIVE",
        "hospice": "PALLIATIVE",
        
        "extracorporeal": "EXTRACORPOREAL",
        "ecmo": "EXTRACORPOREAL",
        
        "survey": "SURVEY",
        
        "ventricular": "VENTRICULAR",
        "vad": "VENTRICULAR"
    }
    
    """
    Valid program identifiers in the knowledge base
    Used for validation and program list display
    """
    VALID_PROGRAMS: List[str] = [
        "NIAHO-HOSP",
        "NIAHO-CAH",
        "NIAHO-PSY",
        "NIAHO-PROC",
        "CARDIAC_VASCULAR",
        "CYBERSECURITY",
        "EXTRACORPOREAL",
        "GLYCEMIC",
        "INFECTION",
        "ORTHO_SPINE",
        "PALLIATIVE",
        "STROKE",
        "SURVEY",
        "VENTRICULAR"
    ]
    
    """
    NIAHO sub-programs for clarification when user says just "NIAHO"
    """
    NIAHO_SUB_PROGRAMS: List[str] = [
        "NIAHO-HOSP",
        "NIAHO-CAH",
        "NIAHO-PSY",
        "NIAHO-PROC"
    ]
    
    # ------------------------------------------------------------
    # Confidence Thresholds
    # ------------------------------------------------------------
    
    """
    Low confidence threshold for reranking scores
    If average score falls below this, ask user for clarification
    """
    LOW_CONFIDENCE_THRESHOLD: float = 0.40
    
    """
    High confidence threshold indicating excellent match
    Used for logging and potential optimization decisions
    """
    HIGH_CONFIDENCE_THRESHOLD: float = 0.7
    
    """
    Minimum score threshold for including a document in results
    Documents below this score are filtered out before reranking
    """
    MIN_RETRIEVAL_SCORE: float = 0.1
    
    # ------------------------------------------------------------
    # Conversation Context Management
    # ------------------------------------------------------------
    
    """
    Enable program persistence across conversation turns
    When enabled, program from previous turn is reused if not specified
    """
    ENABLE_PROGRAM_PERSISTENCE: bool = True
    
    """
    Context keys for tracking metadata across conversation
    """
    CONTEXT_PROGRAM_KEY: str = "active_program"
    CONTEXT_LAST_QUERY_KEY: str = "last_query"
    CONTEXT_TURN_COUNT_KEY: str = "turn_count"
    
    # ------------------------------------------------------------
    # Redis Configuration 
    # ------------------------------------------------------------
    
    """
    Redis connection settings for conversation history and JTI cache
    """
    REDIS_HOST: str = Field(
        default="localhost",
        description="Redis server hostname (ElastiCache endpoint in production)"
    )
    REDIS_PORT: int = Field(
        default=6379,
        description="Redis server port"
    )
    REDIS_DB: int = Field(
        default=0,
        description="Redis database number (0-15)"
    )
    REDIS_PASSWORD: Optional[str] = Field(
        default=None,
        description="Redis password (optional, not needed with security groups)"
    )
    REDIS_SSL: bool = Field(
        default=True,
        description="Use TLS/SSL for Redis connection (required for ElastiCache with encryption)"
    )
    REDIS_SOCKET_TIMEOUT: int = Field(
        default=5,
        description="Socket timeout in seconds"
    )
    REDIS_SOCKET_CONNECT_TIMEOUT: int = Field(
        default=5,
        description="Socket connect timeout in seconds"
    )
    REDIS_MAX_CONNECTIONS: int = Field(
        default=50,
        description="Maximum connections in the pool"
    )
    
    # ------------------------------------------------------------
    # Conversation Management
    # ------------------------------------------------------------
    
    """
    Conversation history storage settings
    """
    CONVERSATION_TTL: int = Field(
        default=3600,
        description="Conversation TTL in seconds (3600 = 1 hour)"
    )
    MAX_CONVERSATION_TURNS: int = Field(
        default=20,
        description="Maximum turns to store per conversation (20 turns = 10 exchanges)"
    )
    ENABLE_CONVERSATION_HISTORY: bool = Field(
        default=True,
        description="Enable conversation history storage (can disable for testing)"
    )
    
    # ------------------------------------------------------------
    # Bedrock Agent (TwoPhaseComplianceAgent)
    # ------------------------------------------------------------
    BEDROCK_AGENT_ID: str = "KR8M1UIKM0"
    BEDROCK_AGENT_ALIAS_ID: str = "$LATEST"
    BEDROCK_AGENT_SESSION_TTL: int = 3600
    INVOKE_AGENT_TIMEOUT_SECS: int = 180
    
    # Agent KBs and retrieval defaults (S3 Vector Store limits: 1–30)
    KB_CHANGE_HISTORY_ID: str = "IONPGLEZYX"
    KB_POLICIES_ID: str = "OYKJME0HZX"
    CHANGE_KB_TOP_K: int = 30
    POLICIES_KB_TOP_K: int = 10
    CHANGE_KB_SEARCH_TYPE: str = "SEMANTIC"
    POLICIES_KB_SEARCH_TYPE: str = "SEMANTIC"
    
    # ------------------------------------------------------------
    # Multi‑Agent Policy Compliance (Supervisor + collaborators)
    # ------------------------------------------------------------
    POLICY_COMPLIANCE_SUPERVISOR_AGENT_ID: Optional[str] = "IC8BW9ZWQE"
    POLICY_COMPLIANCE_SUPERVISOR_AGENT_ALIAS_ID: Optional[str] = "ZYIBLHYKDS"
    POLICY_DISCOVERY_AGENT_ID: Optional[str] = "JM2A0ABLQS"
    POLICY_DISCOVERY_AGENT_ALIAS_ID: Optional[str] = "X4TTHZ4HH7"
    STANDARDS_MATCHING_AGENT_ID: Optional[str] = "SUYLW3GARL"
    STANDARDS_MATCHING_AGENT_ALIAS_ID: Optional[str] = "ID9OFQSDCX"
    COMPLIANCE_ANALYSIS_AGENT_ID: Optional[str] = "EQAAC9JSJX"
    COMPLIANCE_ANALYSIS_AGENT_ALIAS_ID: Optional[str] = "LV1ZXS0G7M"
    
    DEFAULT_ASSIGNEE_USER_ID: str = "686eebd1ef6460ea9d22a727"
    ENABLE_SUPERVISOR_FALLBACK: bool = True
    POLICY_COMPLIANCE_MIN_INTERVAL_SECS: float = 2.0
    POLICY_COMPLIANCE_MAX_RETRIES: int = 3
    
    # ------------------------------------------------------------
    # Knowledge Base IDs for Policy Compliance (console expectations)
    # ------------------------------------------------------------
    POLICIES_KB_ID: str = "OYKJME0HZX"
    NIAHO_STANDARDS_KB_ID: str = "VALTHOXJFF"
    GLOBAL_STANDARDS_KB_ID: str = "VALTHOXJFF"
    
    # ------------------------------------------------------------
    # S3 Storage
    # ------------------------------------------------------------
    AGENT_RUNS_BUCKET: str = "medlaunch-processed-analytics-data"
    AGENT_RUNS_PREFIX: str = "agent_runs"
    STRUCTURED_DATA_BUCKET: str = "medlaunch-structured-data"
    POLICIES_S3_ROOT: str = "hospitals"
    S3_ANALYTICS_BUCKET: str = "medlaunch-processed-analytics-data"
    
    # ------------------------------------------------------------
    # Models (Nova)
    # ------------------------------------------------------------
    NOVA_MICRO_MODEL_ID: str = "amazon.nova-micro-v1:0"
    NOVA_LITE_MODEL_ID: str = "amazon.nova-lite-v1:0"
    NOVA_PRO_MODEL_ID: str = "amazon.nova-pro-v1:0"
    NOVA_PROMPT_ROUTER_ARN: str
    
    # ------------------------------------------------------------
    # Re-ranking (Cohere)
    # ------------------------------------------------------------
    RERANK_MODEL_ID: str = "cohere.rerank-v3-5:0"
    RERANK_TOP_K: int = 5
    
    # ------------------------------------------------------------
    # Persistence (MongoDB)
    # ------------------------------------------------------------
    MONGODB_URI: Optional[str] = None
    MONGODB_DB_NAME: str = "medlaunch"
    MONGODB_COLLECTION_NAME: str = "policy_compliance_reports"
    
    # ------------------------------------------------------------
    # Messaging (SQS)
    # ------------------------------------------------------------
    SQS_QUEUE_URL: str = "https://sqs.us-east-1.amazonaws.com/102282314102/medlaunch-processing-queue"
    SQS_QUEUE_NAME: str = "medlaunch-processing-queue"
    SQS_REGION: str = "us-east-1"
    SQS_ENABLE_PUBLISH: bool = True
    
    # ------------------------------------------------------------
    # Security
    # ------------------------------------------------------------
    JWT_SECRET_KEY: str = Field(..., env="JWT_SECRET_KEY")
    JTI_CACHE_TTL_MINUTES: int = 15
    @property
    def JTI_CACHE_TTL_SECONDS(self) -> int:
        return self.JTI_CACHE_TTL_MINUTES * 60
    JWT_ALGORITHM: str = "HS256"
    JWT_AUDIENCE: str = "medlaunch-api"
    JWT_ISSUER: str = "medlaunch-auth"
    JWT_LEEWAY_SECONDS: int = 30
    JWT_REQUIRE_JTI: bool = True
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()