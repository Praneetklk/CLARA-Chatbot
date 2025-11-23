# schemas/request_models.py
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any
from enum import Enum

class UserTier(str, Enum):
    """User subscription tiers"""
    FREE = "free"
    PAID = "paid"

class CountMethod(str, Enum):
    """Token counting methods"""
    BEDROCK = "bedrock"
    COUNT_TOKENS = "count_tokens"
    ESTIMATE = "estimate"
    NONE = "none"

class ModelType(str, Enum):
    """Model types for backward compatibility"""
    NOVA_MICRO = "nova-micro"
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"

class QueryComplexity(str, Enum):
    """Query complexity levels for metadata"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class ProgramType(str, Enum):
    """
    DNV Healthcare Standards Programs
    Comprehensive list of all available certification programs
    """
    NIAHO_HOSP = "NIAHO-HOSP"
    NIAHO_CAH = "NIAHO-CAH"
    NIAHO_PSY = "NIAHO-PSY"
    NIAHO_PROC = "NIAHO-PROC"
    CARDIAC_VASCULAR = "CARDIAC_VASCULAR"
    CYBERSECURITY = "CYBERSECURITY"
    EXTRACORPOREAL = "EXTRACORPOREAL"
    GLYCEMIC = "GLYCEMIC"
    INFECTION = "INFECTION"
    ORTHO_SPINE = "ORTHO_SPINE"
    PALLIATIVE = "PALLIATIVE"
    STROKE = "STROKE"
    SURVEY = "SURVEY"
    VENTRICULAR = "VENTRICULAR"

class ConfidenceLevel(str, Enum):
    """
    Confidence levels for metadata extraction and search results
    Used to determine if clarification is needed
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

class SearchType(str, Enum):
    """
    Knowledge base search types for AWS Bedrock
    """
    SEMANTIC = "SEMANTIC"
    HYBRID = "HYBRID"

class RouterDecision(BaseModel):
    intent: Literal["greeting", "acknowledgement", "out_of_scope", "other"]
    action: Literal["answer_here", "continue"]
    assistant_reply: str = Field(max_length=200, description="One short sentence")

# ============================================================================
# METADATA EXTRACTION MODELS
# ============================================================================

class MetadataFilter(BaseModel):
    """
    Structured filters for knowledge base retrieval.
    Converts to AWS Bedrock OpenSearch filter format via `to_bedrock_filter()`.

    Notes:
    - We ALWAYS include `program` when present.
    - If a full chapter code (e.g., "QM.1") is present, we AND it with program.
    - `edition` and `domain_code` are kept for future use (no KB filter output yet).
    """
    program: Optional[str] = Field(None, description="Program to filter by (strict filter), e.g., ACPC")
    edition: Optional[str] = Field(None, description="Edition (kept for future use)")
    domain_code: Optional[str] = Field(None, description="Domain prefix like QM, IC (optional hint)")
    chapter_code: Optional[str] = Field(None, description="Full chapter like QM.1 (when available)")

    def to_bedrock_filter(self) -> Optional[Dict[str, Any]]:
        """
        Convert to AWS Bedrock OpenSearch KB filter.

        Produces one of:
          None
          {"equals": {"key": "program", "value": "ACPC"}}
          {"and": [
              {"equals": {"key": "program", "value": "ACPC"}},
              {"equals": {"key": "chapter", "value": "QM.1"}}
          ]}
        """
        terms: List[Dict[str, Any]] = []

        if self.program:
            terms.append({"equals": {"key": "program", "value": self.program}})

        # Only include a chapter filter when we have a full code like "QM.1".
        if self.chapter_code:
            terms.append({"equals": {"key": "chapter_code", "value": self.chapter_code}})

        if self.domain_code:
            terms.append({"equals": {"key": "domain_code", "value": self.domain_code}})

        if not terms:
            return None
        if len(terms) == 1:
            return terms[0]
        return {"and": terms}

    class Config:
        json_schema_extra = {
            "example": {
                "program": "ACPC",
                "edition": None,
                "domain_code": "QM",
                "chapter_code": "QM.1"
            }
        }

class ExtractedMetadata(BaseModel):
    """
    Metadata extracted from user query and conversation context
    Output of metadata_extraction_service
    """
    program: Optional[str] = Field(None, description="Extracted program identifier (e.g., NIAHO-HOSP)")
    program_confidence: ConfidenceLevel = Field(..., description="Confidence level of program extraction")
    chapter_hints: List[str] = Field(default_factory=list, description="Chapter/domain codes mentioned (e.g., CC.1, QM)")
    needs_clarification: bool = Field(..., description="Whether user clarification is required")
    clarification_reason: Optional[str] = Field(None, description="Why clarification is needed")
    source: str = Field(..., description="Source of metadata: query, context, or default")

    # NEW FIELDS: Added to support the enhanced functionality
    filter: Optional[MetadataFilter] = Field(None, description="Structured filter for knowledge base retrieval")
    clarification_message: Optional[str] = Field(None, description="Pre-formatted clarification message for user")

    class Config:
        json_schema_extra = {
            "example": {
                "program": "NIAHO-HOSP",
                "program_confidence": "high",
                "chapter_hints": ["CC.1", "quality_management"],
                "needs_clarification": False,
                "clarification_reason": None,
                "source": "query",
                "filter": {"program": "NIAHO-HOSP"},
                "clarification_message": None
            }
        }

class ConfidenceMetrics(BaseModel):
    """
    Confidence metrics from reranking results
    Used to determine if results are sufficient or clarification is needed
    """
    avg_score: float = Field(..., ge=0.0, le=1.0, description="Average reranking score")
    max_score: float = Field(..., ge=0.0, le=1.0, description="Maximum reranking score")
    min_score: float = Field(..., ge=0.0, le=1.0, description="Minimum reranking score")
    num_results: int = Field(..., ge=0, description="Number of results retrieved")
    is_low_confidence: bool = Field(..., description="Whether confidence is below threshold")

    class Config:
        json_schema_extra = {
            "example": {
                "avg_score": 0.75,
                "max_score": 0.92,
                "min_score": 0.58,
                "num_results": 5,
                "is_low_confidence": False
            }
        }

class ClarificationResponse(BaseModel):
    """
    Response when clarification is needed from user
    Contains questions to help narrow down the query
    """
    needs_clarification: bool = Field(True, description="Always true for this response type")
    clarification_type: str = Field(..., description="Type of clarification: program, ambiguous, low_confidence")
    message: str = Field(..., description="Clarification message to display to user")
    suggested_programs: Optional[List[str]] = Field(None, description="List of suggested programs to choose from")
    detected_hints: Optional[Dict[str, Any]] = Field(None, description="Any hints detected from the query")

    class Config:
        json_schema_extra = {
            "example": {
                "needs_clarification": True,
                "clarification_type": "program",
                "message": "I'd be happy to help! Which program are you asking about?",
                "suggested_programs": ["NIAHO-HOSP", "NIAHO-CAH", "CARDIAC_VASCULAR"],
                "detected_hints": {"chapter": "CC.1", "topic": "organization"}
            }
        }

# ============================================================================
# EXISTING MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """
    Request model for DNV assistant queries
    """
    userPrompt: str = Field(..., description="User's question about DNV standards")
    user_tier: UserTier = Field(..., description="User subscription level")
    userId: str = Field(..., description="Unique user identifier")
    organizationId: str = Field(..., description="Organization identifier")
    conversationId: Optional[str] = Field(None, description="Conversation identifier for context")
    attachments: Optional[List[str]] = Field(None, description="Attachment text content to reference")
    context: Optional[Dict[str, Any]] = Field(None, description="Previous conversation context")
    timestamp: Optional[str] = Field(None, description="Request timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "userPrompt": "What are NIAHO infection control requirements?",
                "user_tier": "paid",
                "userId": "user-123e4567-e89b-12d3-a456-426614174000",
                "organizationId": "org-123e4567-e89b-12d3-a456-426614174000",
                "conversationId": "conv-123e4567-e89b-12d3-a456-426614174000",
                "attachments": ["Policy document content here..."],
                "context": {"previous_topic": "infection_control", "last_question": "general_requirements"},
                "timestamp": "2023-12-05T10:30:00Z"
            }
        }

class UsageBlock(BaseModel):
    """Token usage information"""
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")
    total_tokens: int = Field(..., description="Total number of tokens")
    invocations: int = Field(..., description="Number of model calls for this request")
    model: str = Field(..., description="Exact model ID used")
    count_method: CountMethod = Field(..., description="Method used for token counting")

class QueryResponse(BaseModel):
    """
    Response model for queries
    Enhanced with metadata extraction and confidence information
    """
    response: str = Field(..., description="AI-generated answer")
    model_used: str = Field(..., description="Human-readable model name")
    query_id: str = Field(..., description="Unique query identifier")
    sources: List[Dict[str, Any]] = Field(..., description="Retrieved source documents")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata including program, confidence, filters")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    usage: UsageBlock = Field(..., description="Token usage information")

    """
    Enhanced metadata fields (added to metadata dict):
    - program_used: str - Program filter applied
    - program_confidence: str - Confidence level of program extraction
    - search_type: str - HYBRID or SEMANTIC
    - filters_applied: Dict - Actual filters used in retrieval
    - confidence_metrics: Dict - Reranking confidence scores
    - clarification_triggered: bool - Whether clarification was needed
    """

# ============================================================================
# LEGACY SCHEMAS
# ============================================================================

class QuerySchema(BaseModel):
    """Legacy schema for /send_chat endpoint"""
    query: str = ""

class ChatResponseSchema(BaseModel):
    """Legacy response schema"""
    message: str = ""

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    bedrock_status: Optional[str] = None
    knowledge_base_status: Optional[str] = None
    sqs_status: Optional[str] = None

# ============================================================================
# AGENT MODELS
# ============================================================================

class AgentRunOverrides(BaseModel):
    """Override settings for agent run configuration"""
    change_kb_top_k: Optional[int] = Field(None, ge=1, le=30, description="Top-K results for Change KB (S3 Vector Store limit: 1-30)")
    policies_kb_top_k: Optional[int] = Field(None, ge=1, le=30, description="Top-K results for Policies KB (S3 Vector Store limit: 1-30)")
    change_kb_search_type: Optional[str] = Field("SEMANTIC", description="Search type for Change KB")
    policies_kb_search_type: Optional[str] = Field("SEMANTIC", description="Search type for Policies KB")
    batch_size: Optional[int] = Field(5, ge=1, le=10, description="Changes per batch for policy processing")
    compact: Optional[bool] = Field(False, description="Use compact response format to reduce JSON size")
    max_evidence_chars: Optional[int] = Field(240, ge=50, le=1000, description="Max evidence text length in compact mode")

class AgentRunRequest(BaseModel):
    """Request model for TwoPhaseComplianceAgent runs"""
    revision_id: str = Field(..., description="Revision ID to analyze")
    org_id: str = Field(..., description="Organization ID")
    policies: Optional[List[str]] = Field(None, description="Policy IDs to analyze (auto-discover if omitted)")
    overrides: Optional[AgentRunOverrides] = Field(None, description="Configuration overrides")
    write_to_s3: Optional[bool] = Field(True, description="Whether to write results to S3")

    class Config:
        json_schema_extra = {
            "example": {
                "revision_id": "25-1",
                "org_id": "6851a3a3a8e69d553a033e85",
                "policies": [
                    "adm.1_1_",
                    "medical_staff_policies",
                    "Periop-Nursing-Scope-Standards-of-Practice - SS example",
                    "ss.51_1_"
                ],
                "overrides": {
                    "change_kb_top_k": 80,
                    "policies_kb_top_k": 10
                },
                "write_to_s3": True
            }
        }

class EnumerationStats(BaseModel):
    """Statistics about change enumeration"""
    total_changes: int
    counts_by_standard: List[Dict[str, Any]]

class ChangeEnumerated(BaseModel):
    """Individual change item from enumeration"""
    standard_code: str
    standard_name: Optional[str] = None
    sr_code: str
    content_type: str
    title: str

class NearestNonMatch(BaseModel):
    """Information about nearest non-matching section"""
    candidate_section_id: str
    why_not_relevant: List[str]
    snippet: str

class Discovery(BaseModel):
    """Discovery phase results"""
    status: str
    section_id: str
    evidence: str
    nearest_nonmatch: NearestNonMatch

class Compliance(BaseModel):
    """Compliance analysis results"""
    status: str
    evidence_policy: str
    evidence_change: str
    gaps: List[str]
    recommendations: List[str]
    confidence: float

class ComplianceResult(BaseModel):
    """Individual compliance check result"""
    policy_id: str
    standard_code: str
    content_type: str
    title: str
    sr_code: str
    discovery: Discovery
    compliance: Compliance
    change_source_key: str
    policy_source_key: str

class PolicySummary(BaseModel):
    """Summary statistics per policy"""
    policy_id: str
    totals: Dict[str, int]

class StandardSummary(BaseModel):
    """Summary statistics per standard code"""
    standard_code: str
    changes: int
    matched_policies: List[str]

class Summaries(BaseModel):
    """Aggregated summaries"""
    per_policy: List[PolicySummary]
    per_standard_code: List[StandardSummary]

class RunDiagnostics(BaseModel):
    """Diagnostic information about the run"""
    enumeration: Dict[str, Any]
    policy_auto_discovery: Dict[str, Any]
    retrieval: Dict[str, Any]
    warnings: List[str]
    errors: List[str]

class AgentRunResponse(BaseModel):
    """Response model for agent run"""
    run_id: str = Field(..., description="Unique run identifier")
    revision_id: str = Field(..., description="Revision ID that was analyzed")
    org_id: str = Field(..., description="Organization ID")
    policies: List[str] = Field(..., description="Final list of policies used")
    s3_key: Optional[str] = Field(None, description="S3 location if written")
    summary: Dict[str, Any] = Field(..., description="Summary statistics from the analysis")

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "run_20250917_a1b2c3",
                "revision_id": "25-1",
                "org_id": "6851a3a3a8e69d553a033e85",
                "policies": ["adm.1_1_", "medical_staff_policies"],
                "s3_key": "s3://medlaunch-processed-analytics-data/agent_runs/run_20250917_a1b2c3/rev-25-1/results.json",
                "summary": {
                    "total_changes": 19,
                    "total_results": 18,
                    "enumeration_stats": {
                        "total_changes": 19,
                        "counts_by_standard": [
                            {"standard_code": "OB", "count": 4},
                            {"standard_code": "SS", "count": 6},
                            {"standard_code": "PE", "count": 4},
                            {"standard_code": "AN", "count": 1},
                            {"standard_code": "RC", "count": 1},
                            {"standard_code": "QM", "count": 2},
                            {"standard_code": "IC", "count": 1}
                        ]
                    },
                    "summaries": {
                        "per_policy": [
                            {
                                "policy_id": "medical_staff_policies",
                                "totals": {
                                    "changes": 18,
                                    "truly_relevant": 3,
                                    "no_truly_relevant": 15,
                                    "compliant": 2,
                                    "needs_update": 1,
                                    "not_comparable": 0
                                }
                            }
                        ]
                    },
                    "run_diagnostics": {
                        "warnings": [],
                        "errors": []
                    }
                }
            }
        }

# ============================================================================
# POLICY COMPLIANCE MODELS
# ============================================================================

class PolicyComplianceRequest(BaseModel):
    """Request model for the multi-agent policy compliance analysis"""

    facility_id: str = Field(..., alias="facilityId", description="Facility ID")
    file_name: str = Field(..., alias="fileName", description="Policy file name")
    scan_type: Optional[str] = Field("manual", alias="scanType", description="Scan type")
    program_selected: str = Field("Gen", description="NIAHO program type")
    edition_selected: str = Field("25-1", description="NIAHO edition")
    s3_file_id: Optional[str] = Field("", alias="s3FileId", description="S3 file ID for the policy document")
    user_id: Optional[str] = Field("", alias="userID", description="User ID who initiated the analysis")

class PolicyMetadata(BaseModel):
    policy_title: str
    total_sections: int
    sections_analyzed: int
    program_selected: str
    edition_selected: str
    analysis_timestamp: str

class Evidence(BaseModel):
    policy_quote: Optional[str] = None
    standard_quote: str
    policy_source_keys: Optional[List[str]] = None
    standard_source_keys: List[str]

class MissingComponent(BaseModel):
    component_type: str
    niaho_requirement: str
    severity: str
    description: str
    evidence: Evidence

class ComplianceGap(BaseModel):
    standard_code: str
    gap_type: str
    gap_description: str
    impact_level: str
    evidence: Evidence

class EnhancementOpportunity(BaseModel):
    enhancement_type: str
    opportunity_description: str
    business_value: str
    evidence: Evidence

class OverallAssessment(BaseModel):
    completeness_score: float
    compliance_score: float
    total_findings: int
    missing_components_count: int
    compliance_gaps_count: int
    enhancement_opportunities_count: int
    critical_issues_count: int
    recommendation_summary: str

class StandardsCoverage(BaseModel):
    total_standards_evaluated: int
    fully_addressed_standards: List[str]
    partially_addressed_standards: List[str]
    unaddressed_applicable_standards: List[str]

class PolicyAnalysis(BaseModel):
    policy_metadata: PolicyMetadata
    missing_essential_components: List[MissingComponent]
    compliance_gaps: List[ComplianceGap]
    policy_enhancement_opportunities: List[EnhancementOpportunity]
    overall_assessment: OverallAssessment
    standards_coverage_analysis: StandardsCoverage

class Report(BaseModel):
    policy_analysis: PolicyAnalysis

class Diagnostics(BaseModel):
    workflow_type: str
    supervisor_agent_id: str
    collaborator_agents_used: Optional[List[str]] = None
    sections_analyzed: Optional[int] = None
    standards_evaluated: Optional[int] = None
    processing_time_seconds: Optional[float] = None
    warnings: Optional[List[str]] = None
    errors: Optional[List[str]] = None

class PolicyComplianceResponse(BaseModel):
    """This is the new, comprehensive response model that matches the Supervisor Agent's output."""
    success: bool
    org_id: str
    policy_id: str
    report: Optional[Report] = None
    s3_location: Optional[str] = None
    diagnostics: Diagnostics
