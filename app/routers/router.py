# routers/router.py
"""
FastAPI Router for Policy Compliance and Chat Services
"""

from typing import Any, Dict

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
    status
)
from uuid import uuid4
import time

from core.auth import AuthenticatedPrincipal, verify_jwt_token
from core.logger import logger
from core.rate_limiter import limit_param, limiter
from core.config import settings
from integrations.sqs_client import publish_json
from schemas.request_models import (
    AgentRunRequest,
    AgentRunResponse,
    ChatResponseSchema,
    CountMethod,
    HealthResponse,
    PolicyComplianceRequest,
    QueryRequest,
    QueryResponse,
    QuerySchema,
)
from schemas.sqs_models import JobEnvelope, ResultRef
from services.agent_service import AgentService
from services.llm_service import llm_service
from services.policy_compliance_service import policy_compliance_service


# ============================================================================
# ROUTER CONFIGURATION
# ============================================================================

router = APIRouter(
    prefix="/api/v1",
    tags=["AI Services"],
    responses={
        403: {"description": "Forbidden - Invalid JWT"},
        429: {"description": "Too Many Requests"},
        500: {"description": "Internal Server Error"}
    }
)

agent_service = AgentService()


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Validates service connectivity and dependencies"
)
@limiter.limit(limit_param)
async def check_health(request: Request) -> HealthResponse:
    """
    Comprehensive health check for all service dependencies.
    
    Checks:
    - Bedrock connectivity
    - Knowledge base configuration
    - SQS availability (if enabled)
    """
    health_status = HealthResponse(
        status="healthy",
        message="DNV AI Assistant is operational"
    )
    
    # Check Bedrock connectivity
    try:
        from core.aws_client import get_bedrock_client
        bedrock_client = get_bedrock_client()
        bedrock_client.list_foundation_models()
        health_status.bedrock_status = "connected"
    except Exception as e:
        logger.error(f"Bedrock health check failed: {e}")
        health_status.bedrock_status = f"error: {str(e)[:100]}"
        health_status.status = "degraded"
    
    # Check Knowledge Base configuration
    try:
        from core.aws_client import get_bedrock_agent_client
        bedrock_agent = get_bedrock_agent_client()
        health_status.knowledge_base_status = "configured"
    except Exception as e:
        logger.error(f"Knowledge base health check failed: {e}")
        health_status.knowledge_base_status = f"error: {str(e)[:100]}"
        health_status.status = "degraded"
    
    # Check SQS if enabled
    if settings.SQS_ENABLE_PUBLISH:
        try:
            from core.aws_client import get_sqs_client
            sqs = get_sqs_client()
            sqs.get_queue_attributes(
                QueueUrl=settings.SQS_QUEUE_URL,
                AttributeNames=["QueueArn"]
            )
            health_status.sqs_status = "connected"
        except Exception as e:
            logger.error(f"SQS health check failed: {e}")
            health_status.sqs_status = f"error: {str(e)[:100]}"
            health_status.status = "degraded"
    else:
        health_status.sqs_status = "disabled"
    
    return health_status


# ============================================================================
# Clara Chatbot ENDPOINTS
# ============================================================================

@router.post(
    "/chat",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="AI Chat Assistant",
    description="Process queries about DNV standards with AI assistance"
)
@limiter.limit(limit_param)
async def chat_endpoint(
    request: Request,
    response: Response,
    body: QueryRequest,
    principal: AuthenticatedPrincipal = Depends(verify_jwt_token)
) -> QueryResponse:
    """
    AI-powered chat endpoint for DNV standards queries.
    
    Model Routing:
    - Free tier: Nova Micro (efficient, fast responses)
    - Paid tier: Nova Prompt Router (intelligent routing between models)
    
    Response includes:
    - AI-generated answer
    - Source documents from knowledge base
    - Token usage metrics
    - Processing metadata
    """
    request_id = principal.request_id
    
    try:
        query_response = llm_service.process_query(body)
        
        # Set observability headers
        response.headers["x-request-id"] = request_id
        response.headers["x-model-used"] = query_response.usage.model
        response.headers["x-processing-ms"] = str(query_response.processing_time_ms)
        response.headers["x-usage-count-method"] = query_response.usage.count_method.value
        
        # Add token metrics if available
        if query_response.usage.count_method != CountMethod.NONE:
            response.headers["x-tokens-prompt"] = str(query_response.usage.prompt_tokens)
            response.headers["x-tokens-completion"] = str(query_response.usage.completion_tokens)
            response.headers["x-tokens-total"] = str(query_response.usage.total_tokens)
        
        logger.info(
            f"Query processed: id={query_response.query_id}, "
            f"model={query_response.usage.model}, "
            f"tokens={query_response.usage.total_tokens}"
        )
        
        return query_response
        
    except Exception as e:
        response.headers["x-request-id"] = request_id
        logger.exception(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


# ============================================================================
# AGENT ENDPOINTS
# ============================================================================

@router.post(
    "/agent/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Compliance Agent",
    description="Execute TwoPhaseComplianceAgent for policy analysis"
)
@limiter.limit(limit_param)
async def run_compliance_agent(
    request: Request,
    body: AgentRunRequest,
    principal: AuthenticatedPrincipal = Depends(verify_jwt_token)
) -> AgentRunResponse:
    """
    Execute compliance analysis across multiple policies.
    
    Features:
    - Auto-discovery of policies if not specified
    - Batched processing for efficiency
    - Matrix completeness guarantee
    - S3 storage of results
    
    Response includes complete analysis matrix with
    len(results) == len(policies) × len(changes_enumerated)
    """
    try:
        response = await agent_service.run_agent(body)
        
        logger.info(
            f"Agent run completed: run_id={response.run_id}, "
            f"policies={len(response.policies)}, "
            f"s3_key={response.s3_key}"
        )
        
        return response
        
    except Exception as e:
        logger.exception(f"Agent run failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent run failed: {str(e)}"
        )


# ============================================================================
# POLICY COMPLIANCE ENDPOINTS
# ============================================================================

@router.post(
    "/policy-compliance/analyze",
    status_code=status.HTTP_200_OK,
    summary="Analyze Policy Compliance",
    description="Synchronous policy compliance analysis"
)
@limiter.limit(limit_param)
async def analyze_policy_compliance(
    request: Request,
    body: PolicyComplianceRequest,
    principal: AuthenticatedPrincipal = Depends(verify_jwt_token)
) -> Dict[str, Any]:
    """
    Analyze policy compliance against NIAHO standards.
    
    Uses multi-agent collaboration:
    - Supervisor agent coordinates the workflow
    - Policy discovery agent extracts policy information
    - Standards matching agent maps to NIAHO standards
    - Compliance analysis agent performs assessment
    
    Returns complete analysis with findings, scores, and recommendations.
    """
    try:
        logger.info(
            f"Starting policy compliance analysis: "
            f"facility={body.facility_id}, file={body.file_name}"
        )
        
        result = await policy_compliance_service.analyze_policy_compliance(
            facility_id=body.facility_id,
            file_name=body.file_name,
            scan_type=body.scan_type,
            program_selected=body.program_selected,
            edition_selected=body.edition_selected,
            s3_file_id=body.s3_file_id,
            user_id=body.user_id,
            defaults=getattr(body, 'defaults', None)
        )
        
        logger.info(
            f"Analysis completed: "
            f"score={result.get('scanScore', 0)}, "
            f"grade={result.get('scanGrade', 'N/A')}"
        )
        
        return result
        
    except Exception as e:
        logger.exception(f"Policy compliance analysis failed: {e}")
        
        # Return standardized error response
        return {
            "facilityId": body.facility_id,
            "fileName": body.file_name,
            "scanType": body.scan_type,
            "affectedPolicies": [],
            "overallSummary": f"Analysis failed: {str(e)}",
            "scanScore": 0.0,
            "scanGrade": "F",
            "scanStatus": "Failed",
            "complianceStatus": "UNKNOWN",
            "s3FileId": body.s3_file_id,
            "userId": body.user_id,
            "diagnostics": {
                "error": str(e),
                "timestamp": time.time()
            }
        }


@router.post(
    "/policy-compliance/submit",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Policy Compliance Job",
    description="Asynchronous policy compliance analysis with SQS notification"
)
@limiter.limit(limit_param)
async def submit_policy_compliance(
    request: Request,
    body: PolicyComplianceRequest,
    background_tasks: BackgroundTasks,
    principal: AuthenticatedPrincipal = Depends(verify_jwt_token)
) -> Dict[str, str]:
    """
    Submit policy compliance analysis as background job.
    
    Process:
    1. Immediately returns job_id with 202 Accepted
    2. Runs analysis in background
    3. Publishes results to SQS queue
    
    Use this endpoint for long-running analyses or when
    results can be processed asynchronously.
    """
    job_id = str(uuid4())
    
    async def _run_and_publish():
        """Background task for running analysis and publishing results."""
        start_time = time.time()
        status_str = "success"
        
        logger.info(
            f"Starting background job: job_id={job_id}, "
            f"facility={body.facility_id}, file={body.file_name}"
        )
        
        try:
            # Run analysis
            result_dict = await policy_compliance_service.analyze_policy_compliance(
                facility_id=body.facility_id,
                file_name=body.file_name,
                scan_type=body.scan_type,
                program_selected=body.program_selected,
                edition_selected=body.edition_selected,
                s3_file_id=body.s3_file_id,
                user_id=body.user_id,
                defaults=getattr(body, 'defaults', None)
            )
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Job completed: job_id={job_id}, "
                f"duration={processing_time:.2f}s"
            )
            
        except Exception as e:
            logger.exception(f"Job failed: job_id={job_id}")
            status_str = "error"
            result_dict = {"error": str(e)}
        
        # Prepare SQS envelope
        result_ref = ResultRef()
        
        if status_str == "success":
            # Map fields for backend compatibility
            mapped_result = _map_result_for_sqs(result_dict, body)
            result_ref.inline = mapped_result
        else:
            # Error case - minimal response
            result_ref.inline = {
                "success": False,
                "facilityId": body.facility_id,
                "fileName": body.file_name,
                "error": str(result_dict.get("error", "Unknown error"))
            }
        
        # Create SQS envelope
        envelope = JobEnvelope(
            job_id=job_id,
            status=status_str,
            result_ref=result_ref,
            metadata={
                "facilityId": body.facility_id,
                "fileName": body.file_name,
                "program_selected": body.program_selected,
                "edition_selected": body.edition_selected,
                "duration_ms": int((time.time() - start_time) * 1000),
                "endpoint": "/api/v1/policy-compliance/submit"
            },
            published_at_ms=int(time.time() * 1000)
        ).dict()
        
        # Publish to SQS if enabled
        if settings.SQS_ENABLE_PUBLISH:
            try:
                msg_id = publish_json(envelope, fifo=False)
                logger.info(f"SQS message published: job_id={job_id}, msg_id={msg_id}")
            except Exception as e:
                logger.error(f"SQS publish failed: job_id={job_id}, error={e}")
        else:
            logger.info(f"SQS publishing disabled for job_id={job_id}")
    
    # Add background task
    background_tasks.add_task(_run_and_publish)
    
    return {
        "job_id": job_id,
        "status": "accepted"
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _map_result_for_sqs(
    result: Dict[str, Any],
    request: PolicyComplianceRequest
) -> Dict[str, Any]:
    """
    Map result fields for backend compatibility.
    
    Performs field name transformations and enum value mappings
    to match backend expectations.
    """
    mapped = result.copy()
    
    # Add organization and policy IDs
    mapped["org_id"] = request.facility_id
    mapped["policy_id"] = request.file_name
    
    # Ensure userId and s3FileId are preserved from request
    mapped["userId"] = request.user_id
    mapped["s3FileId"] = request.s3_file_id
    
    # Map scan type enum
    if mapped.get("scanType") == "manual":
        mapped["scanType"] = "MANUAL"
    
    # Map impact levels in affected policies
    if "affectedPolicies" in mapped and isinstance(mapped["affectedPolicies"], list):
        for finding in mapped["affectedPolicies"]:
            if isinstance(finding, dict) and "impactLevel" in finding:
                impact_level = finding["impactLevel"]
                
                # Map to backend enum values
                impact_mapping = {
                    "High": "High Priority",
                    "Medium": "Medium Priority",
                    "Low": "Low Priority"
                }
                
                if impact_level in impact_mapping:
                    finding["impactLevel"] = impact_mapping[impact_level]
    
    return mapped


# ============================================================================
# LEGACY ENDPOINTS (DEPRECATED)
# ============================================================================

@router.post(
    "/send_chat",
    status_code=status.HTTP_200_OK,
    deprecated=True,
    include_in_schema=False
)
@limiter.limit(limit_param)
async def send_chat_legacy(
    request: Request,
    body: QuerySchema
) -> ChatResponseSchema:
    """
    Legacy endpoint - deprecated.
    
    This endpoint is maintained only for backward compatibility
    with legacy systems. New integrations should use /api/v1/chat.
    """
    logger.warning("Legacy /send_chat endpoint called - consider migrating to /api/v1/chat")
    
    try:
        query_request = QueryRequest(
            userPrompt=body.query,
            user_tier="free",
            userId="legacy-user",
            organizationId="legacy-org",
            conversationId=None,
            attachments=None,
            context=None
        )
        
        response = llm_service.process_query(query_request)
        
        return ChatResponseSchema(message=response.response)
        
    except Exception as e:
        logger.exception("Legacy chat endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )