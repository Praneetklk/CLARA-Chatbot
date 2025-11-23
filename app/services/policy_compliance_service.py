"""
Policy Compliance Service

This module orchestrates AWS Bedrock's multi-agent collaboration system for
policy compliance analysis against NIAHO standards.
"""

import json
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field
from core.logger import logger
from core.config import settings


# ============================================================================
# DOMAIN MODELS
# ============================================================================

class AgentConfig(BaseModel):
    """Configuration for a Bedrock agent."""
    agent_id: str
    alias_id: str
    name: str
    role: str


class ComplianceAnalysisRequest(BaseModel):
    """Request parameters for compliance analysis."""
    facility_id: str
    file_name: str
    scan_type: str = "manual"
    program_selected: str = "Gen"
    edition_selected: str = "25-1"
    s3_file_id: str = ""
    user_id: str = ""
    session_id: Optional[str] = None
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID for this analysis."""
        if self.session_id:
            return self.session_id
        timestamp = int(time.time())
        return f"policy_compliance_{self.facility_id}_{timestamp}"


class ComplianceAnalysisResult(BaseModel):
    """Result of compliance analysis."""
    facility_id: str
    file_name: str
    scan_type: str
    affected_policies: List[Dict[str, Any]]
    overall_summary: str
    scan_score: float
    scan_grade: str
    scan_status: str
    compliance_status: str
    s3_file_id: str
    user_id: str
    aws_response_id: Optional[str] = None
    storage_location: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# INTERFACES (PROTOCOLS)
# ============================================================================

class AgentInvoker(Protocol):
    """Interface for agent invocation."""
    
    async def invoke_agent(
        self,
        config: AgentConfig,
        session_id: str,
        input_data: Dict[str, Any],
        knowledge_bases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Invoke a Bedrock agent."""
        ...


class ResponseProcessor(Protocol):
    """Interface for processing agent responses."""
    
    def process_response(
        self,
        raw_response: Dict[str, Any],
        request: ComplianceAnalysisRequest
    ) -> ComplianceAnalysisResult:
        """Process raw agent response into structured result."""
        ...


class StorageHandler(Protocol):
    """Interface for storing analysis results."""
    
    async def store_result(
        self,
        result: ComplianceAnalysisResult,
        request: ComplianceAnalysisRequest
    ) -> Optional[str]:
        """Store analysis result and return storage location."""
        ...


# ============================================================================
# IMPLEMENTATIONS
# ============================================================================

class BedrockAgentInvoker:
    """Handles AWS Bedrock agent invocations with retry logic."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._last_request_time = 0
        self._min_request_interval = settings.POLICY_COMPLIANCE_MIN_INTERVAL_SECS
        
    async def invoke_agent(
        self,
        config: AgentConfig,
        session_id: str,
        input_data: Dict[str, Any],
        knowledge_bases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Invoke agent with exponential backoff retry."""
        from core.aws_client import get_bedrock_agent_client
        from botocore.exceptions import ClientError, EventStreamError
        import random
        
        client = get_bedrock_agent_client()
        
        # Configure session state with knowledge bases
        session_state = {
            "knowledgeBaseConfigurations": knowledge_bases
        }
        
        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                await self._apply_rate_limit()
                
                logger.info(
                    f"Invoking agent {config.name} (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                
                response = client.invoke_agent(
                    agentId=config.agent_id,
                    agentAliasId=config.alias_id,
                    sessionId=session_id,
                    inputText=json.dumps(input_data),
                    sessionState=session_state
                )
                
                return self._extract_completion(response)
                
            except (ClientError, EventStreamError) as e:
                if self._is_throttling_error(e) and attempt < self.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"Throttling error, retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
                    continue
                raise
                
        raise Exception("Maximum retries exceeded")
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        import random
        base = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, 1)
        return base + jitter
    
    def _is_throttling_error(self, error: Exception) -> bool:
        """Check if error is a throttling exception."""
        error_str = str(error).lower()
        return any(term in error_str for term in ['throttling', 'rate'])
    
    def _extract_completion(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract completion from streaming response."""
        completion = ""
        completion_stream = response.get('completion')
        
        if not completion_stream:
            raise ValueError("No completion stream in response")
        
        for event in completion_stream:
            if 'chunk' in event and 'bytes' in event['chunk']:
                chunk_text = event['chunk']['bytes'].decode('utf-8')
                completion += chunk_text
        
        if not completion.strip():
            raise ValueError("Empty completion from agent")
        
        try:
            return json.loads(completion)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse agent response: {e}")
            raise ValueError(f"Invalid JSON response from agent: {e}")


class ComplianceResponseProcessor:
    """Processes agent responses into structured results."""
    
    def process_response(
        self,
        raw_response: Dict[str, Any],
        request: ComplianceAnalysisRequest
    ) -> ComplianceAnalysisResult:
        """Transform raw agent response to structured result."""
        
        # Extract fields with defaults
        affected_policies = raw_response.get('affectedPolicies', [])
        
        # Calculate scores and grades
        scan_score = raw_response.get('scanScore', 0.0)
        scan_grade = self._calculate_grade(scan_score)
        
        # Determine status
        scan_status = "Completed" if affected_policies is not None else "Failed"
        compliance_status = "NEEDS_UPDATE" if len(affected_policies) > 0 else "OK"
        
        return ComplianceAnalysisResult(
            facility_id=request.facility_id,
            file_name=request.file_name,
            scan_type=request.scan_type,
            affected_policies=affected_policies,
            overall_summary=raw_response.get('overallSummary', ''),
            scan_score=scan_score,
            scan_grade=scan_grade,
            scan_status=scan_status,
            compliance_status=compliance_status,
            s3_file_id=request.s3_file_id,
            user_id=request.user_id,
            aws_response_id=raw_response.get('awsResponseId'),
            diagnostics=self._build_diagnostics(raw_response)
        )
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _build_diagnostics(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Build diagnostics from response."""
        return {
            "workflow_type": "multi_agent_collaboration",
            "supervisor_agent_id": settings.POLICY_COMPLIANCE_SUPERVISOR_AGENT_ID,
            "sections_analyzed": len(response.get('affectedPolicies', [])),
            "processing_timestamp": datetime.now(timezone.utc).isoformat()
        }


class S3StorageHandler:
    """Handles S3 storage operations."""
    
    async def store_result(
        self,
        result: ComplianceAnalysisResult,
        request: ComplianceAnalysisRequest
    ) -> Optional[str]:
        """Store result in S3 and return location."""
        try:
            from core.aws_client import get_s3_client
            
            s3_client = get_s3_client()
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            
            s3_key = (
                f"policy_analysis/{request.facility_id}/"
                f"{request.file_name}/{timestamp}.json"
            )
            
            s3_client.put_object(
                Bucket=settings.S3_ANALYTICS_BUCKET,
                Key=s3_key,
                Body=json.dumps(result.dict(), indent=2),
                ContentType='application/json'
            )
            
            s3_uri = f"s3://{settings.S3_ANALYTICS_BUCKET}/{s3_key}"
            logger.info(f"Stored analysis result: {s3_uri}")
            
            return s3_uri
            
        except Exception as e:
            logger.error(f"Failed to store result in S3: {e}")
            return None


# ============================================================================
# MAIN SERVICE
# ============================================================================

class PolicyComplianceService:
    """
    Orchestrates policy compliance analysis using multi-agent collaboration.
    
    This service follows SOLID principles:
    - Single Responsibility: Orchestrates the analysis workflow
    - Open/Closed: Easy to extend with new processors/storage backends
    - Dependency Inversion: Depends on abstractions, not concrete implementations
    """
    
    def __init__(
        self,
        agent_invoker: Optional[AgentInvoker] = None,
        response_processor: Optional[ResponseProcessor] = None,
        storage_handler: Optional[StorageHandler] = None
    ):
        """Initialize with dependency injection."""
        self.agent_invoker = agent_invoker or BedrockAgentInvoker()
        self.response_processor = response_processor or ComplianceResponseProcessor()
        self.storage_handler = storage_handler or S3StorageHandler()
        
        # Load agent configuration
        self.supervisor_config = AgentConfig(
            agent_id=settings.POLICY_COMPLIANCE_SUPERVISOR_AGENT_ID,
            alias_id=settings.POLICY_COMPLIANCE_SUPERVISOR_AGENT_ALIAS_ID,
            name="PolicyComplianceSupervisor",
            role="Coordinator"
        )
        
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate required configuration."""
        if not self.supervisor_config.agent_id or not self.supervisor_config.alias_id:
            raise ValueError(
                "Supervisor agent not configured. Please update settings with "
                "POLICY_COMPLIANCE_SUPERVISOR_AGENT_ID and "
                "POLICY_COMPLIANCE_SUPERVISOR_AGENT_ALIAS_ID"
            )
        
        logger.info(f"PolicyComplianceService initialized with {self.supervisor_config.name}")
    
    async def analyze_policy_compliance(
        self,
        facility_id: str,
        file_name: str,
        scan_type: str = "manual",
        program_selected: str = "Gen",
        edition_selected: str = "25-1",
        s3_file_id: str = "",
        user_id: str = "",
        defaults: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze policy compliance using multi-agent collaboration.
        
        Returns console-style JSON for compatibility with existing systems.
        """
        start_time = time.time()
            
        # Create request object
        request = ComplianceAnalysisRequest(
            facility_id=facility_id,
            file_name=file_name,
            scan_type=scan_type,
            program_selected=program_selected,
            edition_selected=edition_selected,
            s3_file_id=s3_file_id,
            user_id=user_id
        )
        
        session_id = request.generate_session_id()
        
        logger.info(
            f"Starting compliance analysis: "
            f"facility={facility_id}, file={file_name}, session={session_id}"
        )
        
        try:
            # Prepare input for supervisor agent
            supervisor_input = self._prepare_supervisor_input(request, defaults)
            
            # Configure knowledge bases
            knowledge_bases = self._configure_knowledge_bases()
            
            # Invoke supervisor agent
            raw_response = await self.agent_invoker.invoke_agent(
                config=self.supervisor_config,
                session_id=session_id,
                input_data=supervisor_input,
                knowledge_bases=knowledge_bases
            )
            
            # Process response
            result = self.response_processor.process_response(raw_response, request)
            
            # Store result
            storage_location = await self.storage_handler.store_result(result, request)
            result.storage_location = storage_location
            
            # Add processing metadata
            processing_time = time.time() - start_time
            result.diagnostics.update({
                "session_id": session_id,
                "processing_time_seconds": round(processing_time, 2),
            })
            
            logger.info(
                f"Analysis completed in {processing_time:.2f}s: "
                f"score={result.scan_score}, grade={result.scan_grade}"
            )
            
            # Return as dictionary for JSON response
            return result.dict()
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            
            # Return error response in expected format
            return self._create_error_response(request, str(e))
    
    def _prepare_supervisor_input(
        self, 
        request: ComplianceAnalysisRequest,
        defaults: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare input data for supervisor agent."""
        supervisor_input = {
            "facilityId": request.facility_id,
            "fileName": request.file_name,
            "scanType": request.scan_type,
            "program_selected": request.program_selected,
            "edition_selected": request.edition_selected,
            "s3FileId": request.s3_file_id,
            "userId": request.user_id
        }
        
        if defaults:
            supervisor_input["defaults"] = defaults
        
        return supervisor_input
    
    def _configure_knowledge_bases(self) -> List[Dict[str, Any]]:
        """Configure knowledge base settings."""
        return [
                        {
                            "knowledgeBaseId": settings.POLICIES_KB_ID,
                            "retrievalConfiguration": {
                                "vectorSearchConfiguration": {
                                    "numberOfResults": 10,
                                    "overrideSearchType": "SEMANTIC"
                                }
                            }
                        },
                        {
                            "knowledgeBaseId": settings.NIAHO_STANDARDS_KB_ID,
                            "retrievalConfiguration": {
                                "vectorSearchConfiguration": {
                                    "numberOfResults": 30,
                                    "overrideSearchType": "SEMANTIC"
                                }
                            }
                        }
                    ]
    
    def _create_error_response(
        self, 
        request: ComplianceAnalysisRequest,
        error_message: str
    ) -> Dict[str, Any]:
        """Create error response in expected format."""
        return {
            "facilityId": request.facility_id,
            "fileName": request.file_name,
            "scanType": request.scan_type,
                "affectedPolicies": [],
            "overallSummary": f"Analysis failed: {error_message}",
                "scanScore": 0.0,
                "scanGrade": "F",
                "scanStatus": "Failed",
                "complianceStatus": "UNKNOWN",
            "s3FileId": request.s3_file_id,
            "userId": request.user_id,
            "awsResponseId": None,
            "diagnostics": {
                "workflow_type": "multi_agent_collaboration",
                "error": error_message
            }
        }


# ============================================================================
# SERVICE INSTANCE
# ============================================================================

# Initialize the service for use in the application
policy_compliance_service = PolicyComplianceService()