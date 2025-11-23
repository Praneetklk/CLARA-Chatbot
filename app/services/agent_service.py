# services/agent_service.py
import boto3
import json
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from core.config import settings
from core.logger import logger
from models.bedrock_models import BedrockModels
from schemas.request_models import AgentRunRequest, AgentRunResponse, AgentRunOverrides


class AgentService:
    """
    Service for orchestrating TwoPhaseComplianceAgent runs
    Handles policy discovery, per-policy invocation, matrix completion, and S3 writes
    """
    
    # S3 Vector Store limits
    MIN_TOP_K = 1
    MAX_TOP_K = 30
    
    def __init__(self):
        self.bedrock_models = BedrockModels()
        from core.aws_client import get_s3_client
        self.s3_client = get_s3_client()
        self.agent_id = settings.BEDROCK_AGENT_ID
        self.agent_alias_id = settings.BEDROCK_AGENT_ALIAS_ID
        
    def _clamp_top_k(self, value: int, name: str, diagnostics_notes: List[str]) -> int:
        """
        Clamp Top-K value to S3 Vector Store limits (1-30) and log warnings
        """
        if value < self.MIN_TOP_K:
            clamped = self.MIN_TOP_K
            warning = f"Clamped {name} {value} → {clamped} (minimum: {self.MIN_TOP_K})"
            logger.warning(warning)
            diagnostics_notes.append(warning)
            return clamped
        elif value > self.MAX_TOP_K:
            clamped = self.MAX_TOP_K
            warning = f"Clamped {name} {value} → {clamped} (S3 Vector Store limit: {self.MAX_TOP_K})"
            logger.warning(warning)
            diagnostics_notes.append(warning)
            return clamped
        else:
            return value
        
    async def run_agent(self, request: AgentRunRequest) -> AgentRunResponse:
        """
        Main orchestration method for TwoPhaseComplianceAgent runs
        """
        run_id = self._generate_run_id()
        session_id = f"session_{run_id}"
        
        logger.info(f"Starting agent run {run_id} for revision {request.revision_id}, org {request.org_id}")
        
        try:
            # Step 1: Policy auto-discovery if needed
            policies = self._resolve_policies(request.org_id, request.policies)
            
            # Step 2: Run agent per policy
            canonical_changes = None
            all_results = []
            enumeration_stats = None
            # Initialize diagnostics with clamping tracking
            retrieval_notes = ["Agent does URI-based filtering per console mode instructions", "Server-side Top-K limits applied"]
            
            # Pre-calculate clamped values for diagnostics
            raw_change_kb_top_k = request.overrides.change_kb_top_k if request.overrides else settings.CHANGE_KB_TOP_K
            raw_policies_kb_top_k = request.overrides.policies_kb_top_k if request.overrides else settings.POLICIES_KB_TOP_K
            
            effective_change_kb_top_k = self._clamp_top_k(raw_change_kb_top_k, "change_kb_top_k", retrieval_notes)
            effective_policies_kb_top_k = self._clamp_top_k(raw_policies_kb_top_k, "policies_kb_top_k", retrieval_notes)
            
            run_diagnostics = {
                "enumeration": {},
                "policy_auto_discovery": {
                    "attempted": request.policies is None or len(request.policies) == 0,
                    "discovered_policy_ids": policies,
                    "notes": []
                },
                "retrieval": {
                    "policies_kb_top_k": effective_policies_kb_top_k,
                    "change_kb_top_k": effective_change_kb_top_k,
                    "effective_policies_kb_top_k": effective_policies_kb_top_k,
                    "effective_change_kb_top_k": effective_change_kb_top_k,
                    "filters_applied": ["URI-based filtering by agent"],
                    "notes": retrieval_notes
                },
                "warnings": [],
                "errors": []
            }
            
            # Step 2: Mandatory Phase-0 enumeration first
            logger.info("Running mandatory Phase-0 enumeration")
            try:
                phase0_result = self._run_phase0_enumeration(
                    session_id=session_id,
                    revision_id=request.revision_id,
                    org_id=request.org_id,
                    overrides=request.overrides,
                    diagnostics_notes=retrieval_notes
                )
                
                if phase0_result.get("changes_enumerated"):
                    canonical_changes = phase0_result.get("changes_enumerated", [])
                    enumeration_stats = phase0_result.get("enumeration_stats", {})
                    run_diagnostics["enumeration"] = phase0_result.get("run_diagnostics", {}).get("enumeration", {})
                    logger.info(f"Phase-0 enumeration successful: {len(canonical_changes)} changes")
                else:
                    logger.error("Phase-0 enumeration returned no changes")
                    canonical_changes = []
                    enumeration_stats = {"total_changes": 0, "counts_by_standard": []}
                    
            except Exception as e:
                error_msg = f"Phase-0 enumeration failed: {str(e)}"
                logger.error(error_msg)
                run_diagnostics["errors"].append(error_msg)
                canonical_changes = []
                enumeration_stats = {"total_changes": 0, "counts_by_standard": []}
            
            # Step 3: Process policies in batches using canonical changes
            for i, policy_id in enumerate(policies):
                logger.info(f"Processing policy {i+1}/{len(policies)}: {policy_id}")
                
                try:
                    policy_results = self._run_policy_batched(
                        session_id=session_id,
                        revision_id=request.revision_id,
                        org_id=request.org_id,
                        policy_id=policy_id,
                        canonical_changes=canonical_changes,
                        overrides=request.overrides,
                        diagnostics_notes=retrieval_notes
                    )
                    
                    all_results.extend(policy_results)
                    
                except Exception as e:
                    error_msg = f"Failed to process policy {policy_id}: {str(e)}"
                    logger.error(error_msg)
                    run_diagnostics["errors"].append(error_msg)
            
            # Step 4: Ensure matrix completeness
            complete_results, backfill_count = self._ensure_matrix_completeness(
                policies=policies,
                changes_enumerated=canonical_changes,
                results=all_results
            )
            
            if backfill_count > 0:
                run_diagnostics["warnings"].append(f"Backfilled {backfill_count} missing policy-change pairs")
            
            # Step 5: Rebuild summaries
            summaries = self._rebuild_summaries(policies, complete_results)
            
            # Step 6: Build final agent JSON
            agent_json = {
                "revision_id": request.revision_id,
                "org_id": request.org_id,
                "policies": policies,
                "enumeration_stats": enumeration_stats or {"total_changes": 0, "counts_by_standard": []},
                "changes_enumerated": canonical_changes or [],
                "results": complete_results,
                "summaries": summaries,
                "run_diagnostics": run_diagnostics
            }
            
            # Step 7: Write to S3 if requested
            s3_key = None
            if request.write_to_s3:
                s3_key = self._write_to_s3(run_id, request.revision_id, agent_json)
                
            logger.info(f"Agent run {run_id} completed successfully with {len(complete_results)} results")
            
            # Create summary instead of returning full agent_json
            summary = {
                "total_changes": len(agent_json.get("changes_enumerated", [])),
                "total_results": len(agent_json.get("results", [])),
                "enumeration_stats": agent_json.get("enumeration_stats", {}),
                "summaries": agent_json.get("summaries", {}),
                "run_diagnostics": agent_json.get("run_diagnostics", {})
            }
            
            return AgentRunResponse(
                run_id=run_id,
                revision_id=request.revision_id,
                org_id=request.org_id,
                policies=policies,
                s3_key=s3_key,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Agent run {run_id} failed: {str(e)}")
            raise
    
    def _resolve_policies(self, org_id: str, policies: Optional[List[str]]) -> List[str]:
        """
        Resolve policy list: use provided policies or auto-discover from S3
        """
        if policies and len(policies) > 0:
            logger.info(f"Using provided policies: {policies}")
            return policies
            
        # Auto-discover policies from S3
        logger.info(f"Auto-discovering policies for org {org_id}")
        try:
            # Use the structured data bucket where policies actually live
            bucket_name = settings.STRUCTURED_DATA_BUCKET
            prefix = f"hospitals/{org_id}/kb/policies/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                Delimiter="/"
            )
            
            discovered_policies = []
            for prefix_info in response.get("CommonPrefixes", []):
                # Extract policy ID from prefix like "hospitals/org/kb/policies/policy_id/"
                policy_path = prefix_info["Prefix"]
                policy_id = policy_path.rstrip("/").split("/")[-1]
                if policy_id:  # Skip empty policy IDs
                    discovered_policies.append(policy_id)
            
            logger.info(f"Discovered {len(discovered_policies)} policies: {discovered_policies}")
            return discovered_policies
            
        except Exception as e:
            logger.error(f"Policy auto-discovery failed: {str(e)}")
            raise
    
    def _run_phase0_enumeration(
        self,
        session_id: str,
        revision_id: str,
        org_id: str,
        overrides: Optional[AgentRunOverrides],
        diagnostics_notes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run dedicated Phase-0 enumeration with adaptive sizing to prevent truncation
        """
        if diagnostics_notes is None:
            diagnostics_notes = []
            
        # Build session state with only Change KB, using reduced Top-K for enumeration
        raw_change_kb_top_k = overrides.change_kb_top_k if overrides else settings.CHANGE_KB_TOP_K
        
        # Use smaller Top-K for enumeration to prevent large responses
        enumeration_top_k = min(raw_change_kb_top_k, 15)  # Limit to 15 for enumeration
        change_kb_top_k = self._clamp_top_k(enumeration_top_k, "enumeration_change_kb_top_k", diagnostics_notes)
        change_kb_search_type = overrides.change_kb_search_type if overrides else settings.CHANGE_KB_SEARCH_TYPE
        
        session_state = {
            "knowledgeBaseConfigurations": [
                {
                    "knowledgeBaseId": settings.KB_CHANGE_HISTORY_ID,
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": change_kb_top_k,
                            "overrideSearchType": change_kb_search_type
                        }
                    }
                }
            ]
        }
        
        # Phase-0 input with compact mode enabled for enumeration
        input_text = json.dumps({
            "revision_id": revision_id,
            "org_id": org_id,
            "policies": [],  # Empty array triggers enumeration-only mode
            "compact": True,  # Always use compact for enumeration
            "max_evidence_chars": 100  # Very short evidence for enumeration
        })
        
        logger.info(f"Running Phase-0 enumeration with Top-K={change_kb_top_k}, compact=True")
        
        try:
            response = self.bedrock_models.invoke_agent(
                agent_id=self.agent_id,
                agent_alias_id=self.agent_alias_id,
                session_id=session_id,
                input_text=input_text,
                session_state=session_state,
                enable_trace=True
            )
            
            agent_response = response.get("completion", {})
            
            # Validate response structure
            if not isinstance(agent_response, dict):
                raise ValueError(f"Agent returned invalid response format: {type(agent_response)}")
            
            # Log enumeration success
            changes_count = len(agent_response.get("changes_enumerated", []))
            logger.info(f"Phase-0 enumeration completed: {changes_count} changes enumerated")
            
            return agent_response
            
        except json.JSONDecodeError as parse_error:
            logger.error(f"Phase-0 enumeration JSON parse failed: {parse_error}")
            # Try with even more aggressive settings
            logger.info("Retrying Phase-0 with ultra-compact settings")
            
            # Ultra-compact retry
            session_state["knowledgeBaseConfigurations"][0]["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"] = 10
            
            input_text = json.dumps({
                "revision_id": revision_id,
                "org_id": org_id,
                "policies": [],
                "compact": True,
                "max_evidence_chars": 50  # Ultra-short evidence
            })
            
            try:
                response = self.bedrock_models.invoke_agent(
                    agent_id=self.agent_id,
                    agent_alias_id=self.agent_alias_id,
                    session_id=session_id,
                    input_text=input_text,
                    session_state=session_state,
                    enable_trace=True
                )
                
                agent_response = response.get("completion", {})
                if not isinstance(agent_response, dict):
                    raise ValueError(f"Agent returned invalid response format: {type(agent_response)}")
                
                changes_count = len(agent_response.get("changes_enumerated", []))
                logger.info(f"Phase-0 enumeration retry succeeded: {changes_count} changes enumerated")
                
                return agent_response
                
            except Exception as retry_error:
                logger.error(f"Phase-0 enumeration retry also failed: {retry_error}")
                raise
            
        except Exception as e:
            logger.error(f"Phase-0 enumeration failed: {str(e)}")
            raise

    def _calculate_adaptive_batch_size(self, policy_id: str, total_changes: int, compact: bool) -> int:
        """
        Calculate optimal batch size based on policy type and characteristics
        """
        # Policy complexity heuristics
        policy_lower = policy_id.lower()
        if any(term in policy_lower for term in ["medical_staff", "clinical", "care", "treatment"]):
            complexity = "high"  # Rich clinical content
        elif any(term in policy_lower for term in ["adm.", "admin", "document", "control", "procedure"]):
            complexity = "low"   # Administrative policies
        else:
            complexity = "medium"
        
        # Adaptive sizing based on complexity and mode
        size_map = {
            "high": {"compact": 2, "full": 1},
            "medium": {"compact": 3, "full": 2}, 
            "low": {"compact": 4, "full": 3}
        }
        
        mode = "compact" if compact else "full"
        optimal_size = size_map[complexity][mode]
        
        # Don't exceed total changes
        return min(optimal_size, total_changes)
    
    def _should_use_compact_proactively(self, policy_id: str, batch_size: int) -> bool:
        """
        Determine if we should start with compact mode proactively
        """
        # Always use compact for larger batches
        if batch_size > 2:
            return True
            
        # Use compact for clinical policies (rich content)
        policy_lower = policy_id.lower()
        if any(term in policy_lower for term in ["medical_staff", "clinical", "care", "treatment"]):
            return True
            
        return False
    
    def _monitor_response_size(self, response_length: int, batch_size: int) -> dict:
        """
        Monitor response size and provide optimization suggestions
        """
        risk_threshold = 5000
        warning_threshold = 4000
        
        if response_length > risk_threshold:
            risk_level = "high"
            suggested_batch_size = 1
            suggest_compact = True
        elif response_length > warning_threshold:
            risk_level = "medium"
            suggested_batch_size = max(1, batch_size // 2)
            suggest_compact = True
        else:
            risk_level = "low"
            suggested_batch_size = batch_size
            suggest_compact = False
            
        return {
            "size_risk": risk_level,
            "response_length": response_length,
            "suggested_batch_size": suggested_batch_size,
            "suggest_compact": suggest_compact
        }
    
    def _run_policy_batched(
        self,
        session_id: str,
        revision_id: str,
        org_id: str,
        policy_id: str,
        canonical_changes: List[Dict],
        overrides: Optional[AgentRunOverrides],
        diagnostics_notes: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Run the agent for a policy using adaptive batched processing of changes
        """
        if not canonical_changes:
            logger.warning(f"No changes to process for policy {policy_id}")
            return []
        
        # Use adaptive batch sizing instead of fixed size
        requested_batch_size = overrides.batch_size if overrides else 5
        requested_compact = overrides.compact if overrides else False
        max_evidence_chars = overrides.max_evidence_chars if overrides else 240
        
        # Calculate optimal batch size
        optimal_batch_size = self._calculate_adaptive_batch_size(
            policy_id, len(canonical_changes), requested_compact
        )
        
        # Use the smaller of requested vs optimal
        batch_size = min(requested_batch_size, optimal_batch_size)
        
        # Determine if compact should be used proactively
        proactive_compact = self._should_use_compact_proactively(policy_id, batch_size)
        compact = requested_compact or proactive_compact
        
        logger.info(f"Adaptive batching for {policy_id}: batch_size={batch_size} (optimal={optimal_batch_size}), compact={compact} (proactive={proactive_compact})")
        
        # Create change_keys from canonical_changes
        change_keys = []
        for change in canonical_changes:
            change_key = {
                "standard_code": change.get("standard_code"),
                "sr_code": change.get("sr_code", "_"),
                "content_type": change.get("content_type")
            }
            change_keys.append(change_key)
        
        # Split into batches
        batches = [change_keys[i:i + batch_size] for i in range(0, len(change_keys), batch_size)]
        logger.info(f"Processing {len(change_keys)} changes in {len(batches)} batches for policy {policy_id}")
        
        all_results = []
        
        for batch_idx, batch_changes in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch_changes)} changes")
            
            try:
                batch_results = self._run_agent_for_policy_batch(
                    session_id=session_id,
                    revision_id=revision_id,
                    org_id=org_id,
                    policy_id=policy_id,
                    change_keys=batch_changes,
                    compact=compact,
                    max_evidence_chars=max_evidence_chars,
                    overrides=overrides,
                    diagnostics_notes=diagnostics_notes
                )
                
                all_results.extend(batch_results)
                
                # Monitor response size for future optimization
                if hasattr(self, '_last_response_length'):
                    size_analysis = self._monitor_response_size(self._last_response_length, len(batch_changes))
                    if size_analysis["size_risk"] != "low":
                        logger.info(f"Response size analysis: {size_analysis}")
                        diagnostics_notes.append(f"Batch {batch_idx + 1} size risk: {size_analysis['size_risk']} ({size_analysis['response_length']} chars)")
                
                logger.info(f"Batch {batch_idx + 1} completed successfully with {len(batch_results)} results")
                
            except json.JSONDecodeError as parse_error:
                logger.warning(f"Batch {batch_idx + 1} JSON parse failed: {parse_error}")
                
                # Enhanced retry logic based on error type
                retry_batch_size = self._calculate_retry_batch_size(str(parse_error), len(batch_changes))
                retry_compact = True
                
                if retry_batch_size > 0 and retry_batch_size < len(batch_changes):
                    logger.info(f"Retrying batch {batch_idx + 1} with size={retry_batch_size}, compact={retry_compact}")
                    
                    try:
                        # Split into optimal retry batches
                        retry_batches = [batch_changes[i:i + retry_batch_size] 
                                       for i in range(0, len(batch_changes), retry_batch_size)]
                        
                        for retry_idx, retry_batch in enumerate(retry_batches):
                            sub_results = self._run_agent_for_policy_batch(
                                session_id=session_id,
                                revision_id=revision_id,
                                org_id=org_id,
                                policy_id=policy_id,
                                change_keys=retry_batch,
                                compact=retry_compact,
                                max_evidence_chars=max_evidence_chars,
                                overrides=overrides,
                                diagnostics_notes=diagnostics_notes
                            )
                            all_results.extend(sub_results)
                            logger.info(f"Retry batch {retry_idx + 1}/{len(retry_batches)} succeeded")
                        
                        logger.info(f"Batch {batch_idx + 1} retry completed with {len(retry_batches)} sub-batches")
                        
                    except Exception as retry_error:
                        logger.error(f"Batch {batch_idx + 1} retry failed: {retry_error}")
                        # Skip this batch - matrix completeness will backfill
                        
                else:
                    logger.error(f"Batch {batch_idx + 1} failed and cannot be retried (size={len(batch_changes)})")
                    # Skip this batch - matrix completeness will backfill
                    
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                # Skip this batch - matrix completeness will backfill
        
        logger.info(f"Policy {policy_id} processing completed: {len(all_results)} total results")
        return all_results
    
    def _calculate_retry_batch_size(self, error_message: str, current_batch_size: int) -> int:
        """
        Calculate optimal retry batch size based on error type
        """
        if "Unterminated string" in error_message:
            # Truncation issue - very aggressive reduction
            return 1
        elif "Expecting property name" in error_message:
            # Structure issue - moderate reduction
            return max(1, current_batch_size // 3)
        elif "Invalid control character" in error_message:
            # Encoding issue - small reduction
            return max(1, current_batch_size // 2)
        else:
            # Unknown error - conservative approach
            return max(1, current_batch_size // 2)

    def _run_agent_for_policy_batch(
        self,
        session_id: str,
        revision_id: str,
        org_id: str,
        policy_id: str,
        change_keys: List[Dict],
        compact: bool,
        max_evidence_chars: int,
        overrides: Optional[AgentRunOverrides],
        diagnostics_notes: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Run the agent for a specific batch of changes for a policy
        """
        # Build session state with knowledge base configurations
        session_state = self._build_session_state(
            revision_id=revision_id,
            org_id=org_id,
            policy_id=policy_id,
            overrides=overrides,
            diagnostics_notes=diagnostics_notes
        )
        
        # Input text for the agent - structured JSON format with batch scope
        input_data = {
            "revision_id": revision_id,
            "org_id": org_id,
            "policies": [policy_id],
            "change_keys": change_keys,
            "compact": compact,
            "max_evidence_chars": max_evidence_chars
        }
        
        input_text = json.dumps(input_data)
        
        try:
            response = self.bedrock_models.invoke_agent(
                agent_id=self.agent_id,
                agent_alias_id=self.agent_alias_id,
                session_id=session_id,
                input_text=input_text,
                session_state=session_state,
                enable_trace=True
            )
            
            agent_response = response.get("completion", {})
            
            # Validate response structure
            if not isinstance(agent_response, dict):
                raise ValueError(f"Agent returned invalid response format: {type(agent_response)}")
            
            # Extract results from the batch response
            results = agent_response.get("results", [])
            
            # Store response length for monitoring (approximate)
            response_length = len(str(agent_response))
            self._last_response_length = response_length
            
            logger.info(f"Batch returned {len(results)} results for {len(change_keys)} changes (response: {response_length} chars)")
            return results
            
        except Exception as e:
            logger.error(f"Agent batch invocation failed for policy {policy_id}: {str(e)}")
            raise
    
    def _build_session_state(
        self,
        revision_id: str,
        org_id: str,
        policy_id: str,
        overrides: Optional[AgentRunOverrides],
        diagnostics_notes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build session state with KB configurations and filters
        """
        if diagnostics_notes is None:
            diagnostics_notes = []
            
        # Apply overrides or use defaults, then clamp to S3 Vector Store limits
        raw_change_kb_top_k = overrides.change_kb_top_k if overrides else settings.CHANGE_KB_TOP_K
        raw_policies_kb_top_k = overrides.policies_kb_top_k if overrides else settings.POLICIES_KB_TOP_K
        
        # Clamp Top-K values and record any adjustments
        change_kb_top_k = self._clamp_top_k(raw_change_kb_top_k, "change_kb_top_k", diagnostics_notes)
        policies_kb_top_k = self._clamp_top_k(raw_policies_kb_top_k, "policies_kb_top_k", diagnostics_notes)
        
        change_kb_search_type = overrides.change_kb_search_type if overrides else settings.CHANGE_KB_SEARCH_TYPE
        policies_kb_search_type = overrides.policies_kb_search_type if overrides else settings.POLICIES_KB_SEARCH_TYPE
        
        return {
            "knowledgeBaseConfigurations": [
                {
                    "knowledgeBaseId": settings.KB_CHANGE_HISTORY_ID,
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": change_kb_top_k,
                            "overrideSearchType": change_kb_search_type
                        }
                    }
                },
                {
                    "knowledgeBaseId": settings.KB_POLICIES_ID,
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": policies_kb_top_k,
                            "overrideSearchType": policies_kb_search_type
                        }
                    }
                }
            ]
        }
    
    def _ensure_matrix_completeness(
        self,
        policies: List[str],
        changes_enumerated: List[Dict],
        results: List[Dict]
    ) -> tuple[List[Dict], int]:
        """
        Ensure complete matrix: policies × changes_enumerated
        Backfill missing pairs with NO_TRULY_RELEVANT_SECTION + SKIPPED
        """
        if not changes_enumerated:
            logger.warning("No changes enumerated, returning empty results")
            return [], 0
        
        # Create a set of existing (policy_id, standard_code, sr_code) combinations
        existing_combinations = set()
        for result in results:
            key = (result.get("policy_id"), result.get("standard_code"), result.get("sr_code"))
            existing_combinations.add(key)
        
        # Generate expected combinations
        expected_combinations = set()
        for policy_id in policies:
            for change in changes_enumerated:
                key = (policy_id, change.get("standard_code"), change.get("sr_code"))
                expected_combinations.add(key)
        
        # Find missing combinations
        missing_combinations = expected_combinations - existing_combinations
        backfill_count = len(missing_combinations)
        
        if backfill_count > 0:
            logger.info(f"Backfilling {backfill_count} missing policy-change pairs")
        
        # Create backfill results
        complete_results = list(results)  # Copy existing results
        
        for policy_id, standard_code, sr_code in missing_combinations:
            # Find the corresponding change details
            change_details = next(
                (change for change in changes_enumerated 
                 if change.get("standard_code") == standard_code and change.get("sr_code") == sr_code),
                None
            )
            
            if change_details:
                backfill_result = {
                    "policy_id": policy_id,
                    "standard_code": change_details.get("standard_code"),
                    "sr_code": change_details.get("sr_code"),
                    "content_type": change_details.get("content_type"),
                    "discovery": {
                        "status": "NO_TRULY_RELEVANT_SECTION",
                        "section_id": "",
                        "evidence": "",
                        "nearest_nonmatch": {
                            "candidate_section_id": "",
                            "why_not_relevant": [],
                            "snippet": ""
                        }
                    },
                    "compliance": {
                        "status": "SKIPPED",
                        "evidence_policy": "",
                        "evidence_change": "",
                        "gaps": [],
                        "recommendations": [],
                        "confidence": 0.0
                    },
                    "change_source_key": change_details.get("change_source_key", ""),
                    "policy_source_key": ""
                }
                complete_results.append(backfill_result)
        
        return complete_results, backfill_count
    
    def _rebuild_summaries(self, policies: List[str], results: List[Dict]) -> Dict[str, Any]:
        """
        Rebuild consistent summaries from the complete results matrix
        """
        # Per-policy summaries
        per_policy = []
        for policy_id in policies:
            policy_results = [r for r in results if r.get("policy_id") == policy_id]
            
            totals = {
                "changes": len(policy_results),
                "truly_relevant": len([r for r in policy_results if r.get("discovery", {}).get("status") == "TRULY_RELEVANT"]),
                "no_truly_relevant": len([r for r in policy_results if r.get("discovery", {}).get("status") == "NO_TRULY_RELEVANT_SECTION"]),
                "compliant": len([r for r in policy_results if r.get("compliance", {}).get("status") == "COMPLIANT"]),
                "needs_update": len([r for r in policy_results if r.get("compliance", {}).get("status") == "NEEDS_UPDATE"]),
                "not_comparable": len([r for r in policy_results if r.get("compliance", {}).get("status") == "NOT_COMPARABLE"])
            }
            
            per_policy.append({
                "policy_id": policy_id,
                "totals": totals
            })
        
        # Per-standard-code summaries
        standard_groups = {}
        for result in results:
            standard_code = result.get("standard_code")
            if standard_code not in standard_groups:
                standard_groups[standard_code] = {
                    "changes": 0,
                    "matched_policies": set()
                }
            
            standard_groups[standard_code]["changes"] += 1
            policy_id = result.get("policy_id")
            if policy_id:
                standard_groups[standard_code]["matched_policies"].add(policy_id)
        
        per_standard_code = []
        for standard_code, data in standard_groups.items():
            per_standard_code.append({
                "standard_code": standard_code,
                "changes": data["changes"],
                "matched_policies": list(data["matched_policies"])
            })
        
        return {
            "per_policy": per_policy,
            "per_standard_code": per_standard_code
        }
    
    def _write_to_s3(self, run_id: str, revision_id: str, agent_json: Dict[str, Any]) -> str:
        """
        Write the merged JSON to S3 and return the S3 key
        """
        bucket = settings.AGENT_RUNS_BUCKET
        key = f"{settings.AGENT_RUNS_PREFIX}/{run_id}/rev-{revision_id}/results.json"
        
        try:
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(agent_json, indent=2),
                ContentType="application/json"
            )
            
            s3_uri = f"s3://{bucket}/{key}"
            logger.info(f"Agent results written to {s3_uri}")
            return s3_uri
            
        except Exception as e:
            logger.error(f"Failed to write results to S3: {str(e)}")
            raise
    
    def _generate_run_id(self) -> str:
        """
        Generate a unique run ID with timestamp and hash
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{random_suffix}"
