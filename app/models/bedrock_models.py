# models/bedrock_models.py 
from typing import Dict, List, Optional, Any, Tuple
from core.config import settings
from core.logger import logger
from core.aws_client import get_bedrock_runtime_client, get_bedrock_agent_client, validate_aws_credentials
from schemas.request_models import CountMethod
import json

validate_aws_credentials()

bedrock_runtime_client = get_bedrock_runtime_client()
bedrock_agent_client = get_bedrock_agent_client()
logger.info("Bedrock clients initialized")


class BedrockModels:
    """
    Handles all Bedrock model operations including retrieval and generation
    ENHANCED: Now supports OpenSearch Serverless with HYBRID search and metadata filtering
    """
    
    def __init__(self):
        self.runtime_client = bedrock_runtime_client
        self.agent_client = bedrock_agent_client
    
    def retrieve_from_kb(
        self, 
        query: str, 
        kb_id: str = None,
        top_k: int = None,
        search_type: str = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Retrieve from Bedrock Knowledge Base with OpenSearch Serverless
        
        ENHANCED: Now supports HYBRID search and metadata filtering
        
        Args:
            query: Search query text
            kb_id: Knowledge base ID (defaults to OpenSearch KB)
            top_k: Number of results to retrieve
            search_type: "SEMANTIC" or "HYBRID" (defaults to HYBRID for OpenSearch)
            metadata_filter: Structured metadata filter (e.g., {"program": "NIAHO-HOSP"})
            
        Returns:
            AWS Bedrock retrieve response with retrievalResults
        """
        kb_id = kb_id or settings.OPENSEARCH_KB_ID
        top_k = top_k or settings.KB_RETRIEVE_TOP_K
        search_type = search_type or settings.KB_SEARCH_TYPE
        
        """
        Build retrieval configuration for OpenSearch
        HYBRID search combines vector similarity with keyword matching (BM25)
        This provides better results than pure semantic search for specific terms
        """
        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": top_k,
                "overrideSearchType": search_type
            }
        }
        
        """
        Add metadata filter if provided
        Filters are applied before vector/keyword search to reduce search space
        This ensures we only search within the specified program/edition
        """
        if metadata_filter:
            bedrock_filter = self._build_bedrock_filter(metadata_filter)
            if bedrock_filter:
                retrieval_config["vectorSearchConfiguration"]["filter"] = bedrock_filter
                logger.info(
                    f"Applying metadata filter",
                    extra={
                        "kb_id": kb_id,
                        "filter": metadata_filter,
                        "search_type": search_type
                    }
                )
        
        try:
            response = self.agent_client.retrieve(
                knowledgeBaseId=kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration=retrieval_config
            )
            
            num_results = len(response.get("retrievalResults", []))
            logger.info(
                f"Retrieved {num_results} documents from KB",
                extra={
                    "kb_id": kb_id,
                    "search_type": search_type,
                    "top_k": top_k,
                    "has_filter": metadata_filter is not None
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"KB Retrieval failed: {str(e)}")
            raise
    
    def _build_bedrock_filter(self, metadata_filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build AWS Bedrock filter format from structured metadata filter
        
        OpenSearch filters use the equals operator for exact matching
        Multiple filters are combined with AND logic
        
        Filter format:
        {
            "andAll": [
                {"equals": {"key": "program", "value": "NIAHO-HOSP"}},
                {"equals": {"key": "edition", "value": "25-0"}}
            ]
        }
        
        Args:
            metadata_filter: Structured filter dict from MetadataFilter
            
        Returns:
            Bedrock-formatted filter dict, or None if no valid filters
        """
        filter_conditions = []
        
        """
        Program filter (most important - isolates program documents)
        """
        if metadata_filter.get("program"):
            filter_conditions.append({
                "equals": {
                    "key": "program",
                    "value": metadata_filter["program"]
                }
            })
        
        """
        Edition filter (optional - we only have latest editions)
        Included for future compatibility if multiple editions exist
        """
        if metadata_filter.get("edition"):
            filter_conditions.append({
                "equals": {
                    "key": "edition",
                    "value": metadata_filter["edition"]
                }
            })
        
        """
        Chapter code filter (optional - strict filtering by chapter)
        Currently not used as we let hybrid search handle chapter matching
        Kept for future enhancement if needed
        """
        if metadata_filter.get("chapter_code"):
            filter_conditions.append({
                "equals": {
                    "key": "chapter_code",
                    "value": metadata_filter["chapter_code"]
                }
            })
        
        """
        Domain code filter (optional - filter by domain like CC, QM, IC)
        """
        if metadata_filter.get("domain_code"):
            filter_conditions.append({
                "equals": {
                    "key": "domain_code",
                    "value": metadata_filter["domain_code"]
                }
            })
        
        if not filter_conditions:
            return None
        
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        
        return {"andAll": filter_conditions}
    
    def invoke_agent(
        self,
        agent_id: str,
        agent_alias_id: str,
        session_id: str,
        input_text: str,
        session_state: Dict = None,
        enable_trace: bool = True
    ) -> Dict[str, Any]:
        """
        Invoke Bedrock Agent Runtime with knowledge base configurations
        UNCHANGED - Agent invocation logic remains the same
        """
        try:
            invoke_params = {
                "agentId": agent_id,
                "agentAliasId": agent_alias_id,
                "sessionId": session_id,
                "inputText": input_text,
                "enableTrace": enable_trace
            }
            
            if session_state:
                invoke_params["sessionState"] = session_state
            
            logger.info(f"Invoking agent {agent_id} with session {session_id}")
            logger.info(f"Input text: {input_text}")
            logger.info(f"Session state KB count: {len(session_state.get('knowledgeBaseConfigurations', [])) if session_state else 0}")
            logger.debug(f"Full session state: {json.dumps(session_state, indent=2) if session_state else 'None'}")
            
            response = self.agent_client.invoke_agent(**invoke_params)
            
            completion = ""
            trace_data = {}
            
            if 'completion' in response:
                event_stream = response['completion']
                logger.debug(f"EventStream type: {type(event_stream)}")
                
                try:
                    chunk_count = 0
                    total_events = 0
                    got_final_chunk = False
                    for event in event_stream:
                        total_events += 1
                        event_keys = list(event.keys()) if isinstance(event, dict) else []
                        logger.info(f"Event #{total_events}: {event_keys}")
                        
                        if 'chunk' in event:
                            chunk = event['chunk']
                            chunk_count += 1
                            
                            if 'bytes' in chunk:
                                chunk_text = chunk['bytes'].decode('utf-8')
                                completion += chunk_text
                                got_final_chunk = True
                                logger.info(f"📝 Chunk {chunk_count} ({len(chunk_text)} chars, total: {len(completion)}): {chunk_text[:50]}...")
                            elif 'attribution' in chunk:
                                logger.debug(f"Chunk {chunk_count}: Attribution chunk received")
                            else:
                                logger.debug(f"Chunk {chunk_count}: Unknown chunk type: {list(chunk.keys())}")
                        elif 'messageComplete' in event:
                            logger.info("🔚 Message completion event received")
                            logger.info(f"Current completion length: {len(completion)} chars")
                        elif 'metadata' in event:
                            metadata = event.get('metadata', {})
                            logger.info(f"📋 Metadata event: {metadata}")
                        elif 'returnControl' in event:
                            logger.info("🔄 Return control event received")
                        elif 'error' in event:
                            error_details = event.get('error', {})
                            logger.error(f"❌ Error event: {error_details}")
                            raise Exception(f"Agent error: {error_details}")
                        elif 'trace' in event:
                            trace_event = event.get('trace', {})
                            if 'trace' in trace_event:
                                trace_details = trace_event['trace']
                                trace_data.update(trace_details)
                                
                                if 'orchestrationTrace' in trace_details:
                                    orch_trace = trace_details['orchestrationTrace']
                                    if 'modelInvocationInput' in orch_trace:
                                        logger.info("🔍 Agent model invocation started")
                                    if 'modelInvocationOutput' in orch_trace:
                                        logger.info("✅ Agent model invocation completed")
                                    if 'rationale' in orch_trace:
                                        rationale = orch_trace['rationale']['text'][:200]
                                        logger.info(f"💭 Agent rationale: {rationale}...")
                                
                                if 'knowledgeBaseLookupInput' in trace_details:
                                    kb_input = trace_details['knowledgeBaseLookupInput']
                                    kb_id = kb_input.get('knowledgeBaseId', 'unknown')
                                    query = kb_input.get('text', '')[:100]
                                    logger.info(f"🔍 KB Lookup ({kb_id}): {query}...")
                                
                                if 'knowledgeBaseLookupOutput' in trace_details:
                                    kb_output = trace_details['knowledgeBaseLookupOutput']
                                    results = kb_output.get('retrievedReferences', [])
                                    logger.info(f"📚 KB Results: {len(results)} references found")
                                    
                                    for i, ref in enumerate(results[:2]):
                                        location = ref.get('location', {}).get('s3Location', {})
                                        uri = location.get('uri', 'unknown')
                                        logger.info(f"   📄 Result {i+1}: {uri}")
                            
                            logger.debug("Captured trace data")
                        else:
                            logger.debug(f"Unknown event type: {event}")
                            
                except Exception as stream_error:
                    logger.error(f"Error processing event stream: {stream_error}")
                    logger.error(f"Event stream type: {type(event_stream)}")
                    logger.error(f"Completion so far ({len(completion)} chars): {completion[:1000]}")
                    raise
                    
                logger.info(f"Finished processing {total_events} total events, {chunk_count} chunks from event stream, got_final_chunk={got_final_chunk}")
            else:
                logger.warning("No 'completion' key in response")
                logger.debug(f"Response keys: {list(response.keys())}")
            
            logger.info(f"Total completion length: {len(completion)} characters")
            
            if completion:
                sample_completion = completion[:500] + "..." if len(completion) > 500 else completion
                logger.info(f"Agent completion sample: {sample_completion}")
                
                open_braces = completion.count('{')
                close_braces = completion.count('}')
                open_brackets = completion.count('[')
                close_brackets = completion.count(']')
                logger.info(f"JSON structure check - Braces: {open_braces}/{close_braces}, Brackets: {open_brackets}/{close_brackets}")
            else:
                logger.warning("Empty completion received from agent")
            
            try:
                agent_response = json.loads(completion) if completion else {}
                logger.info(f"Agent response parsed successfully, type: {type(agent_response)}")
                
                if isinstance(agent_response, dict):
                    changes_count = len(agent_response.get("changes_enumerated", []))
                    results_count = len(agent_response.get("results", []))
                    logger.info(f"Agent returned: {changes_count} changes, {results_count} results")
                
            except json.JSONDecodeError as parse_error:
                logger.error(f"JSON parse error: {parse_error}")
                logger.error(f"Raw completion length: {len(completion)} characters")
                logger.error(f"Raw completion (first 2000 chars): {completion[:2000]}")
                logger.error(f"Raw completion (last 500 chars): ...{completion[-500:]}")
                
                try:
                    if completion.count('{') > completion.count('}'):
                        logger.error("JSON appears to be missing closing braces")
                    if completion.count('[') > completion.count(']'):
                        logger.error("JSON appears to be missing closing brackets")
                    if completion.count('"') % 2 != 0:
                        logger.error("JSON appears to have unmatched quotes")
                except:
                    pass
                
                raise
            
            return {
                "completion": agent_response,
                "trace": trace_data,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Agent invocation failed: {str(e)}")
            raise
    
    def generate_response(
        self,
        model_id: str,
        messages: List[Dict],
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None
    ) -> Dict[str, Any]:
        """
        Generate response using Bedrock Converse API
        Supports both direct models (Nova Micro) and prompt routers (Nova Router)
        UNCHANGED - Generation logic remains the same
        """
        inference_config = {
            "maxTokens": max_tokens or settings.DEFAULT_MAX_TOKENS,
            "temperature": temperature or settings.DEFAULT_TEMPERATURE,
            "topP": top_p or settings.DEFAULT_TOP_P
        }
        
        try:
            if "prompt-router" in model_id:
                logger.info(f"Using Nova Prompt Router for generation")
            else:
                logger.info(f"Using direct model: {model_id}")
            
            response = self.runtime_client.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig=inference_config
            )
            
            actual_model = model_id
            if "prompt-router" in model_id:
                trace = response.get("trace", {})
                prompt_router = trace.get("promptRouter", {})
                invoked_model = prompt_router.get("invokedModelId")
                if invoked_model:
                    actual_model = invoked_model
                    logger.info(f"Router selected: {invoked_model}")
            else:
                actual_model = response.get("metrics", {}).get("modelId", model_id)
            
            return {
                "response": response["output"]["message"]["content"][0]["text"],
                "usage": response.get("usage", {}),
                "actual_model": actual_model,
                "stop_reason": response.get("stopReason")
            }
        except Exception as e:
            logger.error(f"Generation failed with {model_id}: {str(e)}")
            raise
    
    def count_tokens(self, model_id: str, prompt: str) -> Tuple[int, CountMethod]:
        """
        Count tokens for a given prompt using Bedrock
        Fallback to estimation if API not available
        UNCHANGED
        """
        try:
            response = self.runtime_client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 1}
            )
            
            usage = response.get("usage", {})
            if "inputTokens" in usage:
                return usage["inputTokens"], CountMethod.BEDROCK
            
        except Exception as e:
            logger.warning(f"Bedrock token counting failed: {str(e)}")
        
        return self.estimate_tokens(prompt), CountMethod.ESTIMATE
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count using character approximation
        Roughly 1 token per 4 characters for most models
        UNCHANGED
        """
        return max(1, len(text) // 4)
    
    def extract_usage_from_response(
        self, 
        response_data: Dict[str, Any], 
        actual_model: str
    ) -> Tuple[int, int, int, CountMethod]:
        """
        Extract token usage from Bedrock response
        Returns (prompt_tokens, completion_tokens, total_tokens, count_method)
        UNCHANGED
        """
        usage = response_data.get("usage", {})
        
        if usage and "inputTokens" in usage and "outputTokens" in usage:
            prompt_tokens = usage["inputTokens"]
            completion_tokens = usage["outputTokens"]
            total_tokens = usage.get("totalTokens", prompt_tokens + completion_tokens)
            return prompt_tokens, completion_tokens, total_tokens, CountMethod.BEDROCK
        
        return 0, 0, 0, CountMethod.NONE
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text using a Bedrock embedding model.

        Uses the model configured in settings.BEDROCK_EMBEDDING_MODEL_ID.
        The exact response shape depends on the model; this implementation
        handles common Bedrock patterns (Titan v2, Cohere-style).
        """
        if not text:
            return []

        model_id = settings.BEDROCK_EMBEDDING_MODEL_ID
        payload = {
            "inputText": text
        }

        try:
            logger.info(
                "Calling Bedrock embedding model",
                extra={
                    "model_id": model_id,
                    "text_preview": text[:80]
                }
            )

            response = self.runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
            )

            raw_body = response.get("body")
            if hasattr(raw_body, "read"):
                body_str = raw_body.read()
            else:
                body_str = raw_body

            data = json.loads(body_str)

            # Common Bedrock embedding response shapes:

            # 1) Titan Text Embeddings v2:
            #    { "embedding": [ ... ] }
            if "embedding" in data:
                return data["embedding"]

            # 2) Cohere-like:
            #    { "embeddings": [ { "embedding": [ ... ] }, ... ] }
            if "embeddings" in data:
                emb = data["embeddings"][0]
                if isinstance(emb, dict) and "embedding" in emb:
                    return emb["embedding"]
                return emb

            # 3) Generic fallback if model uses "vector" key
            if "vector" in data:
                return data["vector"]

            raise ValueError(
                f"Unexpected embedding response format from {model_id}: {list(data.keys())}"
            )

        except Exception as e:
            logger.error(
                f"Embedding generation failed: {str(e)}",
                extra={"model_id": model_id},
            )
            # Let the caller decide how to handle failure
            raise
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None,
        model_id: str = None
    ) -> List[Dict]:
        """
        Re-rank documents using Bedrock Rerank API
        UNCHANGED
        """
        import json
        
        top_k = top_k or settings.RERANK_TOP_K
        model_id = model_id or settings.RERANK_MODEL_ID
        
        if not documents:
            return []
        
        try:
            rerank_docs = []
            for i, doc in enumerate(documents):
                content = doc.get("content", {}).get("text", "")
                if content.strip():
                    rerank_docs.append({
                        "text": content,
                        "index": i
                    })
            
            if not rerank_docs:
                return documents[:top_k]
            
            response = self.runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "query": query,
                    "documents": rerank_docs,
                    "top_k": min(top_k, len(rerank_docs))
                })
            )
            
            result = json.loads(response["body"].read())
            reranked_indices = [item["index"] for item in result.get("results", [])]
            
            reranked_docs = []
            for idx in reranked_indices:
                if idx < len(documents):
                    reranked_docs.append(documents[idx])
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {str(e)}")
            return documents[:top_k]


bedrock_models = BedrockModels()


def get_bedrock_llm(*args, **kwargs):
    """Legacy function for compatibility"""
    logger.warning("get_bedrock_llm is deprecated, use bedrock_models directly")
    return bedrock_models