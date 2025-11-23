# services/llm_service.py
import uuid
import time
from typing import Optional, Dict, Any
from datetime import datetime

from schemas.request_models import (
    QueryRequest,
    QueryResponse,
    QueryComplexity,
    UsageBlock,
    CountMethod,
    ExtractedMetadata,
    ConfidenceMetrics,
    MetadataFilter,   # ← ADDED
)
from services.model_selector import model_selector
from services.vector_search import vector_search_service
from services.metadata_extraction_service import metadata_extraction_service
from services.conversation_service import conversation_service  # NEW IMPORT (already present)
from services.first_turn_router import route_greeting_or_ack  # NEW IMPORT
from models.bedrock_models import bedrock_models
from models import prompts
from core.config import settings
from core.logger import logger
from services.program_classifier import classify_program, ProgramClassification  # NEW


class LLMService:
    """
    Main LLM service orchestrating the RAG pipeline with conversation history.
    """

    def process_query(self, request: QueryRequest) -> QueryResponse:
        """
        Process query through enhanced RAG pipeline with conversation history.

        UPDATED: Adds a first-turn router for greetings/acknowledgements/out_of_scope,
        and a program classifier hint before metadata extraction, plus enriched vector
        search filters using program + chapter.
        """
        query_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(
            f"Processing query {query_id} for {request.user_tier.value} user",
            extra={
                "query_id": query_id,
                "user_id": request.userId,
                "org_id": request.organizationId,
                "conversation_id": request.conversationId,
                "query_preview": request.userPrompt[:100],
            },
        )

        try:
            # ---------------------------
            # NEW STEP: Retrieve history
            # ---------------------------
            conversation_messages = []
            last_program_from_history = None
            has_history = False

            if request.conversationId and settings.ENABLE_CONVERSATION_HISTORY:
                conversation_messages = conversation_service.build_messages_array(
                    conversation_id=request.conversationId,
                    user_id=request.userId,  # Security: validate ownership
                )

                last_program_from_history = conversation_service.get_last_program(
                    request.conversationId
                )
                has_history = bool(conversation_messages)

                logger.info(
                    "Retrieved conversation history",
                    extra={
                        "query_id": query_id,
                        "conversation_id": request.conversationId,
                        "previous_turns": len(conversation_messages),
                        "last_program": last_program_from_history,
                    },
                )


            # -------------------------------------------------------
            # NEW: First-turn router (greetings / acknowledgements / out_of_scope)
            # Trigger only when no history exists for this conversation
            # -------------------------------------------------------
            if (
                settings.ENABLE_FIRST_TURN_ROUTER
                and not has_history
                and request.userPrompt
                and request.userPrompt.strip()
            ):
                decision = route_greeting_or_ack(request.userPrompt)
                if decision and decision.intent in (
                    "greeting",
                    "acknowledgement",
                    "out_of_scope",
                ):
                    processing_time = int((time.time() - start_time) * 1000)

                    # Save a minimal first turn to history (if enabled)
                    if request.conversationId and settings.ENABLE_CONVERSATION_HISTORY:
                        save_success = conversation_service.save_turn(
                            conversation_id=request.conversationId,
                            user_id=request.userId,
                            org_id=request.organizationId,
                            user_tier=request.user_tier.value,
                            user_message=request.userPrompt,  # original user prompt
                            assistant_message=decision.assistant_reply,
                            user_metadata={
                                "timestamp": datetime.utcnow().isoformat(),
                                "router_intent": decision.intent,
                                "router_action": decision.action,
                            },
                            assistant_metadata={
                                "timestamp": datetime.utcnow().isoformat(),
                                "model_used": settings.ROUTER_MODEL_ID,
                                "processing_time_ms": processing_time,
                                "router": True,
                            },
                            last_program=None,  # router doesn’t infer program
                        )
                        if not save_success:
                            logger.warning(
                                "Failed to save router turn",
                                extra={
                                    "query_id": query_id,
                                    "conversation_id": request.conversationId,
                                },
                            )

                    # Build immediate response
                    usage = UsageBlock(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        invocations=1,
                        model=settings.ROUTER_MODEL_ID,
                        count_method=CountMethod.NONE,
                    )

                    response = QueryResponse(
                        response=decision.assistant_reply,
                        model_used="Nova Micro (router)",
                        query_id=query_id,
                        sources=[],
                        metadata={
                            "routing_path": "router:first_turn",
                            "router_intent": decision.intent,
                            "router_action": decision.action,
                            "had_conversation_history": False,
                            "processing_time_ms": processing_time,
                        },
                        processing_time_ms=processing_time,
                        usage=usage,
                    )

                    self._log_interaction(
                        query_id=query_id,
                        request=request,
                        response=response,
                        actual_model=settings.ROUTER_MODEL_ID,
                        extracted_metadata=None,
                        conversation_history_count=0,
                    )
                    return response
                # else: continue with normal pipeline

            # -------------------------------------------------------
            # NEW: Program classification (single cheap Nova-Micro call)
            # Run once per request after router, before extraction
            # -------------------------------------------------------
            program_hint_from_classifier: Optional[str] = None
            classifier: Optional[ProgramClassification] = None

            def _score_to_label(score: float) -> str:
                # keep these bins simple and conservative
                if score >= 0.85:
                    return "high"
                if score >= 0.65:
                    return "medium"
                if score > 0:
                    return "low"
                return "none"

            def _reason_from_classifier(c: Dict[str, Any]) -> str:
                src = c.get("source")
                if src == "alias":
                    return "Matched known alias/domain."
                if src == "llm":
                    return "LLM pick from short whitelist."
                if c.get("is_dnv_related") is False:
                    return "Not DNV-related."
                return "Heuristic classification."

            try:
                classifier = classify_program(request.userPrompt) or {}
                conf_score = float(classifier.get("confidence") or 0.0)
                confidence_label = _score_to_label(conf_score)
                reason = classifier.get("notes") or _reason_from_classifier(classifier)

                logger.info(
                    {
                        "event": "program_classifier_raw",
                        "dnv_related": bool(classifier.get("is_dnv_related")),
                        "program_code": classifier.get("program_code"),
                        "confidence_label": confidence_label,
                        "confidence_score": conf_score,
                        "needs_clarification": bool(
                            classifier.get("needs_clarification")
                        ),
                        "candidates": (classifier.get("candidates") or [])[:3],
                        "reason": reason,
                    }
                )

                # Keep confident hint (medium/high) to bias extraction
                if (
                    classifier.get("is_dnv_related")
                    and classifier.get("program_code")
                    and confidence_label in ("high", "medium")
                ):
                    program_hint_from_classifier = classifier["program_code"]

                # If it's a NEW conversation and clearly DNV but no confident program,
                # short-circuit to a clean clarification.
                if (
                    not has_history
                    and classifier.get("is_dnv_related")
                    and not program_hint_from_classifier
                ):
                    processing_time = int((time.time() - start_time) * 1000)
                    top = [c for c in (classifier.get("candidates") or [])][:3]
                    suggested_programs = top if top else settings.VALID_PROGRAMS[:5]

                    clar_msg = prompts.PROGRAM_CLARIFICATION_TEMPLATE.format(
                        query=request.userPrompt,
                        context_hint=reason,
                        program_suggestions=prompts.format_program_suggestions(
                            suggested_programs
                        ),
                    )

                    usage = UsageBlock(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        invocations=0,
                        model=settings.ROUTER_MODEL_ID,
                        count_method=CountMethod.NONE,
                    )

                    resp = QueryResponse(
                        response=clar_msg,
                        model_used="Clarification Required",
                        query_id=query_id,
                        sources=[],
                        metadata={
                            "routing_path": "classifier:program",
                            "clarification_needed": True,
                            "clarification_type": "program_missing",
                            "classifier": {
                                "dnv_related": classifier.get("is_dnv_related"),
                                "confidence_label": confidence_label,
                                "confidence_score": conf_score,
                                "reason": reason,
                                "candidates": classifier.get("candidates"),
                            },
                            "processing_time_ms": processing_time,
                        },
                        processing_time_ms=processing_time,
                        usage=usage,
                    )

                    self._log_interaction(
                        query_id=query_id,
                        request=request,
                        response=resp,
                        actual_model=settings.ROUTER_MODEL_ID,
                        extracted_metadata=None,
                        conversation_history_count=len(conversation_messages),
                    )
                    return resp

            except Exception as e:
                logger.warning({"event": "program_classifier_failed", "error": str(e)})

            # -------------------------------------------------------
            # STEP 1: Extract metadata (with last_program + classifier hint)
            # -------------------------------------------------------
            # NOTE: Keep this block. It passes history + classifier hint into extraction,
            # and we can fall back to the classifier hint if extractor missed.
            conversation_context_for_extraction = {"last_program": last_program_from_history}

            # Merge with any additional context from request body
            if request.context:
                conversation_context_for_extraction.update(request.context)

            # NEW: pass Nova Micro's program hint (canonical code) to the extractor
            if program_hint_from_classifier:
                conversation_context_for_extraction["program_hint"] = program_hint_from_classifier

            extracted_metadata = metadata_extraction_service.extract_metadata(
                query=request.userPrompt,
                conversation_context=conversation_context_for_extraction,
                user_id=request.userId,
                org_id=request.organizationId,
            )

            
            # -------------------------------------------------------
            # STEP 3: Vector search with metadata filters (now enriched)
            # -------------------------------------------------------
            # Build an effective filter:
            effective_filter = None
            if not effective_filter:
                clf_filter_dict = (classifier or {}).get("metadata_filter")
                if clf_filter_dict:
                    try:
                        effective_filter = MetadataFilter(**clf_filter_dict)
                        logger.info(
                            {
                                "event": "vector_filter_enriched",
                                "query_id": query_id,
                                "filter_preview": clf_filter_dict,
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            {"event": "vector_filter_build_failed", "error": str(e)}
                        )

            if(classifier.get("confidence_score") > 0.0):
                search_results = vector_search_service.search(
                query=request.userPrompt,
                top_k=8,
                include_metadata=True,
                metadata_filter=effective_filter,  # ← use effective filter
                )

            if not search_results["context"]:
                logger.warning(
                    "No relevant context found",
                    extra={
                        "query_id": query_id,
                        "program": extracted_metadata.program,
                        "filter": extracted_metadata.filter.model_dump(
                            exclude_none=True
                        )
                        if extracted_metadata.filter
                        else None,
                    },
                )

            # -------------------------------------------------------
            # STEP 4: Check confidence metrics
            # -------------------------------------------------------
            confidence_metrics = search_results.get("confidence")
            # if confidence_metrics and confidence_metrics.is_low_confidence:
            if (
                confidence_metrics
                and confidence_metrics.is_low_confidence
                and confidence_metrics.num_results == 0
                ):   
                
                
                logger.info(
                    "Low confidence detected",
                    extra={
                        "query_id": query_id,
                        "avg_score": confidence_metrics.avg_score,
                        "threshold": settings.LOW_CONFIDENCE_THRESHOLD,
                    },
                )

                return self._build_low_confidence_response(
                    query_id=query_id,
                    request=request,
                    extracted_metadata=classifier,
                    confidence_metrics=confidence_metrics,
                    processing_time=int((time.time() - start_time) * 1000),
                )

            # -------------------------------------------------------
            # STEP 5: Select model based on user tier
            # -------------------------------------------------------
            model_id, complexity = model_selector.select_model(
                query=request.userPrompt, user_tier=request.user_tier
            )

            # -------------------------------------------------------
            # STEP 6: Build prompt
            # -------------------------------------------------------
            system_prompt = self._build_prompt(
                query=request.userPrompt,
                context=search_results["context"],
                model_id=model_id,
                attachments=request.attachments,
                conversation_context=request.context,  # Still use context for display
                extracted_metadata=classifier,
            )

            # -------------------------------------------------------
            # STEP 7: Build messages array for Bedrock Converse API
            # -------------------------------------------------------
            current_message = {"role": "user", "content": [{"text": system_prompt}]}

            messages = conversation_messages + [current_message]

            logger.info(
                "Built messages array for Bedrock",
                extra={
                    "query_id": query_id,
                    "total_messages": len(messages),
                    "history_messages": len(conversation_messages),
                    "has_conversation_context": len(conversation_messages) > 0,
                },
            )

            # -------------------------------------------------------
            # STEP 8: Generate response
            # -------------------------------------------------------
            response_data = bedrock_models.generate_response(
                model_id=model_id,
                messages=messages,  # includes conversation history
                max_tokens=settings.DEFAULT_MAX_TOKENS,
                temperature=settings.DEFAULT_TEMPERATURE,
            )

            ai_response_text = response_data["response"]
            actual_model = response_data.get("actual_model", model_id)

            # -------------------------------------------------------
            # STEP 9: Calculate metrics and token usage
            # -------------------------------------------------------
            processing_time = int((time.time() - start_time) * 1000)

            (
                prompt_tokens,
                completion_tokens,
                total_tokens,
                count_method,
            ) = bedrock_models.extract_usage_from_response(response_data, actual_model)

            if count_method == CountMethod.NONE:
                prompt_tokens, fallback_method = bedrock_models.count_tokens(
                    actual_model, system_prompt
                )
                completion_tokens = bedrock_models.estimate_tokens(ai_response_text)
                total_tokens = prompt_tokens + completion_tokens
                count_method = fallback_method

            usage = UsageBlock(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                invocations=1,
                model=actual_model,
                count_method=count_method,
            )

            model_display = model_selector.get_model_description(actual_model)
            if "prompt-router" in model_id and actual_model != model_id:
                specific_model = model_selector.get_model_description(actual_model)
                model_display = f"Nova Prompt Router \u2192 {specific_model}"

            # -------------------------------------------------------
            # STEP 10: Save conversation turn to Redis
            # -------------------------------------------------------
            if request.conversationId and settings.ENABLE_CONVERSATION_HISTORY:
                save_success = conversation_service.save_turn(
                    conversation_id=request.conversationId,
                    user_id=request.userId,
                    org_id=request.organizationId,
                    user_tier=request.user_tier.value,
                    user_message=request.userPrompt,  # Store original prompt
                    assistant_message=ai_response_text,
                    user_metadata={
                        "timestamp": datetime.utcnow().isoformat(),
                        "extracted_program": classifier.get("program_code"),
                        "program_confidence": classifier.get("confidence_label"),
                        "chapter_hints": classifier.get("chapter_code"),
                    },
                    assistant_metadata={
                        "timestamp": datetime.utcnow().isoformat(),
                        "model_used": actual_model,
                        "sources_count": len(search_results.get("sources", [])),
                        "processing_time_ms": processing_time,
                        "confidence_score": confidence_metrics.avg_score
                        if confidence_metrics
                        else None,
                    },
                    last_program=classifier.get("program_code"),
                )

                if not save_success:
                    logger.warning(
                        "Failed to save conversation turn",
                        extra={
                            "query_id": query_id,
                            "conversation_id": request.conversationId,
                        },
                    )

            # -------------------------------------------------------
            # STEP 11: Build response with enhanced metadata
            # -------------------------------------------------------
            response = QueryResponse(
                response=ai_response_text,
                model_used=model_display,
                query_id=query_id,
                sources=(search_results.get("sources") or []),
                metadata=self._build_metadata(
                    request=request,
                    complexity=complexity,
                    processing_time=processing_time,
                    actual_model=actual_model,
                    router_model=model_id,
                    extracted_metadata=classifier,
                    confidence_metrics=confidence_metrics,
                    had_conversation_history=len(conversation_messages) > 0,
                ),
                processing_time_ms=processing_time,
                usage=usage,
            )

            self._log_interaction(
                query_id,
                request,
                response,
                actual_model,
                classifier,
                len(conversation_messages),
            )

            return response

        except Exception as e:
            logger.error(
                f"Query processing failed for {query_id}: {str(e)}",
                extra={"query_id": query_id, "error": str(e)},
                exc_info=True,
            )
            return self._fallback_response(query_id)

    def _build_prompt(
        self,
        query: str,
        context: str,
        model_id: str,
        attachments: Optional[list] = None,
        conversation_context: Optional[dict] = None,
        extracted_metadata: Optional[ProgramClassification] = None,
    ) -> str:
        """
        Build system prompt with RAG context.

        NOTE: Conversation history is handled separately via messages array,
        not included in this prompt. This is the system/context prompt only.

        UNCHANGED from your existing implementation - works as-is.
        """
        attachment_text = ""
        if attachments:
            attachment_content = "\n".join(attachments)
            attachment_text = f"""
Additional Reference Materials:
{attachment_content}"""

        context_text = ""
        if conversation_context:
            context_items = []
            for key, value in conversation_context.items():
                context_items.append(f"- {key}: {value}")
            if context_items:
                context_text = f"""
Conversation Context:
{chr(10).join(context_items)}"""

        if not context:
            return f"""You are a DNV Healthcare Standards expert assistant.{context_text}{attachment_text}
Question: {query}
Note: Limited context is available in the knowledge base for this query. 
Please provide general guidance based on DNV NIAHO standards and suggest contacting DNV directly at healthcare@dnv.com for specific requirements.
Answer:"""

        if extracted_metadata and extracted_metadata.get("program_code"):
            program_display = self._get_program_display_name(extracted_metadata.get("program_code"))

            if "nova-micro" in model_id.lower():
                return f"""Answer this question about {program_display}.{context_text}{attachment_text}
Context from {program_display} Standards:
{context}
Question: {query}
Provide a brief, accurate answer:"""

            return prompts.QA_PROMPT_WITH_PROGRAM_TEMPLATE.format(
                program=program_display, context=context, question=query
            ) + context_text + attachment_text

        if "nova-micro" in model_id.lower():
            return f"""Answer this healthcare question based on DNV standards.{context_text}{attachment_text}
Context from DNV Standards:
{context}
Question: {query}
Provide a brief, accurate answer:"""

        return f"""You are an expert assistant for DNV Healthcare Standards and NIAHO accreditation.{context_text}{attachment_text}
Relevant DNV Standards Documentation:
{context}
User Question: {query}
Please provide a comprehensive answer that:
1. Directly addresses the question with specific requirements
2. References relevant standards and sections
3. Includes practical implementation guidance
4. Notes any critical compliance considerations
5. Consider any provided reference materials and conversation context
Answer:"""

    def _build_clarification_response(
        self,
        query_id: str,
        request: QueryRequest,
        extracted_metadata: ProgramClassification,
        processing_time: int,
    ) -> QueryResponse:
        """
        Build response when user clarification is needed for metadata

        NEW: Handles program clarification flow
        """
        clarification_message = extracted_metadata.clarification_message

        if not clarification_message:
            clarification_message = (
                "Could you please provide more details to help me answer your question accurately?"
            )

        # Format program suggestions if available
        suggested_programs = []
        if "niaho" in request.userPrompt.lower():
            suggested_programs = settings.NIAHO_SUB_PROGRAMS
        else:
            suggested_programs = settings.VALID_PROGRAMS[:5]

        program_list = prompts.format_program_suggestions(suggested_programs)

        full_message = prompts.PROGRAM_CLARIFICATION_TEMPLATE.format(
            query=request.userPrompt,
            context_hint=extracted_metadata.clarification_message or "",
            program_suggestions=program_list,
        )

        fallback_usage = UsageBlock(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            invocations=0,
            model="clarification",
            count_method=CountMethod.NONE,
        )

        return QueryResponse(
            response=full_message,
            model_used="Clarification Required",
            query_id=query_id,
            sources=[],
            metadata={
                "clarification_needed": True,
                "clarification_type": "program_missing",
                "extracted_metadata": extracted_metadata.model_dump(),
                "processing_time_ms": processing_time,
            },
            processing_time_ms=processing_time,
            usage=fallback_usage,
        )

    def _build_low_confidence_response(
        self,
        query_id: str,
        request: QueryRequest,
        extracted_metadata: ProgramClassification,
        confidence_metrics: ConfidenceMetrics,
        processing_time: int,
    ) -> QueryResponse:
        """
        Build response when search confidence is low

        NEW: Asks for clarification instead of generating potentially incorrect answer
        """
        clarification_questions = prompts.build_low_confidence_questions(
            query=request.userPrompt, chapter_hints=extracted_metadata.get("chapter_code")
        )

        clarification_message = prompts.LOW_CONFIDENCE_CLARIFICATION_TEMPLATE.format(
            query=request.userPrompt, clarification_questions=clarification_questions
        )

        fallback_usage = UsageBlock(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            invocations=0,
            model="clarification",
            count_method=CountMethod.NONE,
        )

        return QueryResponse(
            response=clarification_message,
            model_used="Clarification Required",
            query_id=query_id,
            sources=[],
            metadata={
                "clarification_needed": True,
                "clarification_type": "low_confidence",
                "confidence_metrics": confidence_metrics.model_dump(),
                "processing_time_ms": processing_time,
            },
            processing_time_ms=processing_time,
            usage=fallback_usage,
        )

    def _build_metadata(
        self,
        request: QueryRequest,
        complexity: QueryComplexity,
        processing_time: int,
        actual_model: str,
        router_model: str,
        extracted_metadata: Optional[ProgramClassification] = None,
        confidence_metrics: Optional[ConfidenceMetrics] = None,
        had_conversation_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Build response metadata.

        UPDATED: Added conversation history indicator.
        """
        metadata = {
            "complexity_detected": complexity.value,
            "actual_model_used": actual_model,
            "routing_path": "direct" if "micro" in actual_model.lower() else "routed",
            "user_tier": request.user_tier.value,
            "organization_id": request.organizationId,
            "processing_time_ms": processing_time,
            "kb_id": settings.OPENSEARCH_KB_ID,
            "search_type": settings.KB_SEARCH_TYPE,
            "had_conversation_history": had_conversation_history,
        }

        if "prompt-router" in router_model and actual_model != router_model:
            metadata["router_selected_model"] = actual_model
            metadata["router_used"] = True
        else:
            metadata["router_used"] = False

        if extracted_metadata:
            metadata["program_used"] = extracted_metadata.get("program_code")
            metadata["program_confidence"] = extracted_metadata.get("confidence_label")
            metadata["extraction_source"] = extracted_metadata.get("source")
            metadata["chapter_hints"] = extracted_metadata.get("chapter_code")

        if confidence_metrics:
            metadata["confidence_score"] = confidence_metrics.avg_score
            metadata["confidence_max_score"] = confidence_metrics.max_score
            metadata["is_low_confidence"] = confidence_metrics.is_low_confidence

        return metadata

    def _get_program_display_name(self, program_code: str) -> str:
        """
        Get human-readable program name
        """
        program_names = {
            "NIAHO-HOSP": "NIAHO Hospital Accreditation",
            "NIAHO-CAH": "NIAHO Critical Access Hospital",
            "NIAHO-PSY": "NIAHO Psychiatric Hospital",
            "NIAHO-PROC": "NIAHO Procedural Sedation",
            "CARDIAC_VASCULAR": "Cardiac and Vascular Certification",
            "STROKE": "Stroke Center Certification",
            "INFECTION": "Infection Prevention and Control",
            "CYBERSECURITY": "Cybersecurity Certification",
            "GLYCEMIC": "Glycemic Control Certification",
            "ORTHO_SPINE": "Orthopedic and Spine Certification",
            "PALLIATIVE": "Palliative Care Certification",
            "EXTRACORPOREAL": "Extracorporeal Life Support",
            "VENTRICULAR": "Ventricular Assist Device",
            "SURVEY": "Survey Process",
        }

        return program_names.get(program_code, program_code)

    def _fallback_response(self, query_id: str) -> QueryResponse:
        """
        Generate fallback response on error
        UNCHANGED
        """
        fallback_usage = UsageBlock(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            invocations=0,
            model="fallback",
            count_method=CountMethod.NONE,
        )

        return QueryResponse(
            response="I apologize, but I'm unable to process your question at this moment. Please contact DNV support at healthcare@dnv.com for assistance.",
            model_used="Fallback",
            query_id=query_id,
            sources=[],
            metadata={"error": "fallback_response", "processing_time_ms": 0},
            processing_time_ms=0,
            usage=fallback_usage,
        )

    def _log_interaction(
        self,
        query_id: str,
        request: QueryRequest,
        response: QueryResponse,
        actual_model: str,
        extracted_metadata: Optional[ProgramClassification] = None,
        conversation_history_count: int = 0,
    ):
        """
        Log interaction for monitoring.

        UPDATED: Added conversation history count.
        """
        log_data = {
            "event": "query_processed",
            "query_id": query_id,
            "user_tier": request.user_tier.value,
            "user_id": request.userId,
            "organization_id": request.organizationId,
            "conversation_id": request.conversationId,
            "had_conversation_history": conversation_history_count > 0,
            "conversation_history_messages": conversation_history_count,
            "model_id": actual_model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "invocations": response.usage.invocations,
                "count_method": response.usage.count_method.value,
            },
            "processing_time_ms": response.processing_time_ms,
            "success": True,
        }

        if extracted_metadata:
            log_data["metadata_extraction"] = {
                "program": extracted_metadata.get("program_code"),
                "confidence": extracted_metadata.get("confidence_level"),
                "source": extracted_metadata.get("source"),
                "needs_clarification": extracted_metadata.get("needs_clarification"),
            }

        logger.info(log_data)


llm_service = LLMService()
