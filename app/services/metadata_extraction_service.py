# services/metadata_extraction_service.py
from __future__ import annotations

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from core.logger import logger
from core.config import settings

# Project models
from schemas.request_models import ExtractedMetadata, MetadataFilter, ConfidenceLevel

# Local helpers you already have
from services.program_catalog import (
    extract_chapter_code,
    resolve_program_or_domain,  # returns {"kind": "program"/"domain", ...} or None
    # If you later want a soft default from a domain, you could optionally import:
    # default_program_for_domain
)


def _mk_filter(program: Optional[str], chapter: Optional[str]) -> Optional[MetadataFilter]:
    """
    Build a MetadataFilter that matches your schemas/request_models.py.
    Bedrock filter ultimately uses only 'program' (per your to_bedrock_filter), but we
    still populate 'chapter_code' so the signal survives for any future use.
    """
    if not program and not chapter:
        return None
    return MetadataFilter(
        program=program,
        edition=None,            # reserved for future
        domain_code=None,        # reserved for future
        chapter_code=chapter     # retained for potential downstream use
    )


class MetadataExtractionService:
    """
    Extracts canonical program code + chapter hints + a retrieval filter.

    Priority order:
      1) program_hint from classifier (passed via conversation_context["program_hint"])
      2) last_program from conversation history (conversation_context["last_program"])
      3) local resolution from user text (resolve_program_or_domain), which may yield:
         - a concrete program, or
         - a domain-only signal (we DO NOT force a default program here)
      4) If still ambiguous, mark needs_clarification=True and provide a clear message.
    """

    def extract_metadata(
        self,
        query: str,
        conversation_context: Optional[Dict[str, Any]],
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> ExtractedMetadata:

        # ----------------------------
        # 1) Classifier hint + history
        # ----------------------------
        program_hint = None
        last_program = None
        if conversation_context:
            program_hint = conversation_context.get("program_hint")  # canonical if present
            last_program = conversation_context.get("last_program")

        # -----------------------------------
        # 2) Chapter code from user text, e.g. "QM.1"
        # -----------------------------------
        chapter = extract_chapter_code(query)
        chapter_hints: List[str] = [chapter] if chapter else []

        # ---------------------------------------------------
        # 3) Canonical program selection (hint > history > local)
        # ---------------------------------------------------
        program: Optional[str] = program_hint or last_program
        source = "classifier_hint" if program_hint else ("history" if last_program else "local_or_none")

        if not program:
            # Soft local fallback (no extra model calls)
            try:
                local = resolve_program_or_domain(query)
            except Exception as e:
                logger.debug({"event": "resolve_local_failed", "error": str(e)})
                local = None

            if local:
                if local.get("kind") == "program" and local.get("program"):
                    program = local["program"]
                    if not chapter and local.get("chapter_code"):
                        chapter = local["chapter_code"]
                        chapter_hints = [chapter]
                    source = "local_program_alias"
                elif local.get("kind") == "domain":
                    # Domain-only signal; do NOT force a program here to avoid false positives.
                    # If you later choose to bias a default, you could enable something like:
                    #   program = default_program_for_domain(local.get("domain"))
                    # For now we keep it conservative.
                    if not chapter and local.get("chapter_code"):
                        chapter = local["chapter_code"]
                        chapter_hints = [chapter]
                    source = "local_domain_hint"

        # ---------------------------------------------------
        # 4) Clarification if we still don't have a program
        # ---------------------------------------------------
        needs_clarification = False
        clarification_message: Optional[str] = None

        if not program:
            needs_clarification = True

            # Context-aware, compact suggestions (UI can render a picker)
            # If the text mentioned "niaho", we suggest NIAHO variants first; otherwise show a short list.
            niaho_bias = "niaho" in (query or "").lower()
            if niaho_bias and getattr(settings, "NIAHO_SUB_PROGRAMS", None):
                suggest = settings.NIAHO_SUB_PROGRAMS
            else:
                suggest = getattr(settings, "VALID_PROGRAMS", [])[:5] or ["NIAHO-HOSP", "CARDIAC_VASCULAR", "STROKE"]

            clarification_message = (
                "I can be precise if you tell me which program this is about "
                f"(e.g., {', '.join(suggest)}). "
                "You can also include a chapter like QM.1 if you know it."
            )

        # ---------------------------------------------------
        # 5) Retrieval filter (program + chapter hint)
        # ---------------------------------------------------
        retrieval_filter = _mk_filter(program, chapter)

        # ---------------------------------------------------
        # 6) Confidence: MEDIUM if classifier hint, else LOW
        # ---------------------------------------------------
        if program_hint:
            program_conf = ConfidenceLevel.MEDIUM
        elif program:
            program_conf = ConfidenceLevel.LOW   # from history or local alias → softer confidence
        else:
            program_conf = ConfidenceLevel.LOW

        # ---------------------------------------------------
        # 7) Build & log ExtractedMetadata
        # ---------------------------------------------------
        meta = ExtractedMetadata(
            program=program,                             # canonical or None
            program_confidence=program_conf,             # ConfidenceLevel enum
            chapter_hints=chapter_hints,                 # e.g., ["QM.1"]
            needs_clarification=needs_clarification,
            clarification_reason="program_missing" if needs_clarification else None,
            source=source,
            filter=retrieval_filter,                     # MetadataFilter or None
            clarification_message=clarification_message,
        )

        logger.info(
            {
                "event": "metadata_extracted",
                "program": meta.program,
                "program_confidence": meta.program_confidence.value,
                "chapter_hints": meta.chapter_hints,
                "needs_clarification": meta.needs_clarification,
                "source": meta.source,
                "filter_preview": meta.filter.model_dump() if meta.filter else None,
            }
        )

        return meta


metadata_extraction_service = MetadataExtractionService()
