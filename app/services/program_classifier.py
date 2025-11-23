# services/program_classifier.py
from __future__ import annotations
from typing import Optional, Literal, TypedDict, List, Dict  # + Dict

from core.logger import logger
from core.config import settings
from models.bedrock_models import bedrock_models
import re

# Catalog + helpers
from services.program_catalog import (
    DOMAIN_TO_PROGRAMS,
    resolve_program_or_domain,
    extract_chapter_code,
    scan_text_for_any_alias,
)

def _normalize_model_id(mid: str | None) -> str:
    """
    Normalize user/config-provided model ids to a valid Bedrock identifier.
    Default to 'amazon.nova-micro-v1:0' which works with Converse.
    """
    default_id = "amazon.nova-lite-v1:0"
    if not mid:
        return default_id
    if mid.startswith("arn:aws:bedrock:"):
        return mid
    # common config value without suffix -> add ':0'
    if mid.startswith("amazon.nova-lite-v1") and ":" not in mid:
        return "amazon.nova-lite-v1:0"
    return mid

_ROUTER_MODEL_ID = _normalize_model_id(getattr(settings, "ROUTER_LITE_MODEL_ID", "amazon.nova-lite-v1:0"))

# ---------- Public result type (kept name for back-compat) ----------

class ProgramClassification(TypedDict, total=False):
    is_dnv_related: bool
    program_code: Optional[str]             # e.g., "ACPC"
    domain_code: Optional[str]              # e.g., "CARDIAC_VASCULAR"
    chapter_code: Optional[str]             # e.g., "QM.1"
    source: Literal["alias", "llm", "none"]
    confidence: float                       # 0..1 (kept for existing code)
    confidence_score: float                 # duplicate of confidence for logger compatibility
    confidence_label: Literal["high", "medium", "low", "none"]
    needs_clarification: bool
    candidates: List[str]                   # relevant when domain hit is ambiguous
    reason: Optional[str]                   # short human-readable rationale
    notes: Optional[str]
    # NEW: plain dict you can pass to MetadataFilter(**dict)
    metadata_filter: Optional[Dict[str, Optional[str]]]

# ---------- Helpers ----------

def _domain_for_program(program_code: Optional[str]) -> Optional[str]:
    if not program_code:
        return None
    for d, kids in DOMAIN_TO_PROGRAMS.items():
        if program_code in kids:
            return d
    return None

def _score_to_label(score: float | None) -> Literal["high", "medium", "low", "none"]:
    if score is None or score <= 0.0:
        return "none"
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"

# NEW: small helper to build the dict-shaped filter without importing Pydantic here
def _build_metadata_filter(program_code: Optional[str], chapter_code: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
    """
    Return a dict compatible with schemas.request_models.MetadataFilter(**dict).
    As requested: program=<code>, edition=None, domain_code=None, chapter_code=<chapter or None>.
    """
    if not program_code:
        return None
    return {
        "program": program_code,
        "edition": None,
        "domain_code": (chapter_code.split('.', 1)[0][:2] if isinstance(chapter_code, str) and '.' in chapter_code else None),
        "chapter_code": chapter_code
    }

def _ask_nova_pick_program(user_text: str, candidates: List[str]) -> Optional[str]:
    """
    Ask Nova Micro to choose exactly one code from `candidates`.
    We keep the prompt tiny and require STRICT JSON.
    """
    if not candidates:
        return None

    short_list = candidates[:20]  # keep tokens tiny

    prompt = (
        "You classify healthcare accreditation questions for DNV-related programs.\n"
        "IMPORTANT RULES:\n"
        "• Only pick a program code if the user text clearly mentions or strongly implies that specific program\n"
        "  (including obvious misspellings/spacing like 'a c p c' for ACPC).\n"
        "• If the text mentions only chapter codes like 'QM.1' with no program context, return null.\n"
        "• If none of the allowed program codes reasonably fits, return null.\n\n"
        f"User message:\n{user_text}\n\n"
        f"Allowed program codes: {', '.join(short_list)}\n\n"
        'Return ONLY this JSON:\n'
        '{ "program_code": "<one-of-allowed-or-null>" }'
    )

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    try:
        resp = bedrock_models.generate_response(
            model_id=_ROUTER_MODEL_ID,
            messages=messages,
            max_tokens=48,
            temperature=0.0,
            top_p=1.0,
        )
        raw = (resp.get("response") or "").strip()
        picked = _parse_program_pick_json(raw)
        if picked in short_list:
            return picked
        if picked and picked.upper() in short_list:
            return picked.upper()
        return None
    except Exception as e:
        logger.warning(f"program_classifier LLM pick failed: {e}")
        return None

def _parse_program_pick_json(raw: str) -> Optional[str]:
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            import json
            obj = json.loads(raw)
            val = obj.get("program_code")
            if isinstance(val, str):
                return val.strip()
            return None
        except Exception:
            return None
    import re
    m = re.search(r'"program_code"\s*:\s*"([A-Za-z0-9_+-]+)"', raw)
    return m.group(1) if m else None

# ---------- Core entry point ----------

def classify_program(user_text: str) -> ProgramClassification:
    """
    1) Try exact/alias-based resolution (zero-cost, deterministic).
    2) If DOMAIN-only or no hit, ask Nova Micro to choose from a *short* whitelist.
       We NEVER guess purely from chapter codes.
    """
    chapter = extract_chapter_code(user_text)
    resolution = resolve_program_or_domain(user_text)

    # Step 1: quick, deterministic path via alias/domain resolution
    if resolution:
        if resolution.get("kind") == "program":
            score = 0.95  # alias hit is strong
            program_code = resolution["program"]
            out: ProgramClassification = {
                "is_dnv_related": True,
                "program_code": program_code,
                "domain_code": (chapter.split('.', 1)[0][:2] if isinstance(chapter, str) and '.' in chapter else None),
                "chapter_code": chapter,
                "source": "alias",
                "confidence": score,
                "confidence_score": score,
                "confidence_label": _score_to_label(score),
                "needs_clarification": False,
                "candidates": [],
                "reason": f"Matched program alias '{resolution.get('alias_hit','')}'".strip() or "Matched program alias",
                "metadata_filter": _build_metadata_filter(program_code, chapter),  # NEW
            }
            # If multiple alias hints conflict, nudge confidence down slightly
            hits = scan_text_for_any_alias(user_text)
            if len(hits) > 1:
                out["confidence"] = 0.90
                out["confidence_score"] = 0.90
                out["confidence_label"] = _score_to_label(0.90)
                out["notes"] = f"Multiple alias hits detected: {hits}"
            return out

        if resolution.get("kind") == "domain":
            domain = resolution["domain"]
            kids = resolution.get("candidates", [])
            alias_hit = resolution.get("alias_hit", "")
            # Single program in that domain → deterministic
            if len(kids) == 1:
                program_code = kids[0]
                score = 0.90
                return {
                    "is_dnv_related": True,
                    "program_code": program_code,
                    "domain_code": (chapter.split('.', 1)[0][:2] if isinstance(chapter, str) and '.' in chapter else None),
                    "chapter_code": chapter,
                    "source": "alias",
                    "confidence": score,
                    "confidence_score": score,
                    "confidence_label": _score_to_label(score),
                    "needs_clarification": False,
                    "candidates": [],
                    "reason": f"Domain alias '{alias_hit}' has a single child program; auto-selected".strip(),
                    "notes": "Single program under domain alias; auto-selected.",
                    "metadata_filter": _build_metadata_filter(program_code, chapter),  # NEW
                }
            # Multiple programs → try LLM to pick among the domain’s short list
            picked = _ask_nova_pick_program(user_text, kids)
            if picked:
                score = 0.80
                return {
                    "is_dnv_related": True,
                    "program_code": picked,
                    "domain_code": (chapter.split('.', 1)[0][:2] if isinstance(chapter, str) and '.' in chapter else None),
                    "chapter_code": chapter,
                    "source": "llm",
                    "confidence": score,
                    "confidence_score": score,
                    "confidence_label": _score_to_label(score),
                    "needs_clarification": False,
                    "candidates": [],
                    "reason": f"Picked by LLM from domain '{domain}' candidates",
                    "metadata_filter": _build_metadata_filter(picked, chapter),  # NEW
                }
            # Couldn’t pick → ask user to clarify
            score = 0.60
            return {
                "is_dnv_related": True,
                "program_code": None,
                "domain_code": (chapter.split('.', 1)[0][:2] if isinstance(chapter, str) and '.' in chapter else None),
                "chapter_code": chapter,
                "source": "alias",
                "confidence": score,
                "confidence_score": score,
                "confidence_label": _score_to_label(score),
                "needs_clarification": True,
                "candidates": kids,
                "reason": f"Domain '{domain}' detected; ambiguous among {', '.join(kids)}",
                "notes": f"Need user to pick one of: {', '.join(kids)}",
                "metadata_filter": None,  # NEW
            }

    # Step 2: no alias/domain hit — try LLM pick from a small whitelist.
    # IMPORTANT: The prompt forbids guessing from chapter-only mentions; null in that case.
    from services.program_catalog import PROGRAM_CODES  # local import to avoid cycles
    picked = _ask_nova_pick_program(user_text, PROGRAM_CODES)
    if picked:
        score = 0.70
        return {
            "is_dnv_related": True,
            "program_code": picked,
            "domain_code": (chapter.split('.', 1)[0][:2] if isinstance(chapter, str) and '.' in chapter else None)
,           "chapter_code": chapter,
            "source": "llm",
            "confidence": score,
            "confidence_score": score,
            "confidence_label": _score_to_label(score),
            "needs_clarification": False,
            "candidates": [],
            "reason": "Picked by LLM from global short program list",
            "metadata_filter": _build_metadata_filter(picked, chapter),  # NEW
        }

    # Nothing looks DNV-like → not related or too vague
    score = 0.0
    return {
        "is_dnv_related": True,
        "program_code": None,
        "domain_code": (chapter.split('.', 1)[0][:2] if isinstance(chapter, str) and '.' in chapter else None),
        "chapter_code": chapter,
        "source": "none",
        "confidence": score,
        "confidence_score": score,
        "confidence_label": _score_to_label(score),
        "needs_clarification": True,
        "candidates": [],
        "reason": "No DNV program or domain detected",
        "metadata_filter": None,  # NEW
    }
