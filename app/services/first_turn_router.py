from typing import Optional, Dict, Any
import json
import re

from schemas.request_models import RouterDecision
from models.prompts import ROUTER_SYSTEM_PROMPT
from core.config import settings
from core.logger import logger
from models.bedrock_models import bedrock_models  # your working wrapper


def _normalize(mid: str) -> str:
    if not mid:
        return "amazon.nova-micro-v1:0"
    if mid.startswith("arn:aws:bedrock:"):
        return mid
    if mid.startswith("amazon.nova-micro-v1") and ":" not in mid:
        return "amazon.nova-micro-v1:0"
    return mid

_ROUTER_MODEL_ID = _normalize(getattr(settings, "ROUTER_MODEL_ID", "amazon.nova-micro-v1:0"))
_MAX_WORDS = int(getattr(settings, "ROUTER_MAX_GREETING_WORDS", 25))
_MAX_LOG = 400

# ----------------- heuristics
_SMALL_TALK_RE = re.compile(
    r"\b(hi|hello|hey|good\s+(morning|afternoon|evening)|how\s+are\s+you|how['’]s\s+it\s+going)\b",
    re.IGNORECASE,
)
_DOMAIN_HINT_RE = re.compile(
    r"\b(dnv|niaho|cah|hosp|behav|lab|standard|chapter|qm\.\d+|ic\.\d+|cc\.\d+)\b",
    re.IGNORECASE,
)
def _looks_like_small_talk(txt: str) -> bool:
    return bool(_SMALL_TALK_RE.search(txt)) and not _DOMAIN_HINT_RE.search(txt)

# ----------------- JSON extraction/parsing
_CODE_FENCE_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_BRACE = re.compile(r"\{.*\}", re.DOTALL)

def _first_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    m = _CODE_FENCE_JSON.search(text)
    if m:
        return m.group(1).strip()
    m = _FIRST_BRACE.search(text)
    if m:
        return m.group(0).strip()
    return None

def _loads_dict(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        if isinstance(obj, str) and s.startswith('"') and s.endswith('"'):
            # Model returned a JSON string like "\"{...}\""
            inner = s.strip('"')
            return _loads_dict(inner)
        return None
    except Exception:
        return None

def _coerce_single_quotes(s: str) -> Optional[Dict[str, Any]]:
    t = s.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return None
    t = t.replace("\r", "").replace("\n", " ")
    t = t.replace("'", '"')
    t = re.sub(r'\bTrue\b', 'true', t)
    t = re.sub(r'\bFalse\b', 'false', t)
    t = re.sub(r'\bNone\b', 'null', t)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return obj[0]
        return None
    except Exception:
        return None

def _parse_router_json(raw: str) -> Dict[str, Any]:
    """
    Robustly parse model output to {intent, action, assistant_reply}.
    Returns {} if nothing usable.
    """
    if not raw:
        return {}

    block = _first_json_block(raw) or raw
    obj = _loads_dict(block) or _coerce_single_quotes(block)
    if not isinstance(obj, dict):
        return {}

    # Tolerate alternate key names from the model
    def _get(d: Dict[str, Any], *keys: str) -> str:
        for k in keys:
            if k in d:
                return str(d[k])
        return ""

    intent = _get(obj, "intent", "label", "class").strip().lower()
    action = _get(obj, "action", "route", "decision").strip().lower()
    reply  = " ".join(_get(obj, "assistant_reply", "reply", "message", "text").split())

    return {"intent": intent, "action": action, "assistant_reply": reply}

# ----------------- LLM calls

def _call_router_llm(user_text: str, max_words: int) -> str:
    """
    One-user-turn instruction (your wrapper supports only user/assistant roles).
    """
    instruct = ROUTER_SYSTEM_PROMPT.format(max_words=max_words)
    combined = (
        f"{instruct}\n\n"
        f"User message:\n{user_text}\n\n"
        f"Return ONLY one JSON object with double quotes. No code fences. No prose."
    )
    messages = [{"role": "user", "content": [{"text": combined}]}]

    resp = bedrock_models.generate_response(
        model_id=_ROUTER_MODEL_ID,
        messages=messages,
        max_tokens=getattr(settings, "ROUTER_MAX_TOKENS", 128),
        temperature=getattr(settings, "ROUTER_TEMPERATURE", 0.1),
        top_p=getattr(settings, "ROUTER_TOP_P", 0.9),
    )
    return resp.get("response", "") or ""

def _call_greeting_llm(user_text: str, max_words: int) -> str:
    """
    Last-resort: ask Nova to write a short greeting reply (still model-authored).
    """
    prompt = (
        f"You are a DNV standards assistant. Reply briefly (≤ {max_words} words) to this greeting/small-talk.\n"
        f"Message: {user_text}\n"
        f"Keep it friendly and natural."
    )
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    resp = bedrock_models.generate_response(
        model_id=_ROUTER_MODEL_ID,
        messages=messages,
        max_tokens=max_words * 3,
        temperature=getattr(settings, "ROUTER_TEMPERATURE", 0.1),
        top_p=getattr(settings, "ROUTER_TOP_P", 0.9),
    )
    return (resp.get("response", "") or "").strip()

def _policy_out_of_scope_reply(max_words: int) -> str:
    # Simple default line if model omits assistant_reply for out_of_scope
    base = "I can help with DNV healthcare standards questions only."
    words = base.split()
    return " ".join(words[:max_words])

# ----------------- Public API

def route_greeting_or_ack(user_text: str) -> Optional[RouterDecision]:
    """
    If the message is greeting/ack/out_of_scope, return a RouterDecision with the LLM-authored reply.
    Otherwise return None so the main pipeline continues.
    """
    try:
        raw = _call_router_llm(user_text, _MAX_WORDS)
        logger.info(
            {"event": "router_raw_output", "preview": (raw[:_MAX_LOG] + "...") if len(raw) > _MAX_LOG else raw}
        )

        decision = _parse_router_json(raw)

        # Retry once if unusable or missing keys
        if not decision or decision.get("intent") not in {"greeting", "acknowledgement", "out_of_scope", "other"}:
            raw2 = _call_router_llm(user_text + "\nSTRICT: output MUST be a single JSON object.", _MAX_WORDS)
            logger.info(
                {"event": "router_raw_output_retry", "preview": (raw2[:_MAX_LOG] + "...") if len(raw2) > _MAX_LOG else raw2}
            )
            decision = _parse_router_json(raw2)

        intent = decision.get("intent", "")
        action = decision.get("action", "")
        reply  = decision.get("assistant_reply", "")

        if intent in ("greeting", "acknowledgement"):
            # keep model words, just trim
            if not reply:
                # Absolute last resort (still model-authored, not hardcoded)
                reply = _call_greeting_llm(user_text, _MAX_WORDS)
            words = reply.split()
            if len(words) > _MAX_WORDS:
                reply = " ".join(words[:_MAX_WORDS])
            return RouterDecision(intent=intent, action=(action or "answer_here"), assistant_reply=reply)

        # NEW: minimal out-of-scope handling
        if intent == "out_of_scope":
            final = (reply or _policy_out_of_scope_reply(_MAX_WORDS)).strip()
            words = final.split()
            if len(words) > _MAX_WORDS:
                final = " ".join(words[:_MAX_WORDS])
            return RouterDecision(intent="out_of_scope", action=(action or "answer_here"), assistant_reply=final)

        # If it's clearly small talk but model said "other", craft model-authored greeting
        if _looks_like_small_talk(user_text):
            reply = _call_greeting_llm(user_text, _MAX_WORDS)
            if reply:
                words = reply.split()
                if len(words) > _MAX_WORDS:
                    reply = " ".join(words[:_MAX_WORDS])
                return RouterDecision(intent="greeting", action="answer_here", assistant_reply=reply)

        return None

    except Exception as e:
        logger.warning(f"first_turn_router failed closed (wrapper path): {e}")
        return None
