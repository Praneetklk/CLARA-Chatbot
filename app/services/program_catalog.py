# services/program_catalog.py
from __future__ import annotations
import re
from typing import Dict, List, Tuple, Literal, Optional, TypedDict

# =========================
# Canonical identifiers
# =========================

# Domains = high-level buckets
DOMAIN_CODES: List[str] = [
    "CARDIAC_VASCULAR",
    "INFECTION",
    "NIAHO",
    "STROKE",
    # add others (ORTHO_SPINE, GLYCEMIC, etc.) when you’re ready
]

# Programs = concrete, targetable units
# PROGRAM_CODES: List[str] = [
#     "ACPC",  # Advanced Chest Pain Center
#     "HFAC",  # Heart Failure Accreditation/Certification
#     "ACIP",  # Acute Coronary/Cardiac Intervention Program
#     "ASPC",  # Ambulatory Surgery Program Certification (temporary domain assignment below)
#     "CIP",   # Cardiac/Coronary Intervention Program
#     "CAH",   # Critical Access Hospital (NIAHO)
#     "PSY",   # Psychiatric Hospital (NIAHO)
#     "ASR",   # Acute Stroke Ready
#     "CSC",   # Comprehensive Stroke Center
#     "STROKE-PPSC",  # Primary Plus Stroke Center
#     "PSC",   # Primary Stroke Center
# ]

PROGRAM_CODES: List[str] = [
    "HOSP",   
    "CAH",   # Critical Access Hospital (NIAHO)
    "PSY",   # Psychiatric Hospital (NIAHO)
]

# Domain → Programs mapping
DOMAIN_TO_PROGRAMS: Dict[str, List[str]] = {
    "CARDIAC_VASCULAR": ["ACPC", "HFAC"],   # expand as needed
    "INFECTION": ["ASPC","ACIP","CIP"],
    "STROKE": ["ASR","CSC","PPSC","PSC"],  
    "NIAHO": ["CAH", "PSY"]                     # fill when you add stroke programs
} 

# =========================
# Aliases (user phrases → canonical)
# =========================

# Programs
PROGRAM_ALIASES: Dict[str, str] = {
    # CAH — Critical Access Hospital (NIAHO)
    "cah": "CAH",
    "c a h": "CAH",
    "critical access hospital": "CAH",
    "critical access hospitals": "CAH",
    "Critical-Access Hospital" : "CAH",
    "CAH program / CAH accreditation": "CAH",
    "Critical Access Hospital program" : "CAH",
    "critical access health": "CAH",
    "niaho cah": "CAH",

    # PSY — Psychiatric Hospital (NIAHO)
    "psy": "PSY",
    "p s y": "PSY",
    "psychiatric hospital": "PSY",
    "psych hospital": "PSY",
    "niaho psy": "PSY",

    # HAP - Hospital (NIAHO)
    "hap": "HOSP",
    "hosp": "HOSP",
    "hospital" : "HOSP",
    "naiho hospital": "HOSP",
    "acute care hospital": "HOSP",

}

# PROGRAM_ALIASES: Dict[str, str] = {
#     # ACPC — Advanced Chest Pain Center
#     "acpc": "ACPC",
#     "a c p c": "ACPC",
#     "advanced chest pain center": "ACPC",
#     "advanced chest pain centre": "ACPC",
#     "advanced chest pain certification": "ACPC",
#     "advanced chest pain programme": "ACPC",
#     "advanced chest pain program": "ACPC",
#     "chest pain center advanced": "ACPC",
#     "advanced cpc": "ACPC",

#     # HFAC — Heart Failure Accreditation/Certification
#     "hfac": "HFAC",
#     "h f a c": "HFAC",
#     "heart failure accreditation": "HFAC",
#     "heart failure certification": "HFAC",
#     "heart failure program": "HFAC",
#     "heart failure centre": "HFAC",
#     "heart failure center": "HFAC",

#     # ACIP — Acute Coronary/Cardiac Intervention Program
#     "acip": "ACIP",
#     "a c i p": "ACIP",
#     "acute coronary intervention program": "ACIP",
#     "acute cardiac intervention program": "ACIP",
#     "acute coronary intervention": "ACIP",
#     "acute cardiac intervention": "ACIP",
#     "coronary intervention program": "ACIP",

#     # ASPC — Ambulatory Surgery Program Certification
#     "aspc": "ASPC",
#     "a s p c": "ASPC",
#     "ambulatory surgery program certification": "ASPC",
#     "ambulatory surgical program certification": "ASPC",
#     "ambulatory surgery certification": "ASPC",
#     "ambulatory surgery program": "ASPC",

#     # CIP — Cardiac/Coronary Intervention Program
#     "cip": "CIP",
#     "c i p": "CIP",
#     "cardiac intervention program": "CIP",
#     "coronary intervention program": "CIP",
#     "interventional cardiology program": "CIP",
#     "cath lab intervention program": "CIP",

#     # CAH — Critical Access Hospital (NIAHO)
#     "cah": "CAH",
#     "c a h": "CAH",
#     "critical access hospital": "CAH",
#     "niaho cah": "CAH",

#     # PSY — Psychiatric Hospital (NIAHO)
#     "psy": "PSY",
#     "p s y": "PSY",
#     "psychiatric hospital": "PSY",
#     "psych hospital": "PSY",
#     "niaho psy": "PSY",

#     # ASR — Acute Stroke Ready
#     "asr": "ASR",
#     "a s r": "ASR",
#     "acute stroke ready": "ASR",
#     "acute stroke ready hospital": "ASR",
#     "acute stroke ready center": "ASR",
#     "acute stroke ready centre": "ASR",

#     # CSC — Comprehensive Stroke Center
#     "csc": "CSC",
#     "c s c": "CSC",
#     "comprehensive stroke center": "CSC",
#     "comprehensive stroke centre": "CSC",
#     "comprehensive stroke certification": "CSC",

#     # PPSC — Primary Plus Stroke Center
#     "ppsc": "PPSC",
#     "p p s c": "PPSC",
#     "primary plus stroke center": "PPSC",
#     "primary plus stroke centre": "PPSC",
#     "primary+ stroke center": "PPSC",
#     "primary plus stroke certification": "PPSC",
#     "stroke ppsc": "STROKE-PPSC",

#     # PSC — Primary Stroke Center
#     "psc": "PSC",
#     "p s c": "PSC",
#     "primary stroke center": "PSC",
#     "primary stroke centre": "PSC",
#     "primary stroke certification": "PSC",
# }

# Domains
DOMAIN_ALIASES: Dict[str, str] = {
    # Cardiac & Vascular domain
    "cardiac vascular": "CARDIAC_VASCULAR",
    "cardiac-vascular": "CARDIAC_VASCULAR",
    "cardiac & vascular": "CARDIAC_VASCULAR",
    "cardiac and vascular": "CARDIAC_VASCULAR",
    "cardiovascular": "CARDIAC_VASCULAR",
    "cardio vascular": "CARDIAC_VASCULAR",
    "cv": "CARDIAC_VASCULAR",
    "heart & vascular": "CARDIAC_VASCULAR",
    "heart and vascular": "CARDIAC_VASCULAR",
    "heart/vascular": "CARDIAC_VASCULAR",
    "cardiac programs": "CARDIAC_VASCULAR",
    "cardiac services": "CARDIAC_VASCULAR",
    "cardiac domain": "CARDIAC_VASCULAR",
    "cardio domain": "CARDIAC_VASCULAR",
    # intent-ish phrases that still imply the domain
    "chest pain": "CARDIAC_VASCULAR",
    "heart failure": "CARDIAC_VASCULAR",
    "cardiac intervention": "CARDIAC_VASCULAR",
    # frequent typos
    "cardiac vasular": "CARDIAC_VASCULAR",
    "cardic vascular": "CARDIAC_VASCULAR",
    "cardiovacular": "CARDIAC_VASCULAR",

    # Stroke domain
    "stroke": "STROKE",
    "stroke center": "STROKE",
    "stroke programme": "STROKE",
    "stroke program": "STROKE",
    "stroke services": "STROKE",
    "stroke certification": "STROKE",
    "stroke care": "STROKE",
    "stroke domain": "STROKE",
    "stroke ready": "STROKE",
    "primary stroke": "STROKE",
    "primary plus stroke": "STROKE",
    "comprehensive stroke": "STROKE",

    # NIAHO domain (add a domain catch separate from program names)
    "niaho": "NIAHO",
    "niaho standards": "NIAHO",
    "hospital accreditation": "NIAHO",
    "niaho hospital": "NIAHO",
    "niaho programs": "NIAHO",

    # INFECTION domain (add a domain catch separate from program names)
    "INFECTION CONTROL": "INFECTION",
    "infection control": "INFECTION",
    "infection": "INFECTION",
}

# =========================
# Normalization
# =========================

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[_\-./]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_ALIAS_PROGRAM_INDEX: Dict[str, str] = { _normalize(k): v for k, v in PROGRAM_ALIASES.items() }
_ALIAS_DOMAIN_INDEX:  Dict[str, str] = { _normalize(k): v for k, v in DOMAIN_ALIASES.items() }

def _contains_alias(norm_text: str, alias_norm: str) -> bool:
    """
    Safer alias match using word boundaries to avoid false hits like 'cahsee' -> 'cah'.
    Assumes both inputs are normalized via _normalize().
    """
    pat = re.compile(rf"\b{re.escape(alias_norm)}\b")
    return bool(pat.search(norm_text))

# =========================
# Chapters (QM.1, SR.1.2, etc.)
# =========================

# Accept "QM.1", "qm 1", "QM.01", "SR.1.2"
_CHAPTER_RE = re.compile(
    r"\b([A-Z]{2,4})\s*\.?\s*(\d{1,2})(?:\.(\d{1,2}))?\b",
    re.IGNORECASE
)

def extract_chapter_code(text: str) -> Optional[str]:
    m = _CHAPTER_RE.search(text)
    if not m:
        return None
    part1 = m.group(1).upper()
    part2 = str(int(m.group(2)))  # drop leading zeros
    part3 = m.group(3)
    if part3 is None:
        return f"{part1}.{part2}"
    return f"{part1}.{part2}.{int(part3)}"

# =========================
# Resolution helpers
# =========================

class ProgramResolution(TypedDict, total=False):
    kind: Literal["program", "domain"]
    program: str               # when kind == "program"
    domain: str                # when kind == "domain"
    candidates: List[str]      # suggested child programs if domain only
    alias_hit: str             # normalized alias that matched
    chapter_code: str          # e.g., "QM.1" or "SR.1.2" if present

def resolve_program_or_domain(user_text: str) -> Optional[ProgramResolution]:
    """
    Try to resolve a concrete PROGRAM first; else resolve a DOMAIN and suggest its programs.
    Returns None if nothing recognizable is found.
    """
    norm = _normalize(user_text)
    #norm = user_text
    chapter = extract_chapter_code(user_text) or None

    # 1) Exact program alias hit
    for alias_norm, code in _ALIAS_PROGRAM_INDEX.items():
        if _contains_alias(norm, alias_norm):
            if chapter:
                return {"kind": "program", "program": code, "alias_hit": alias_norm, "chapter_code": chapter}
            return {"kind": "program", "program": code, "alias_hit": alias_norm}

    # 2) Domain alias hit → return domain with candidate programs
    for alias_norm, domain in _ALIAS_PROGRAM_INDEX.items():
        if _contains_alias(norm, alias_norm):
            candidates = PROGRAM_ALIASES.get(domain, [])
            res: ProgramResolution = {
                "kind": "domain",
                "domain": domain,
                "alias_hit": alias_norm,
                "candidates": candidates
            }
            if chapter:
                res["chapter_code"] = chapter
            return res

    return None

# Convenience: quick scans (optional debugging/telemetry)
def scan_text_for_any_alias(user_text: str) -> List[Tuple[str, str]]:
    """Return [(alias_hit, canonical)] across both programs and domains."""
    norm = _normalize(user_text)
    hits: List[Tuple[str, str]] = []
    for a_norm, c in _ALIAS_PROGRAM_INDEX.items():
        if _contains_alias(norm, a_norm):
            hits.append((a_norm, c))
    for a_norm, c in _ALIAS_DOMAIN_INDEX.items():
        if _contains_alias(norm, a_norm):
            hits.append((a_norm, c))
    return hits
