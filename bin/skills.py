# app/skills.py
from __future__ import annotations
import re, yaml
from pathlib import Path
from typing import Dict, List, Set

WORD = r"[A-Za-z0-9\.\+\#\-]+"  # allows C++, C#, .NET etc.

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()

def load_taxonomy(path: str = "data/skills.yaml") -> Dict[str, List[str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    skill_map: Dict[str, List[str]] = {}
    for item in data.get("skills", []):
        name = item["name"].lower()
        aliases = [a.lower() for a in item.get("aliases", [])]
        skill_map[name] = sorted(set([name] + aliases))
    return skill_map

def compile_patterns(skill_map: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    patterns: Dict[str, List[re.Pattern]] = {}
    for canonical, terms in skill_map.items():
        pats = []
        for t in terms:
            # word-boundary-ish match; ignore case
            pats.append(re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.IGNORECASE))
        patterns[canonical] = pats
    return patterns

def extract_skills(text: str, patterns: Dict[str, List[re.Pattern]]) -> Set[str]:
    text_norm = normalize(text)
    hits: Set[str] = set()
    for canonical, pats in patterns.items():
        if any(p.search(text_norm) for p in pats):
            hits.add(canonical)
    return hits

def diff_skills(jd_text: str, resume_text: str, patterns: Dict[str, List[re.Pattern]]):
    jd = extract_skills(jd_text, patterns)
    cv = extract_skills(resume_text, patterns)
    missing = sorted(jd - cv)
    return {
        "jd_skills": sorted(jd),
        "resume_skills": sorted(cv),
        "missing_skills": missing,
        "extra_skills": sorted(cv - jd),
        "overlap": sorted(cv & jd),
    }
# Example usage:
# skill_map = load_taxonomy("data/skills.yaml")
# patterns = compile_patterns(skill_map)    