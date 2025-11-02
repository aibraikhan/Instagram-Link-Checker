# utils/domain_groups.py
from __future__ import annotations
import tldextract

def registrable_domain(url: str) -> str:
    ext = tldextract.extract(str(url))
    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}" if ext.domain else ext.suffix
    return ext.domain or "unknown"
