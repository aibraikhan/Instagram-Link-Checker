
from __future__ import annotations
import re, ipaddress
from difflib import SequenceMatcher
from urllib.parse import urlparse as _std_urlparse
from tldextract import extract as tld_extract

def safe_urlparse(u: str):
    """
    Обёртка над urlparse, чтобы не падать на кривых строках.
    Возвращаем объект с полями .scheme, .hostname, .netloc,
    даже если строка была невалидной.
    """
    try:
        return _std_urlparse(u)
    except Exception:
        # вернём пустые поля, чтобы последующий код не умер
        class Dummy:
            scheme = ""
            hostname = ""
            netloc = ""
        return Dummy()

# топ укоротителей из твоего 2_api.py
SHORT_SVC = {
    'bit','goo','tinyurl','ow','t','is','cli','yfrog','migre','ff','url4',
    'twit','su','snipurl','short','budurl','ping','post','just','bkite',
    'snipr','fic','loopt','doiop','kl','wp','rubyurl','om','to','lnkd',
    'db','qr','adf','bitly','cur','ity','q','po','bc','twitthis','u','j',
    'buzurl','cutt','yourls','x'
}

def has_shortening_service(url: str) -> int:
    m = re.search(r'https?://(?:www\.)?(?:[\w-]+\.)*([\w-]+)\.', url or "")
    if not m:
        return 0
    return int(m.group(1).lower() in SHORT_SVC)

def abnormal_url(url: str) -> int:
    h = safe_urlparse(url or "").netloc or ""
    return int(bool(h and h not in (url or "")))

def secure_http(url: str) -> int:
    return int((safe_urlparse(url or "").scheme or "").lower() == "https")

def have_ip_address(url: str) -> int:
    h = safe_urlparse(url or "").hostname or ""
    try:
        ipaddress.ip_address(h)
        return 1
    except Exception:
        return 0

def url_len(url: str) -> int:
    return len((url or "").replace("http://","").replace("https://",""))

def count_chars(url: str):
    s = url or ""
    letters = sum(c.isalpha() for c in s)
    digits  = sum(c.isdigit() for c in s)
    specials= sum(not (c.isalnum()) for c in s)
    return letters, digits, specials

def registrable_domain(url: str) -> str:
    ext = tld_extract(url or "")
    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower() if ext.domain else ext.suffix.lower()
    return (ext.domain or "unknown").lower()

def tld_length(url: str) -> int:
    ext = tld_extract(url or "")
    return len(ext.suffix or "")

def url_region_hash(url: str) -> int:
    # как в testing/train_full_model.py: простая детерминированная хеш-фича корневого домена
    ext = tld_extract(url or "")
    parts = [p for p in [ext.domain, ext.suffix] if p]
    root = ".".join(parts) if parts else (ext.domain or "")
    h = 0
    for ch in root:
        h = ((h << 5) - h) + ord(ch)
        h &= 0xFFFFFFFF
    return abs(h) % 100000000

def netloc_hash(url: str) -> int:
    host = safe_urlparse(url or "").hostname or ""
    h = 0
    for ch in host:
        h = ((h << 5) - h) + ord(ch)
        h &= 0xFFFFFFFF
    return abs(h) % 100000000

def get_feature_vector(url: str) -> list[float]:
    """Возвращает фич-вектор (ровно 12 признаков) в фиксированном порядке."""
    u = url or ""
    p = safe_urlparse(u)

    L = sum(c.isalpha() for c in u)
    D = sum(c.isdigit() for c in u)
    S = sum(not c.isalnum() for c in u)

    return [
        url_len(url),              # 1
        L,                         # 2
        D,                         # 3
        S,                         # 4
        has_shortening_service(url),#5
        abnormal_url(url),         # 6
        secure_http(url),          # 7
        have_ip_address(url),      # 8
        tld_length(url),           # 9
        url_region_hash(url),      # 10
        netloc_hash(url),          # 11
        # бонус-фича: кол-во точек в хосте (стабильно и дёшево)
        (p.hostname or "").count("."),   # кол-во точек в хосте, # 12
    ]
