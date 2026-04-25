from __future__ import annotations
import re, ipaddress
from urllib.parse import urlparse as _std_urlparse
from tldextract import extract as tld_extract

def safe_urlparse(u: str):
    try:
        return _std_urlparse(u)
    except Exception:
        class Dummy:
            scheme = ""; hostname = ""; netloc = ""; path = ""; query = ""
        return Dummy()

def split_url(url: str) -> tuple[str, str]:
    """
    Разбивает URL на (домен, путь). Работает с протоколом и без.
    'egov.kz/cms/ru'         -> ('egov.kz', '/cms/ru')
    'https://egov.kz/cms/ru' -> ('egov.kz', '/cms/ru')
    """
    u = url or ""
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9+\-.]*://', u):
        u = "http://" + u
    p = safe_urlparse(u)
    domain = (p.hostname or "").lower()
    path   = p.path or ""
    if p.query:
        path += "?" + p.query
    return domain, path

SHORT_SVC = {
    'bit','goo','tinyurl','ow','t','is','cli','yfrog','migre','ff','url4',
    'twit','su','snipurl','short','budurl','ping','post','just','bkite',
    'snipr','fic','loopt','doiop','kl','wp','rubyurl','om','to','lnkd',
    'db','qr','adf','bitly','cur','ity','q','po','bc','twitthis','u','j',
    'buzurl','cutt','yourls','x'
}

PATH_SUSPICIOUS = [
    'login','signin','verify','update','secure','confirm','password',
    'credential','download','install','payload','cmd','exec','eval',
    'base64','redirect','wp-admin','phpmyadmin','admin','shell',
    '.exe','.php','.bat','.sh','.js','free','gift','prize','winner',
    'lucky','click','limited','offer','index.php'
]

DOMAIN_SUSPICIOUS = [
    'secure','login','bank','account','update','verify','free',
    'lucky','prize','paypal','amazon','apple','google','microsoft',
    'ebay','netflix','support','help','service'
]

# ── ПРИЗНАКИ ДОМЕНА (стабильные — не меняются при смене пути) ──────────────

def get_domain_features(url: str) -> list[float]:
    """
    10 признаков ТОЛЬКО доменной части.
    egov.kz/cms/ru и egov.kz/login -> ОДИНАКОВЫЙ вектор домена.
    """
    domain, _ = split_url(url)
    u = url or ""
    ext = tld_extract(u)

    is_ip = 0
    try:
        ipaddress.ip_address(domain); is_ip = 1
    except Exception:
        pass

    subdomain_count    = len(ext.subdomain.split('.')) if ext.subdomain else 0
    u_norm = u if re.match(r'^[a-zA-Z]+://', u) else "http://" + u
    is_https           = int(safe_urlparse(u_norm).scheme.lower() == "https")
    suspicious_domain  = sum(1 for w in DOMAIN_SUSPICIOUS if w in domain)
    is_shortened       = int(any(s == (ext.domain or "").lower() for s in SHORT_SVC))

    return [
        len(ext.domain or ""),                          # 1. Длина имени домена
        len(ext.suffix or ""),                          # 2. Длина TLD
        subdomain_count,                                # 3. Кол-во поддоменов
        domain.count('.'),                              # 4. Кол-во точек
        (ext.domain or "").count('-'),                  # 5. Тире в домене
        int(bool(re.search(r'\d', ext.domain or ""))), # 6. Цифры в домене
        is_ip,                                          # 7. IP вместо домена
        is_shortened,                                   # 8. Сервис сокращения
        suspicious_domain,                              # 9. Подозрит. слова в домене
        is_https,                                       # 10. HTTPS
    ]

# ── ПРИЗНАКИ ПУТИ (меняются при смене пути — именно это и нужно) ──────────

def get_path_features(url: str) -> list[float]:
    """
    8 признаков ТОЛЬКО пути/query.
    egov.kz/cms/ru и egov.kz/login -> РАЗНЫЕ векторы пути.
    """
    _, path = split_url(url)

    dangerous_ext = int(bool(re.search(
        r'\.(exe|php|bat|sh|py|js|vbs|cmd|msi|apk|dmg|ps1)(\?|$|#)',
        path.lower()
    )))
    suspicious_path = sum(1 for w in PATH_SUSPICIOUS if w in path.lower())
    has_base64 = int(bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', path)))

    return [
        len(path),                                          # 1. Длина пути
        path.count('/'),                                    # 2. Глубина
        sum(c.isdigit() for c in path),                    # 3. Цифры в пути
        sum(not c.isalnum() and c not in '/-_.' for c in path), # 4. Спецсимволы
        dangerous_ext,                                      # 5. Опасное расширение
        suspicious_path,                                    # 6. Подозрит. слова
        path.count('%'),                                    # 7. % кодирование
        has_base64,                                         # 8. Base64 строки
    ]

# ── ИТОГ: 10 + 8 = 18 признаков ────────────────────────────────────────────

def get_feature_vector(url: str) -> list[float]:
    return get_domain_features(url) + get_path_features(url)


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import os

    BASE_DIR = '../project/csv'
    FILES    = ['train.csv', 'test.csv']

    for file_name in FILES:
        input_path = os.path.join(BASE_DIR, file_name)
        prefix     = file_name.split('.')[0]
        out_path   = os.path.join(BASE_DIR, f'{prefix}_manual_features.npy')

        print(f"\n{'='*45}\nПризнаки: {file_name}\n{'='*45}")
        if not os.path.exists(input_path):
            print(f"❌ Не найден: {input_path}"); continue

        df   = pd.read_csv(input_path)
        urls = df['url'].astype(str).tolist()
        X    = np.array([get_feature_vector(u) for u in tqdm(urls, desc=prefix)],
                        dtype=np.float32)

        print(f"✅ Размерность: {X.shape}  (должно быть [..., 18])")
        np.save(out_path, X)
        print(f"💾 {out_path}")