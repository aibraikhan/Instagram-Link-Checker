# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import numpy as np
import re
import string
import hmac, hashlib
from urllib.parse import urlparse
from tldextract import extract as tld_extract
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/check_url": {"origins": ["chrome-extension://oaimmjgfajelaekcjlcpeedpcbpbchod"]}})


# 1) Загрузка вашего “белого” списка (те самые ~1 000 000 доменов)
WHITELIST_DF = pd.read_csv('whitelist.csv')
WHITELIST_SET = set(WHITELIST_DF['domain'].str.lower().str.strip())

def extract_full_domain(url: str) -> str:
    """
    Возвращает просто <domain>.<suffix>, например:
      "https://sub.example.com/path"  → "example.com"
    """
    try:
        ext = tld_extract(url)
        return f"{ext.domain}.{ext.suffix}".lower()
    except:
        return ''

# --- Все остальные вспомогательные функции для фичей ---

def get_url_length(url: str) -> int:
    url = re.sub(r'^https?://', '', url)
    url = url.replace('www.', '')
    return len(url)

def extract_netloc(url: str) -> str:
    return urlparse(url).netloc or ''

def count_letters(url: str) -> int:
    return sum(c.isalpha() for c in url)

def count_digits(url: str) -> int:
    return sum(c.isdigit() for c in url)

def count_special_chars(url: str) -> int:
    return sum(c in string.punctuation for c in url)

SHORT_SVC = set([
    'bit','goo','tinyurl','ow','t','is','cli','yfrog','migre','ff','url4',
    'twit','su','snipurl','short','budurl','ping','post','just','bkite',
    'snipr','fic','loopt','doiop','kl','wp','rubyurl','om','to','lnkd',
    'db','qr','adf','bitly','cur','ity','q','po','bc','twitthis','u','j',
    'buzurl','cutt','yourls','x'
])

def has_shortening_service(url: str) -> int:
    m = re.search(r'https?://(?:www\.)?(?:[\w-]+\.)*([\w-]+)\.', url)
    if not m:
        return 0
    domain = m.group(1).lower()
    return int(domain in SHORT_SVC)

def abnormal_url(url: str) -> int:
    h = urlparse(url).netloc or ''
    return int(bool(h and h not in url))

def secure_http(url: str) -> int:
    return int(urlparse(url).scheme.lower() == 'https')

def have_ip_address(url: str) -> int:
    h = urlparse(url).hostname or ''
    try:
        import ipaddress
        ipaddress.ip_address(h)
        return 1
    except:
        return 0

def extract_root_domain(url: str) -> str:
    e = tld_extract(url)
    return e.domain or ''

def get_url_region(root_domain: str) -> str:
    parts = root_domain.split('.')
    return parts[-1] if parts else ''

def hash_encode(s: str, mod: int = 10**8) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod

def get_numerical_values(url: str) -> dict:
    """
    Собирает все необходимые числовые фичи, кроме domain_similarity.
    """
    u = url.lower().strip().replace('www.', '')
    root = extract_root_domain(u)
    region = get_url_region(root)

    return {
        'url_len': get_url_length(u),
        'letters_count': count_letters(u),
        'digits_count': count_digits(u),
        'special_chars_count': count_special_chars(u),
        'shortened': has_shortening_service(u),
        'abnormal_url': abnormal_url(u),
        'secure_http': secure_http(u),
        'have_ip': have_ip_address(u),
        'url_region': hash_encode(region),
        'netloc_hash': hash_encode(extract_netloc(u)),
        # больше не нужен 'domain_similarity'
    }

# Загрузка обученной модели (из learning.py вы сохранили best_model_v2.sav или v3)
try:
    pipeline = joblib.load('best_model_v2.sav')
except FileNotFoundError:
    print("Error: best_model_v2.sav not found.")
    exit(1)

API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Переменная API_TOKEN не задана в окружении")

def sign_response(payload: dict) -> str:
    """
    Считает HMAC-SHA256 hex-подпись от JSON-строки payload,
    ключ — API_TOKEN.
    """
    import json, os
    secret = os.environ['API_TOKEN'].encode()
    msg    = json.dumps(payload, sort_keys=True).encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


# --- Основной endpoint ---
@app.route('/check_url', methods=['POST'])
@cross_origin()
def check_url():
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_TOKEN}":
        return jsonify({"error":"unauthorized"}), 401

    data = request.get_json(force=True) or {}
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL not provided'}), 400


    # 1) Сначала проверка по whitelist
    full_dom = extract_full_domain(url)
    if full_dom in WHITELIST_SET:
        # формируем результат и подписываем
        result = {'status': 'benign', 'source': 'whitelist'}
        result['signature'] = sign_response(result)
        return jsonify(result), 200

    # 2) Если домена нет в белом списке — прогоняем через модель
    try:
        feats = get_numerical_values(url)
        arr = np.array(list(feats.values())).reshape(1, -1)
        pred_int = pipeline.predict(arr)[0]
        label_map = {0: 'benign', 1: 'defacement', 2: 'phishing', 3: 'malware'}
        status = label_map.get(int(pred_int), 'unknown')
        result = {'status': status, 'source': 'model'}
        # добавляем подпись именно от этих полей
        result['signature'] = sign_response(result)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)