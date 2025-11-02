# api.py
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import numpy as np
import hmac, hashlib
import pandas as pd
import os

# заменяем локальные helpers на:
from features import get_feature_vector, registrable_domain

app = Flask(__name__)
EXT_ID = os.environ.get("EXTENSION_ID", "")
EXT_ID = os.environ.get("EXTENSION_ID", "")
if EXT_ID:
    CORS(app, resources={r"/check_url": {"origins": [f"chrome-extension://{EXT_ID}"]}})
else:
    CORS(app)  # dev fallback


# 1) Загрузка вашего “белого” списка (те самые ~1 000 000 доменов)
WHITELIST_PATH = os.environ.get("WHITELIST_PATH", "project/py/whitelist.csv")
try:
    WHITELIST_DF = pd.read_csv(WHITELIST_PATH)
    WHITELIST_SET = set(WHITELIST_DF['domain'].astype(str).str.lower().str.strip())
except Exception as e:
    print(f"[WARN] Can't read whitelist at {WHITELIST_PATH}: {e}")
    WHITELIST_SET = set()

# --- Все остальные вспомогательные функции для фичей ---


SHORT_SVC = set([
    'bit','goo','tinyurl','ow','t','is','cli','yfrog','migre','ff','url4',
    'twit','su','snipurl','short','budurl','ping','post','just','bkite',
    'snipr','fic','loopt','doiop','kl','wp','rubyurl','om','to','lnkd',
    'db','qr','adf','bitly','cur','ity','q','po','bc','twitthis','u','j',
    'buzurl','cutt','yourls','x'
])


# Загрузка обученной модели (из learning.py вы сохранили best_model_v2.sav или v3)
try:
    pipeline = joblib.load('project/py/best_model_v4.sav')
except FileNotFoundError:
    print("Error: best_model_v4.sav not found.")
    exit(1)

API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Переменная окружения API_TOKEN не установлена")

def sign_response(payload: dict) -> str:
    """
    Считает HMAC-SHA256 hex-подпись от JSON-строки payload,
    ключ — API_TOKEN.
    """
    import json
    msg = json.dumps(payload, sort_keys=True).encode()
    return hmac.new(API_TOKEN.encode(), msg, hashlib.sha256).hexdigest()


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
    full_dom = registrable_domain(url)
    if full_dom in WHITELIST_SET:
        # формируем результат и подписываем
        result = {'status': 'benign', 'source': 'whitelist'}
        result['signature'] = sign_response(result)
        return jsonify(result), 200

    # 2) Если домена нет в белом списке — прогоняем через модель
    try:
        vec = np.array(get_feature_vector(url), dtype=np.float32).reshape(1, -1)

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(vec)
            if proba.shape[1] == 2:  # бинарная модель
                pred_int = int((proba[:, 1] >= 0.5)[0])
                status = {0: "benign", 1: "malicious"}.get(pred_int, "unknown")
            else:  # мультикласс (4 класса)
                pred_int = int(proba.argmax(axis=1)[0])
                status = {0: "benign", 1: "defacement", 2: "phishing", 3: "malware"}.get(pred_int, "unknown")
        else:
            pred_int = int(pipeline.predict(vec)[0])
            # по умолчанию считаем бинарную
            status = {0: "benign", 1: "malicious"}.get(pred_int, "unknown")

        result = {'status': status, 'source': 'model'}
        # добавляем подпись именно от этих полей
        result['signature'] = sign_response(result)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)