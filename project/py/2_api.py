# 2_api.py
from flask import Flask, json, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import numpy as np
import hmac, hashlib
import pandas as pd
import os
from features import get_feature_vector, registrable_domain

app = Flask(__name__)
EXT_ID = os.environ.get("EXTENSION_ID", "")
if EXT_ID:
    CORS(app, resources={r"/check_url": {"origins": [f"chrome-extension://{EXT_ID}"]}})
else:
    CORS(app)  # dev fallback

# Загрузка белого списка (whitelist)
WHITELIST_PATH = os.environ.get("WHITELIST_PATH", "project/py/whitelist.csv")
try:
    WHITELIST_DF = pd.read_csv(WHITELIST_PATH)
    WHITELIST_SET = set(WHITELIST_DF['domain'].astype(str).str.lower().str.strip())
except Exception as e:
    print(f"[WARN] Can't read whitelist at {WHITELIST_PATH}: {e}")
    WHITELIST_SET = set()

# Загружаем уже обученную модель
MODEL_PATH = "project/py/best_model_v4.sav"
try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}")
    exit(1)

API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_TOKEN is not set in the environment variables")

def sign_response(payload: dict) -> str:
    """Подпись ответа с использованием HMAC-SHA256"""
    secret = API_TOKEN.encode()
    msg = json.dumps(payload, sort_keys=True).encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()

# Основной endpoint для проверки URL
@app.route('/check_url', methods=['POST'])
@cross_origin()
def check_url():
    # Получаем данные из запроса
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_TOKEN}":
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(force=True) or {}
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL not provided'}), 400

    # 1) Сначала проверка по whitelist
    full_dom = registrable_domain(url)
    if full_dom in WHITELIST_SET:
        # Формируем результат и подписываем
        result = {'status': 'benign', 'source': 'whitelist'}
        result['signature'] = sign_response(result)
        return jsonify(result), 200

    # 2) Если домена нет в белом списке — прогоняем через модель
    try:
        vec = np.array(get_feature_vector(url), dtype=np.float32).reshape(1, -1)

        # Прогоняем через модель (не обучаем её снова)
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
            status = {0: "benign", 1: "malicious"}.get(pred_int, "unknown")

        result = {'status': status, 'source': 'model'}
        result['signature'] = sign_response(result)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
