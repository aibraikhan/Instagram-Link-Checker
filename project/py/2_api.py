# 2_api.py
from flask import Flask, json, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import numpy as np
import hmac, hashlib
import os
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
EXT_ID = os.environ.get("EXTENSION_ID", "")
if EXT_ID:
    CORS(app, resources={r"/check_url": {"origins": [f"chrome-extension://{EXT_ID}"]}})
else:
    CORS(app)  # dev fallback

# --- 1. ЗАГРУЗКА ГИБРИДНОЙ МОДЕЛИ ---
MODEL_PATH = "./project/py/best_ensemble_model.pkl" 
try:
    pipeline = joblib.load(MODEL_PATH)
    print("✅ Гибридная модель (Ансамбль) успешно загружена!")
except FileNotFoundError:
    print(f"❌ Ошибка: Модель не найдена по пути {MODEL_PATH}")
    exit(1)

# --- 2. ЗАГРУЗКА DISTILBERT ---
print("⏳ Загрузка DistilBERT для векторизации...")
# Принудительно используем CPU для стабильности вычислений (избегаем бага float32 на Apple MPS)
device = torch.device('cpu') 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
bert_model.eval()
print(f"✅ BERT готов (Устройство: {device})")

API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_TOKEN is not set in the environment variables")

def sign_response(payload: dict) -> str:
    """Подпись ответа с использованием HMAC-SHA256 (надежный строковый метод)"""
    secret = API_TOKEN.encode('utf-8')
    # Склеиваем значения в жестком порядке: "status:detail:source"
    raw_str = f"{payload.get('status', '')}:{payload.get('detail', '')}:{payload.get('source', '')}"
    msg = raw_str.encode('utf-8')
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()

# Основной endpoint для проверки URL
@app.route('/check_url', methods=['POST'])
@cross_origin()
def check_url():
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_TOKEN}":
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(force=True) or {}
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL not provided'}), 400

    # --- ТОЛЬКО ЧЕСТНАЯ ПРОВЕРКА ЧЕРЕЗ НЕЙРОСЕТЬ (Без белых списков) ---
    try:
        # 1. Очистка URL (убираем мусор, чтобы было как в debug_model)
        # Убираем http://, www., а также отрезаем все параметры после ? и #
        clean_url = re.sub(r'^(https?://)?(www\.)?', '', url)
        clean_url = clean_url.split('?')[0].split('#')[0].strip('/')
        print(f"🔍 Проверка чистого URL: {clean_url}")

        # 2. Векторизация (кормим чистый URL)
        encoded = tokenizer(
            [clean_url], padding=True, truncation=True, max_length=64, return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            output = bert_model(**encoded)

        # Получаем чистый вектор
        vec = output.last_hidden_state[:, 0, :].cpu().numpy()

        # 3. Предсказание (ЧИСТЫЙ МАССИВ, НИКАКОГО PANDAS!)
        pred_int = int(pipeline.predict(vec)[0])

        if pred_int == 0:
            status = "benign"
            detail = "safe"
        elif pred_int == 1:
            status = "malicious"
            detail = "phishing"
        elif pred_int == 2:
            status = "malicious"
            detail = "malware"
        elif pred_int == 3:
            status = "malicious"
            detail = "defacement"
        else:
            status = "unknown"
            detail = "unknown"

        result = {
            'status': status, 
            'detail': detail,
            'source': 'hybrid_model'
        }

        # 4. Собираем подпись без пробелов
        raw_str = f"{result['status']}:{result['detail']}:{result['source']}"
        secret = API_TOKEN.encode('utf-8')
        result['signature'] = hmac.new(secret, raw_str.encode('utf-8'), hashlib.sha256).hexdigest()

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Ошибка при обработке {url}: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=False)