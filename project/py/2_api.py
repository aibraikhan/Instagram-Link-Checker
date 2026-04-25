from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import numpy as np
import hmac, hashlib
import os
import torch
import warnings
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from features import get_feature_vector
import tldextract

warnings.filterwarnings('ignore')

app = Flask(__name__)
EXT_ID = os.environ.get("EXTENSION_ID", "")
if EXT_ID:
    CORS(app, resources={r"/check_url": {"origins": [f"chrome-extension://{EXT_ID}"]}})
else:
    CORS(app)

# --- 1. ЗАГРУЗКА МОДЕЛИ И СКЕЙЛЕРА ---
MODEL_PATH  = "../project/brains/best_hybrid_ensemble.pkl"
SCALER_PATH = "../project/brains/manual_feature_scaler.pkl"

try:
    pipeline = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    print("✅ Модель и скейлер загружены!")
except FileNotFoundError as e:
    print(f"❌ Файл не найден: {e}"); exit(1)

# --- 2. MAJESTIC MILLION ---
# Загружается один раз при старте. Хранится в памяти как set — O(1) поиск.
# Путь к файлу: majestic_million.csv, колонка 'Domain'
MAJESTIC_PATH = "../project/csv/whitelist.csv"
majestic_domains: set = set()

try:
    mm = pd.read_csv(MAJESTIC_PATH, usecols=['Domain'])
    majestic_domains = set(mm['Domain'].str.lower().str.strip())
    print(f"✅ Majestic Million: {len(majestic_domains):,} доменов")
except Exception as e:
    print(f"⚠️  Majestic Million не загружен: {e}")

def get_root_domain(url: str) -> str:
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return (ext.domain or "").lower()

# --- 3. BERT ---
print("⏳ Загрузка DistilBERT...")
device     = torch.device('cpu')
tokenizer  = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
bert_model.eval()
print(f"✅ BERT готов")

# --- 4. ТОКЕН ---
API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_TOKEN не задан")

LABEL_MAP = {0: "benign", 1: "phishing", 2: "malware", 3: "defacement"}

def sign_response(payload: dict) -> str:
    raw = f"{payload.get('status','')}:{payload.get('detail','')}:{payload.get('source','')}"
    return hmac.new(API_TOKEN.encode(), raw.encode(), hashlib.sha256).hexdigest()

def bert_embed(text: str) -> np.ndarray:
    enc = tokenizer([text], padding=True, truncation=True,
                    max_length=128, return_tensors='pt').to(device)
    with torch.no_grad():
        vec = bert_model(**enc).last_hidden_state[:, 0, :]
    return vec.cpu().numpy()  # (1, 768)

@app.route('/check_url', methods=['POST'])
@cross_origin()
def check_url():
    if request.headers.get("Authorization", "") != f"Bearer {API_TOKEN}":
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(force=True) or {}
    url  = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL not provided'}), 400

    if url.startswith('chrome-extension') or url.endswith('.pkg'):
        result = {'status': 'benign', 'detail': 'internal', 'source': 'system'}
        result['signature'] = sign_response(result)
        return jsonify(result), 200

    try:
        # ── УРОВЕНЬ 1: Majestic Million (репутация домена) ───────────────────
        root = get_root_domain(url)
        if root and root in majestic_domains:
            print(f"🌐 Majestic hit: {root} → benign")
            result = {'status': 'benign', 'detail': 'safe', 'source': 'majestic_million'}
            result['signature'] = sign_response(result)
            return jsonify(result), 200

        # ── УРОВЕНЬ 2: ML-классификатор ──────────────────────────────────────
        vec_bert          = bert_embed(url)                                      # (1, 768)
        vec_manual_raw    = np.array(get_feature_vector(url), dtype=np.float32).reshape(1, -1)
        vec_manual_scaled = scaler.transform(vec_manual_raw)                     # (1, 18)
        vec_final         = np.hstack((vec_bert, vec_manual_scaled))             # (1, 786)

        pred_int = int(pipeline.predict(vec_final)[0])
        status   = LABEL_MAP.get(pred_int, "unknown")
        detail   = status if status != "benign" else "safe"

        print(f"🧠 ML: {pred_int} → {status} | {url}")

        result = {'status': status, 'detail': detail, 'source': 'hybrid_model'}
        result['signature'] = sign_response(result)
        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Ошибка {url}: {e}")
        return jsonify({'error': str(e), 'status': 'unknown'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=False)