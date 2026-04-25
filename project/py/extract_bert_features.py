import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

BASE_DIR   = '../project/csv'
FILES      = ['train.csv', 'test.csv']
BATCH_SIZE = 64
MAX_LEN    = 128   # 128 > 64 — покрывает длинные URL

# ── Устройство ────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps"); print("🚀 MPS (M2)")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("🚀 CUDA")
else:
    device = torch.device("cpu");  print("🐢 CPU")

tokenizer  = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
bert_model.eval()

def embed(texts: list[str]) -> np.ndarray:
    """CLS-эмбеддинг для списка строк → (N, 768)."""
    out = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="BERT"):
        batch = texts[i : i + BATCH_SIZE]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=MAX_LEN, return_tensors='pt').to(device)
        with torch.no_grad():
            h = bert_model(**enc).last_hidden_state[:, 0, :]
        out.append(h.cpu().numpy())
    return np.vstack(out)

label_mapping = {'benign': 0, 'phishing': 1, 'malware': 2, 'defacement': 3}

for file_name in FILES:
    path_in = os.path.join(BASE_DIR, file_name)
    prefix  = file_name.split('.')[0]

    print(f"\n{'='*50}\n{file_name}\n{'='*50}")
    df   = pd.read_csv(path_in)
    urls = df['url'].astype(str).tolist()
    y    = df['type'].map(label_mapping).fillna(1).astype(int).values

    print(f"  URL-ов: {len(urls):,}")
    print(f"  Пример: '{urls[0]}'")

    # Полный URL целиком → один вектор 768
    E = embed(urls)
    print(f"  ✅ BERT матрица: {E.shape}  (должно быть [..., 768])")

    np.save(os.path.join(BASE_DIR, f'{prefix}_bert_features.npy'), E)
    np.save(os.path.join(BASE_DIR, f'{prefix}_bert_labels.npy'),   y)
    print(f"  💾 Сохранено")