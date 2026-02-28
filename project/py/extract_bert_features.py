import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# --- НАСТРОЙКИ ---
INPUT_FILE = '/Users/alinur/Documents/code for diss/Instagram-Link-Checker/project/py/dataset_final_augmented.csv'
BASE_DIR = os.path.dirname(INPUT_FILE)
OUTPUT_FEATURES = os.path.join(BASE_DIR, 'bert_features.npy')
OUTPUT_LABELS = os.path.join(BASE_DIR, 'bert_labels.npy')

# Для M2 16GB оптимально 64 или 128. 
# Если начнется перегрев (Air без вентилятора), скрипт сам не остановится, но 64 безопаснее.
BATCH_SIZE = 64  
MAX_LEN = 64      

# --- 1. ЗАГРУЗКА ДАННЫХ ---
print(f"Загрузка полного датасета: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# ВАЖНО: Убрали sample(1000), теперь работаем с полным набором
# df = df.sample(1000) 

# Автопоиск колонки с метками
target_col = None
possible_names = ['type', 'label', 'class', 'url_type']
for name in possible_names:
    if name in df.columns:
        target_col = name
        break

if target_col is None:
    print(f"❌ ОШИБКА: Не нашел колонку меток. Доступные: {df.columns.tolist()}")
    exit()

urls = df['url'].astype(str).tolist()

# Маппинг (как в диссертации)
label_mapping = {'benign': 0, 'phishing': 1, 'malware': 2, 'defacement': 3}
y = df[target_col].map(label_mapping).fillna(1).astype(int).values 

print(f"Всего URL для обработки: {len(urls)}")

# --- 2. ИНИЦИАЛИЗАЦИЯ BERT НА M2 (MPS) ---
print("Загрузка модели DistilBERT...")

# Проверяем доступность Metal (GPU на Mac)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 РЕЖИМ M2 АКТИВИРОВАН: Используем GPU (MPS)")
else:
    device = torch.device("cpu")
    print("🐢 РЕЖИМ CPU: MPS недоступен, будет медленно")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

# --- 3. ФУНКЦИЯ ---
def get_bert_embeddings(url_list):
    model.eval()
    all_embeddings = []
    
    # tqdm покажет прогресс бар и примерное время окончания
    for i in tqdm(range(0, len(url_list), BATCH_SIZE), desc="Векторизация"):
        batch = url_list[i : i + BATCH_SIZE]
        
        encoded = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=MAX_LEN, 
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            output = model(**encoded)
        
        # Переносим с GPU на CPU для сохранения
        embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
        
    return np.vstack(all_embeddings)

# --- 4. ЗАПУСК ---
try:
    print("Начинаем процесс...")
    X_features = get_bert_embeddings(urls)
    
    print(f"✅ Готово! Размерность матрицы: {X_features.shape}")
    
    np.save(OUTPUT_FEATURES, X_features)
    np.save(OUTPUT_LABELS, y)
    print(f"Файлы сохранены в папке: {BASE_DIR}")

except KeyboardInterrupt:
    print("\nПроцесс остановлен.")