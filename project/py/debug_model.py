import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import warnings

# Отключаем лишние предупреждения
warnings.filterwarnings('ignore')

print("Загрузка модели...")
pipeline = joblib.load("./project/py/best_ensemble_model.pkl")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")

# Тестовые ссылки (УЖЕ ОЧИЩЕННЫЕ)
urls = [
    "google.com", 
    "egov.kz", 
    "yandex.ru", 
    "secure-login-paypal-update-account.com" # Явный фишинг
]

classes = {0: 'BENIGN (Безопасный)', 1: 'PHISHING', 2: 'MALWARE', 3: 'DEFACEMENT'}

print("\n--- РЕЗУЛЬТАТЫ СКАНИРОВАНИЯ ---")
for u in urls:
    # Векторизация
    encoded = tokenizer([u], padding=True, truncation=True, max_length=64, return_tensors='pt')
    with torch.no_grad():
        vec = bert_model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
    
    # Предсказание (ПОДАЕМ ЧИСТЫЙ NUMPY ARRAY)
    pred_int = int(pipeline.predict(vec)[0])
    probs = pipeline.predict_proba(vec)[0]
    
    print(f"\nURL: {u}")
    print(f"Вердикт: {classes.get(pred_int, 'UNKNOWN')} (Класс {pred_int})")
    print(f"Вероятности:")
    print(f" - Безопасный: {probs[0]*100:.1f}%")
    print(f" - Фишинг:     {probs[1]*100:.1f}%")
    print(f" - Вирус:      {probs[2]*100:.1f}%")
    print(f" - Взлом:      {probs[3]*100:.1f}%")