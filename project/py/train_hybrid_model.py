import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import os
import time

# --- НАСТРОЙКИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_FILE = os.path.join(BASE_DIR, '/Users/alinur/Documents/code for diss/Instagram-Link-Checker/project/py/bert_features.npy')
LABELS_FILE = os.path.join(BASE_DIR, '/Users/alinur/Documents/code for diss/Instagram-Link-Checker/project/py/bert_labels.npy')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'best_ensemble_model.pkl')

print("🚀 Загрузка векторов...")
try:
    X = np.load(FEATURES_FILE)
    y = np.load(LABELS_FILE)
    print(f"✅ Данные: {X.shape}, Метки: {y.shape}")
except FileNotFoundError:
    print("❌ Файлы .npy не найдены.")
    exit()

# 1. Разделение
print("✂️ Разделение на Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Определение моделей
print("⚙️ Инициализация моделей...")

# A. XGBoost
clf_xgb = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=4,
    n_jobs=-1,
    random_state=42
)

# B. LightGBM (Очень быстрый)
clf_lgb = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    objective='multiclass',
    num_class=4,
    n_jobs=-1,
    random_state=42,
    verbosity=-1
)

# C. Random Forest (Для сравнения)
clf_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)

# 3. Цикл обучения и сравнения
models = [
    ('XGBoost', clf_xgb),
    ('LightGBM', clf_lgb),
    ('RandomForest', clf_rf)
]

results = []

print("\n🥊 НАЧАЛО БИТВЫ АЛГОРИТМОВ 🥊")
print("-" * 60)

for name, model in models:
    start_time = time.time()
    print(f"Training {name}...", end=" ")
    
    # Обучение
    model.fit(X_train, y_train)
    
    # Предсказание
    y_pred = model.predict(X_test)
    
    # Метрики
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    elapsed = time.time() - start_time
    
    print(f"✅ Done in {elapsed:.1f}s | Acc: {acc:.4f} | F1: {f1:.4f}")
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'F1-Score': f1,
        'Time (s)': elapsed
    })

# 4. Создание АНСАМБЛЯ (Voting Classifier)
# Мы берем модели, которые уже обучили, и объединяем их голоса
print("-" * 60)
print("🤝 Создание Ансамбля (Voting Classifier)...")

ensemble = VotingClassifier(
    estimators=models,
    voting='soft', # Мягкое голосование (по вероятностям) дает лучший результат
    n_jobs=-1
)

start_time = time.time()
ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)
acc_ens = accuracy_score(y_test, y_pred_ens)
f1_ens = f1_score(y_test, y_pred_ens, average='weighted')
elapsed_ens = time.time() - start_time

print(f"🏆 ENSEMBLE RESULT | Acc: {acc_ens:.4f} | F1: {f1_ens:.4f}")

results.append({
    'Model': 'Ensemble (Hybrid)',
    'Accuracy': acc_ens,
    'F1-Score': f1_ens,
    'Time (s)': elapsed_ens
})

# 5. Вывод итоговой таблицы
results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print("\n📊 ИТОГОВАЯ ТАБЛИЦА (для Диссертации Глава 3):")
print(results_df)

# 6. Сохранение лучшей модели (Ансамбля)
print(f"\n💾 Сохранение ансамбля в {MODEL_SAVE_PATH}...")
joblib.dump(ensemble, MODEL_SAVE_PATH)
print("✅ Готово! Теперь у тебя есть гибридная модель.")