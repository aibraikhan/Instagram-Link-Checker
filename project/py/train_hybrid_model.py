import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import os
import time

# --- НАСТРОЙКИ ---
BASE_DIR = './project/py'
MODEL_SAVE_PATH  = os.path.join(BASE_DIR, 'best_hybrid_ensemble.pkl')
SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'manual_feature_scaler.pkl')

CLASS_NAMES = ['benign', 'phishing', 'malware', 'defacement']

print("🚀 1. Загрузка векторов BERT и ручных признаков...")
try:
    X_train_bert   = np.load(os.path.join(BASE_DIR, 'train_bert_features.npy'))
    X_train_manual = np.load(os.path.join(BASE_DIR, 'train_manual_features.npy'))
    y_train        = np.load(os.path.join(BASE_DIR, 'train_bert_labels.npy'))

    X_test_bert    = np.load(os.path.join(BASE_DIR, 'test_bert_features.npy'))
    X_test_manual  = np.load(os.path.join(BASE_DIR, 'test_manual_features.npy'))
    y_test         = np.load(os.path.join(BASE_DIR, 'test_bert_labels.npy'))
except FileNotFoundError as e:
    print(f"❌ Ошибка: Не найден файл {e.filename}")
    exit()

# ── ИСПРАВЛЕНО: нормализация ручных признаков ────────────────────────────────
# Проблема: BERT-признаки в диапазоне ~[-1, 1], а ручные признаки
# (длина URL, кол-во символов) могут быть 0..500+.
# Без нормализации Random Forest строит сплиты по большим числам и
# фактически игнорирует BERT-эмбеддинги в первых ветках деревьев.
# StandardScaler приводит каждый признак к mean=0, std=1.
# Важно: fit только на train, transform на обоих — иначе data leakage.
print("\n📐 2. Нормализация ручных признаков (StandardScaler)...")
scaler = StandardScaler()
X_train_manual_scaled = scaler.fit_transform(X_train_manual)
X_test_manual_scaled  = scaler.transform(X_test_manual)          # только transform!
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"   💾 Скейлер сохранён: {SCALER_SAVE_PATH}")

print("\n🔗 3. Слияние признаков (768 BERT + 18 manual = 786)...")
X_train = np.hstack((X_train_bert, X_train_manual_scaled))
X_test  = np.hstack((X_test_bert,  X_test_manual_scaled))
print(f"   Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Веса классов для борьбы с дисбалансом ───────────────────────────────────
# malware — всего 4.4% датасета, без весов модели игнорируют редкий класс
counts      = np.bincount(y_train)
total       = len(y_train)
n_classes   = len(counts)
class_weight_dict = {i: total / (n_classes * c) for i, c in enumerate(counts)}
print(f"\n⚖️  Веса классов: { {CLASS_NAMES[i]: f'{w:.2f}' for i,w in class_weight_dict.items()} }")

# --- ОПРЕДЕЛЕНИЕ МОДЕЛЕЙ ---
print("\n⚙️ 4. Инициализация алгоритмов...")

clf_xgb = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=4,
    n_jobs=-1,
    random_state=42,
    # ИСПРАВЛЕНО: передаём веса классов через sample_weight в fit()
    # XGBoost принимает веса через параметр при обучении, не здесь
)

clf_lgb = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    objective='multiclass',
    num_class=4,
    n_jobs=-1,
    random_state=42,
    verbosity=-1,
    class_weight=class_weight_dict,
)

clf_rf = RandomForestClassifier(
    n_estimators=300,       # УЛУЧШЕНО: было 200
    max_depth=15,           # УЛУЧШЕНО: было 12
    n_jobs=-1,
    random_state=42,
    class_weight=class_weight_dict,   # ИСПРАВЛЕНО: добавлен вес классов
)

# Веса для XGBoost через sample_weight (передаётся в fit)
sample_weights = np.array([class_weight_dict[y] for y in y_train])

models = [
    ('XGBoost',      clf_xgb),
    ('LightGBM',     clf_lgb),
    ('RandomForest', clf_rf),
]

results = []

print("\n🥊 НАЧАЛО ОБУЧЕНИЯ МОДЕЛЕЙ")
print("-" * 65)

for name, model in models:
    t0 = time.time()
    print(f"Обучение {name}...", end=" ", flush=True)

    if name == 'XGBoost':
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc      = accuracy_score(y_test, y_pred)
    f1_w     = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')    # ДОБАВЛЕНО: macro F1
    elapsed  = time.time() - t0

    print(f"✅ {elapsed:.1f}s | Acc: {acc:.4f} | F1-weighted: {f1_w:.4f} | F1-macro: {f1_macro:.4f}")

    results.append({
        'Model':        name,
        'Accuracy':     round(acc, 4),
        'F1-weighted':  round(f1_w, 4),
        'F1-macro':     round(f1_macro, 4),
        'Time (s)':     round(elapsed, 1),
    })

# --- АНСАМБЛЬ ---
print("-" * 65)
print("🤝 5. Создание Гибридного Ансамбля (Soft Voting)...")

# Используем уже обученные модели напрямую через fitted_estimators_.
# VotingClassifier.fit() всегда переобучает модели внутри заново —
# чтобы этого избежать, устанавливаем estimators_ вручную.
ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=1)
ensemble.estimators_ = [clf_xgb, clf_lgb, clf_rf]   # уже обученные!
ensemble.le_ = __import__('sklearn.preprocessing', fromlist=['LabelEncoder']).LabelEncoder()
ensemble.le_.fit(y_train)
ensemble.named_estimators_ = dict(zip(['XGBoost','LightGBM','RandomForest'],
                                       [clf_xgb, clf_lgb, clf_rf]))

t0 = time.time()
# fit() не вызываем — модели уже готовы, сразу предсказываем
y_pred_ens = ensemble.predict(X_test)

acc_ens      = accuracy_score(y_test, y_pred_ens)
f1_ens_w     = f1_score(y_test, y_pred_ens, average='weighted')
f1_ens_macro = f1_score(y_test, y_pred_ens, average='macro')
elapsed_ens  = time.time() - t0

print(f"🏆 ENSEMBLE | Acc: {acc_ens:.4f} | F1-weighted: {f1_ens_w:.4f} | F1-macro: {f1_ens_macro:.4f}")

results.append({
    'Model':       'Hybrid Ensemble',
    'Accuracy':    round(acc_ens, 4),
    'F1-weighted': round(f1_ens_w, 4),
    'F1-macro':    round(f1_ens_macro, 4),
    'Time (s)':    round(elapsed_ens, 1),
})

# --- ДЕТАЛЬНЫЙ ОТЧЁТ ---
print("\n📋 ДЕТАЛЬНЫЙ ОТЧЁТ ПО КЛАССАМ (Ensemble):")
print(classification_report(y_test, y_pred_ens, target_names=CLASS_NAMES, digits=4))

# --- ИТОГИ ---
results_df = pd.DataFrame(results).sort_values(by='F1-macro', ascending=False)
print("\n📊 ИТОГОВАЯ ТАБЛИЦА:")
print(results_df.to_string(index=False))

# --- СОХРАНЕНИЕ ---
print(f"\n💾 6. Сохранение модели → {MODEL_SAVE_PATH}")
joblib.dump(ensemble, MODEL_SAVE_PATH)
print("🎉 Готово! Модель и скейлер сохранены.")
print(f"\n   ⚠️  При инференсе не забудь применить скейлер к ручным признакам:")
print(f"      scaler = joblib.load('{SCALER_SAVE_PATH}')")
print(f"      X_manual_scaled = scaler.transform(X_manual)")