import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# 1. Загружаем твои данные
print("Загрузка данных...")
X_all = np.load('./project/py/bert_features.npy')
y_all = np.load('./project/py/bert_labels.npy')

# 2. Разбиваем на Train (80%) и Test (20%)
# (Для графиков обычного train_test_split будет достаточно)
print("Разбиение данных...")
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# ==========================================
# ГРАФИК 1: Кривая обучения (Learning Curve)
# ==========================================
print("Обучение LightGBM и построение Learning Curve...")
lgb_model = lgb.LGBMClassifier(n_estimators=300, random_state=42)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_names=['Train', 'Validation'],
    eval_metric='multi_logloss'
)

evals_result = lgb_model.evals_result_

plt.figure(figsize=(10, 6))
plt.plot(evals_result['Train']['multi_logloss'], label='Ошибка на обучающей выборке (Train)', lw=2)
plt.plot(evals_result['Validation']['multi_logloss'], label='Ошибка на тестовой выборке (Validation)', lw=2)
plt.xlabel('Количество деревьев (Итерации)', fontsize=12)
plt.ylabel('Логистическая ошибка (Multi-LogLoss)', fontsize=12)
plt.title('Кривая обучения (Learning Curve) модели LightGBM', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.savefig('learning_curve_lgbm.png', dpi=300, bbox_inches='tight')
print("✅ График learning_curve_lgbm.png сохранен!")

# ==========================================
# ГРАФИК 2: ROC-AUC кривая
# ==========================================
print("Построение ROC-AUC...")
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Получаем вероятности предсказаний
y_score = lgb_model.predict_proba(X_test)

# Бинаризуем метки для 4 классов
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
class_names = ['Benign (0)', 'Phishing (1)', 'Malware (2)', 'Defacement (3)']
colors = ['green', 'red', 'purple', 'orange']

plt.figure(figsize=(10, 8))
for i, color in zip(range(4), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC-кривые для мультиклассовой классификации', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.savefig('roc_auc_curve.png', dpi=300, bbox_inches='tight')
print("✅ График roc_auc_curve.png сохранен!")