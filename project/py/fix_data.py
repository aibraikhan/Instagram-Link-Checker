import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit

# НАСТРОЙКИ ПУТЕЙ
FILE_PATH = './project/py/dataset_final_augmented.csv'
# NEW_DATASET_PATH = './project/py/phishing_site_urls.csv'
# OUTPUT_PATH = '/project/py/dataset_final_balanced.csv'


print("⏳ Загружаем финальный датасет...")
df = pd.read_csv(FILE_PATH)
df['url'] = df['url'].astype(str)

print("🔍 Извлекаем корневые домены для группировки...")
# Так как мы уже вырезали http:// и www., корень домена — это всё до первого слэша '/'
df['domain'] = df['url'].apply(lambda x: x.split('/')[0])

print(f"📊 Найдено уникальных доменов: {df['domain'].nunique()}")

print("🔀 Разделяем данные (80% Train / 20% Test) с изоляцией доменов...")
# n_splits=1 означает, что нам нужно только одно разбиение
# random_state=42 гарантирует, что каждый раз при запуске разбиение будет одинаковым (воспроизводимость)
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

# Получаем индексы строк для Train и Test
train_idx, test_idx = next(gss.split(df, groups=df['domain']))

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("\n📈 Статистика разделения:")
print(f"Обучающая выборка (Train): {len(train_df)} строк")
print(f"Тестовая выборка (Test): {len(test_df)} строк")

print("\n🛡️ Проверка пересечения доменов (Data Leakage Check):")
train_domains = set(train_df['domain'])
test_domains = set(test_df['domain'])
overlap = train_domains.intersection(test_domains)

if len(overlap) == 0:
    print("✅ Идеально! Утечек данных нет. Ни один домен не попал одновременно в обе выборки.")
else:
    print(f"⚠️ ВНИМАНИЕ: Найдено {len(overlap)} пересекающихся доменов.")

# Удаляем вспомогательную колонку domain перед сохранением
train_df = train_df.drop(columns=['domain'])
test_df = test_df.drop(columns=['domain'])

print("\n💾 Сохраняем файлы...")
train_df.to_csv('./project/py/train.csv', index=False)
test_df.to_csv('./project/py/test.csv', index=False)
print("🎉 Готово! Файлы train.csv и test.csv созданы и полностью готовы к машинному обучению.")