import pandas as pd

# НАСТРОЙКИ ПУТЕЙ
MAIN_DATASET = '/Users/alinur/Documents/code for diss/Instagram-Link-Checker/project/py/malicious_phish.csv'
BENIGN_1M_FILE = '/Users/alinur/Documents/code for diss/Instagram-Link-Checker/project/py/whitelist.csv' # <--- УКАЖИ ПУТЬ К ФАЙЛУ С 1 МЛН
OUTPUT_FILE = '/Users/alinur/Documents/code for diss/Instagram-Link-Checker/project/py/dataset_final_augmented.csv'

# 1. Загружаем основной датасет
print("Загружаем основной датасет...")
try:
    df_main = pd.read_csv(MAIN_DATASET)
    print(f"Основной: {df_main.shape[0]} строк")
except Exception as e:
    print(f"Ошибка открытия основного файла: {e}")
    exit()

# 2. Загружаем список 1 млн benign (предполагаем, что там нет заголовка или колонка называется 'domain')
print("Загружаем список 1 млн...")
try:
    # Если в файле нет заголовков, используем header=None и даем имя колонке 'url'
    # Если заголовок есть, pandas сам поймет, но лучше переименовать в 'url' для совместимости
    df_1m = pd.read_csv(BENIGN_1M_FILE)
    
    # Если колонка называется 'domain', переименуем в 'url'
    if 'domain' in df_1m.columns:
        df_1m.rename(columns={'domain': 'url'}, inplace=True)
    elif df_1m.shape[1] == 1:
        df_1m.columns = ['url'] # Если колонка одна без названия
        
    print(f"Доп. список: {df_1m.shape[0]} строк")
except Exception as e:
    print(f"Ошибка открытия файла с 1 млн: {e}")
    exit()

# 3. Фильтруем ТОЛЬКО .kz из миллионника
print("Ищем .kz домены в списке миллионнике...")
kz_aug = df_1m[df_1m['url'].astype(str).str.endswith('.kz', na=False)].copy()

# Добавляем метку benign
# Проверяем как в основном файле называется колонка меток (type или label)
target_col = 'type' if 'type' in df_main.columns else 'label'
kz_aug[target_col] = 'benign'

print(f"Найдено хороших .kz доменов: {len(kz_aug)}")

# 4. Объединяем
df_final = pd.concat([df_main, kz_aug], ignore_index=True)

# Удаляем дубликаты
df_final.drop_duplicates(subset=['url'], inplace=True)

# 5. Сохраняем
df_final.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Готово! Финальный датасет сохранен: {OUTPUT_FILE}")
print(f"Новый размер: {df_final.shape[0]}")