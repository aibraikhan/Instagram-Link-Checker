import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# 1. Загрузка данных из CSV
def load_data(malicious_file, whitelist_file):
    # Загружаем данные из файла с вредоносными и безопасными ссылками
    df = pd.read_csv(malicious_file)
    
    # Предполагаем, что в CSV есть колонка 'url' с URL и 'label' с метками
    urls = df['url'].astype(str).tolist()
    
    # Преобразуем метки
    label_map = {
        'benign': 0,  # Безопасные ссылки
        'phishing': 1,  # Фишинговые ссылки
        'malware': 2,  # Вредоносные ссылки
        'defacement': 3  # Подозрительные ссылки
    }
    labels = df['label'].map(label_map).astype(int).tolist()
    
    # Загружаем белый список (whitelist) и добавляем метку "benign" для всех ссылок
    whitelist_df = pd.read_csv(whitelist_file)
    whitelist_urls = whitelist_df['domain'].astype(str).tolist()
    
    return urls, labels, whitelist_urls

# 2. Преобразование URL в числовое представление
def preprocess_url(url):
    url = url.lower()
    # Убираем префиксы http:// и https://
    if url.startswith("http://"):
        url = url[7:]
    elif url.startswith("https://"):
        url = url[8:]
    return url

def encode_urls(urls, max_length=100):
    # Инициализируем токенизатор для символов
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(urls)  # Создаём словарь токенов
    sequences = tokenizer.texts_to_sequences(urls)  # Преобразуем URL в числовые последовательности

    # Паддинг последовательностей до одной длины
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    return padded_sequences, tokenizer

# 3. Основная функция
def main():
    # Пути к файлам CSV с URL
    malicious_file = "project/py/malicious_phish.csv"  # Укажи путь к твоему файлу
    whitelist_file = "project/py/whitelist.csv"  # Укажи путь к файлу с белым списком
    
    # Загрузим данные
    urls, labels, whitelist_urls = load_data(malicious_file, whitelist_file)
    
    # Преобразуем URL
    preprocessed_urls = [preprocess_url(url) for url in urls]
    preprocessed_whitelist_urls = [preprocess_url(url) for url in whitelist_urls]
    
    # Преобразуем URL в числовые последовательности с паддингом
    encoded_urls, tokenizer = encode_urls(preprocessed_urls)
    encoded_whitelist_urls, _ = encode_urls(preprocessed_whitelist_urls)  # Белый список можно обрабатывать отдельно
    
    # Сохраняем преобразованные данные (по желанию)
    np.save('encoded_urls.npy', encoded_urls)
    np.save('encoded_whitelist_urls.npy', encoded_whitelist_urls)
    np.save('labels.npy', labels)
    
    # Выводим пример числового представления URL
    print("Encoded URLs:")
    print(encoded_urls[:5])  # Выведем первые 5 преобразованных URL

    # Сохраняем токенизатор для дальнейшего использования
    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        json.dump(tokenizer_json, f)

# Запускаем основной процесс
if __name__ == "__main__":
    main()
