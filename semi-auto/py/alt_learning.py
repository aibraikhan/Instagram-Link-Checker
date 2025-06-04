import re
import string
import hashlib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from plotly.subplots import make_subplots
from plotly import graph_objects as go
from tld import get_tld
from tldextract import extract as tld_extract
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, BaggingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import whois
import nltk
import gensim

from difflib import SequenceMatcher

# suppress warnings
warnings.filterwarnings("ignore")

# --------------------
# Whitelist loading
# --------------------
WHITELIST_DF = pd.read_csv('majestic_million.csv')
WHITELIST_SET = set(WHITELIST_DF['domain'].str.lower().str.strip())

def extract_full_domain(url: str) -> str:
    try:
        ext = tld_extract(url)
        return f"{ext.domain}.{ext.suffix}".lower()
    except:
        return ''

def is_whitelisted_domain(url: str) -> int:
    domain = extract_full_domain(url)
    return int(domain in WHITELIST_SET)

def domain_similarity(url: str) -> float:
    netloc = urlparse(url).netloc or ''
    return max([SequenceMatcher(None, netloc, legit).ratio() for legit in WHITELIST_SET])

# --------------------
# Feature extraction helpers
# --------------------
def get_url_length(url: str) -> int:
    url = re.sub(r"^https?://", "", url)
    return len(url)

def extract_netloc(url: str) -> str:
    return urlparse(url).netloc or ''

PREFIXES = ['http://', 'https://', 'www.']
SHORT_SVC = set([
    'bit', 'goo', 'tinyurl', 'ow', 't', 'is', 'cli', 'yfrog', 'migre',
    'ff', 'url4', 'twit', 'su', 'snipurl', 'short', 'budurl', 'ping',
    'post', 'just', 'bkite', 'snipr', 'fic', 'loopt', 'doiop', 'kl', 'wp',
    'rubyurl', 'om', 'to', 'lnkd', 'db', 'qr', 'adf', 'bitly', 'cur', 'ity',
    'q', 'po', 'bc', 'twitthis', 'u', 'j', 'buzurl', 'cutt', 'yourls', 'x'
])

def extract_pri_domain(url: str) -> str:
    try:
        tld = get_tld(url, as_object=True, fix_protocol=True)
        return tld.parsed_url.netloc
    except:
        return ''


def count_letters(url: str) -> int:
    return sum(c.isalpha() for c in url)

def count_digits(url: str) -> int:
    return sum(c.isdigit() for c in url)

def count_special_chars(url: str) -> int:
    return sum(c in string.punctuation for c in url)


def has_shortening_service(url: str) -> int:
    m = re.search(r'https?://(?:www\.)?(?:[\w-]+\.)*([\w-]+)\.', url)
    if not m:
        return 0
    domain = m.group(1).lower()
    return int(domain in SHORT_SVC)


def abnormal_url(url: str) -> int:
    h = urlparse(url).netloc or ''
    return int(bool(h and h not in url))


def secure_http(url: str) -> int:
    return int(urlparse(url).scheme.lower() == 'https')


def have_ip_address(url: str) -> int:
    h = urlparse(url).hostname or ''
    try:
        import ipaddress
        ipaddress.ip_address(h)
        return 1
    except:
        return 0


def tld_length(url: str) -> int:
    t = get_tld(url, fail_silently=True) or ''
    return len(t)


def get_url_region(primary_domain: str) -> str:
    # simplified: use trailing ccTLD
    cc = '.' + primary_domain.split('.')[-1]
    return cc.upper()


def extract_root_domain(url: str) -> str:
    e = tld_extract(url)
    return e.domain or ''


def hash_encode(s: str, mod: int = 10**8) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def get_numerical_values(url: str) -> dict:
    x = url.lower().strip().replace('www.', '')
    return {
        'url_len': get_url_length(x),
        'letters_count': count_letters(x),
        'digits_count': count_digits(x),
        'special_chars_count': count_special_chars(x),
        'shortened': has_shortening_service(x),
        'abnormal_url': abnormal_url(x),
        'secure_http': secure_http(x),
        'have_ip': have_ip_address(x),
        'url_region': hash_encode(get_url_region(x)),
        'netloc_hash': hash_encode(extract_netloc(x)),
        'is_whitelisted': is_whitelisted_domain(x),
        'domain_similarity': domain_similarity(x)
    }

# --------------------
# Main pipeline
# --------------------

def main():
    # 1. Load data
    df = pd.read_csv('malicious_phish.csv')
    df['url'] = df['url'].astype(str).str.replace(r'^www\.', '', regex=True)
    df['url_type'] = df['type'].map({
        'benign': 0, 'defacement':1, 'phishing':2, 'malware':3
    })

    # 2. Feature engineering
    feat = df['url'].apply(get_numerical_values)
    data = pd.DataFrame(feat.tolist())
    data['url_type'] = df['url_type']
    data.dropna(inplace=True)

    X = data.drop(columns=['url_type'])
    y = data['url_type']

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Evaluate basic classifiers
    classifiers = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        ExtraTreesClassifier(n_estimators=100, random_state=42),
        XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        ),
        LGBMClassifier(
            objective='multiclass',
            n_estimators=100,
            random_state=42
        )
    ]

    results = []
    for clf in classifiers:
        name = clf.__class__.__name__
        # для простоты без пайплайна:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred, average='weighted')
        pre  = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        f1   = f1_score(y_test, y_pred, average='weighted')

        results.append((name, acc, rec, pre, f1))

    # выводим сравнительную табличку
    df_res = pd.DataFrame(
        results, columns=['Model','Accuracy','Recall','Precision','F1']
    ).sort_values('Accuracy', ascending=False)
    print(df_res)

    # 5. Fit best classifier and save
    best_clf = max(zip(classifiers, results), key=lambda x: x[1][1])[0]
    # где x[1][1] — accuracy
    import joblib
    joblib.dump(best_clf, 'best_model_v3.sav')

    # 6. (Optional) Visualize correlation
    corr = X.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Feature Correlation')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()