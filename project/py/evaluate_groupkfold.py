# project/py/evaluate_groupkfold.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve
)
from joblib import dump
from features import get_feature_vector, registrable_domain

# Модели: LGBM/XGB/Cat — выбирай по флагу
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt


def _normalize_url_like(u: str) -> str:
    u = (u or "").strip()
    if not u: return ""
    # если пришел домен без схемы — добавим https://
    if not (u.startswith("http://") or u.startswith("https://")):
        u = "https://" + u
    return u

def _load_whitelist(path: str) -> pd.DataFrame:
    wl = pd.read_csv(path)
    # поддержим оба варианта: 'url' ИЛИ 'domain'
    col = "url" if "url" in wl.columns else ("domain" if "domain" in wl.columns else None)
    if col is None:
        raise ValueError("whitelist.csv must have 'url' or 'domain' column")
    wl = wl[[col]].rename(columns={col: "url"})
    wl["url"] = wl["url"].astype(str).map(_normalize_url_like)
    wl["label"] = 0  # бенign
    # выкинем пустые/битые
    wl = wl[wl["url"].str.len() > 0].drop_duplicates("url")
    return wl


def build_model(kind: str):
    if kind=="lgbm":
        return LGBMClassifier(
            n_estimators=600, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=-1
        )
    if kind=="xgb":
        return XGBClassifier(
            n_estimators=600, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            max_depth=8, eval_metric="logloss",
            random_state=42, n_jobs=-1, tree_method="hist"
        )
    return CatBoostClassifier(
        iterations=800, depth=8, learning_rate=0.05,
        random_seed=42, verbose=False, loss_function="Logloss"
    )

def optimal_f1_threshold(y_true, y_prob):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2*p*r/(p+r+1e-9)
    if len(thr)==0: return 0.5
    i = int(np.nanargmax(f1))
    i = min(i, len(thr)-1)
    return float(thr[i])

def _safe_metrics(y_true, y_prob, thr):
    from sklearn.utils.multiclass import type_of_target
    y_pred = (y_prob >= thr).astype(int)
    m = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
        "thr_f1": thr,
    }
    # AUC/AP считаем только если в y_true есть и 0, и 1
    if len(np.unique(y_true)) == 2:
        m["roc_auc"] = roc_auc_score(y_true, y_prob)
        m["pr_auc"]  = average_precision_score(y_true, y_prob)
    else:
        m["roc_auc"] = float("nan")
        m["pr_auc"]  = float("nan")
    return m

FEATURE_NAMES = [
    "url_len",
    "letters_count",
    "digits_count",
    "special_chars_count",
    "shortened",
    "abnormal_url",
    "secure_http",
    "have_ip",
    "tld_length",
    "url_region_hash",
    "netloc_hash",
    "num_dots_host",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="project/py/malicious_phish.csv")
    ap.add_argument("--whitelist", default="project/py/whitelist.csv")
    ap.add_argument("--use_whitelist", action="store_true")
    ap.add_argument("--textcol", default="url")
    ap.add_argument("--target",  default="label")
    ap.add_argument("--model",   default="lgbm", choices=["lgbm","xgb","cat"])
    ap.add_argument("--folds",   type=int, default=5)
    ap.add_argument("--max_wl_per_domain", type=int, default=50)          # <— кап на домен
    ap.add_argument("--wl_ratio", type=float, default=1.0)                # <— сколько benign на 1 malicious
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[[args.textcol, args.target]].rename(columns={args.textcol:"url", args.target:"label"})
    df["label"] = df["label"].replace({"benign":0,"good":0,"phishing":1,"malicious":1,"defacement":1,"malware":1}).astype(int)
    df["url"] = df["url"].astype(str).map(_normalize_url_like)
    df = df.dropna().drop_duplicates("url")

    # при необходимости подмешиваем whitelist
    if args.use_whitelist:
        if not Path(args.whitelist).exists():
            print(f"[WARN] whitelist file not found: {args.whitelist}")
        else:
            wl = _load_whitelist(args.whitelist)

            # убираем пересечения по registrable_domain, чтобы не было утечки «домен → всегда 0»
            # и чтобы один домен не «забил» весь benign класс:
            wl["rd"] = wl["url"].map(registrable_domain)
            df["rd"] = df["url"].map(registrable_domain)

            # кап по домену
            wl = wl.groupby("rd", as_index=False).head(args.max_wl_per_domain)

            # удерживаем разумную пропорцию benign:malicious
            n_mal = int((df["label"] == 1).sum())
            
            # wl_ratio: добавляем не более wl_ratio * (#malicious) benign-URL из whitelist
            n_wl_target = int(min(len(wl), args.wl_ratio * n_mal))
            wl = wl.sample(n=min(n_wl_target, len(wl)), random_state=42)

            # уберем явные дубликаты URL и конфликтующие домены, если они есть в малишес:
            mal_rds = set(df.loc[df["label"]==1, "rd"])
            wl = wl[~wl["rd"].isin(mal_rds)]

            # сливаем
            df = pd.concat([df.drop(columns=["rd"], errors="ignore"), wl.drop(columns=["rd"], errors="ignore")], ignore_index=True)
            df = df.drop_duplicates("url")

    # фичи + группы
    X = np.vstack([get_feature_vector(u) for u in df["url"].astype(str).tolist()]).astype(np.float32)
    y = df["label"].values
    groups = df["url"].map(registrable_domain).values

    model = build_model(args.model)
    gkf = GroupKFold(n_splits=args.folds)

    all_true, all_prob, rows = [], [], []

    learning_curve_x = None   # ось итераций
    learning_curve_y = None   # метрика на валидации по итерациям (logloss)

    for k, (tr, va) in enumerate(gkf.split(X, y, groups), 1):
        # учим модель с валидационным сетом и логированием
        evals_result = {}
        model.fit(
            X[tr], y[tr],
            eval_set=[(X[va], y[va])],
            eval_metric="logloss",
            callbacks=[],
        )

        # если это первый фолд, снимем кривую обучения
        # LightGBM в sklearn-обёртке хранит модель в model.booster_
        try:
            if k == 1:
                # у LGBMClassifier после fit есть model.evals_result_ (dict)
                # формат: {'valid_0': {'binary_logloss': [..values..]}}
                er = model.evals_result_
                # ключ метрики может называться "binary_logloss" или "logloss"
                valid_dict = er.get("valid_0", {})
                metric_name = "binary_logloss" if "binary_logloss" in valid_dict else "logloss"
                curve = valid_dict.get(metric_name, None)
                if curve is not None:
                    learning_curve_x = list(range(1, len(curve)+1))
                    learning_curve_y = curve
        except Exception as e:
            print(f"[WARN] couldn't extract learning curve for fold {k}: {e}")

        # дальше как раньше: считаем вероятности, метрики и т.д.
        y_prob = model.predict_proba(X[va])[:,1]
        thr = optimal_f1_threshold(y[va], y_prob)
        rows.append({"fold": k, **_safe_metrics(y[va], y_prob, thr)})
        all_true.append(y[va]); all_prob.append(y_prob)


    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    thr = optimal_f1_threshold(y_true, y_prob)
    summary = _safe_metrics(y_true, y_prob, thr)

    # гарантируем, что папка есть
    Path("results").mkdir(exist_ok=True, parents=True)

    # сохраняем метрики по фолдам и сводку
    pd.DataFrame(rows).to_csv("results/fold_metrics_group.csv", index=False)
    with open("results/metrics_group_summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # === Сохранение кривой обучения (если удалось её снять) ===
    if learning_curve_x is not None and learning_curve_y is not None:
        # сохраним в csv (на случай отчёта)
        pd.DataFrame({
            "iteration": learning_curve_x,
            "valid_logloss": learning_curve_y,
        }).to_csv("results/lgbm_learning_curve.csv", index=False)

        # построим график
        plt.figure(figsize=(6,4))
        plt.plot(learning_curve_x, learning_curve_y)
        plt.xlabel("Iteration (trees)")
        plt.ylabel("Validation logloss")
        plt.title("LightGBM learning curve (fold 1)")
        plt.tight_layout()
        plt.savefig("results/lgbm_learning_curve.png", dpi=200)
        plt.close()

        print("Saved learning curve to results/lgbm_learning_curve.png")
    else:
        print("[WARN] no learning curve captured")


    print("Saved to results/metrics_group_summary.json")
    if args.save_model:
        # дообучаем модель на всех данных
        model.fit(X, y)

        # сохраняем модель для API
        Path("project/py").mkdir(exist_ok=True, parents=True)
        dump(model, "project/py/best_model_v4.sav")
        print("Saved model to project/py/best_model_v4.sav")

        # === ГРАФИК ВАЖНОСТИ ПРИЗНАКОВ (ТОЛЬКО ЕСЛИ LGBM / XGB / CatBoost ДАЮТ feature_importances_) ===
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            # подрежем FEATURE_NAMES на случай, если длиннее
            names = FEATURE_NAMES[:len(importances)]

            # отсортируем фичи по важности по убыванию
            order = np.argsort(importances)[::-1]
            sorted_importances = importances[order]
            sorted_names = [names[i] for i in order]

            # сохраним в csv на всякий случай
            fi_df = pd.DataFrame({
                "feature": sorted_names,
                "importance": sorted_importances
            })
            Path("results").mkdir(exist_ok=True, parents=True)
            fi_df.to_csv("results/feature_importance.csv", index=False)

            # построим барчарт и сохраним
            plt.figure(figsize=(8, 4))
            plt.bar(range(len(sorted_importances)), sorted_importances)
            plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha="right")
            plt.ylabel("Feature importance")
            plt.title("Feature importance (final model)")
            plt.tight_layout()
            plt.savefig("results/feature_importance.png", dpi=200)
            plt.close()
            print("Saved feature importance plot to results/feature_importance.png")
        else:
            print("Model has no feature_importances_ attribute; skipping plot.")

        

if __name__ == "__main__":
    main()
