# eval/evaluate_groupkfold.py
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from utils.domain_groups import registrable_domain

# >>> подстрой под твои функции/пайплайны <<<
# ожидаем, что у тебя есть что-то вроде build_rf(), build_lgbm(), build_xgb()
from models import build_rf, build_lgbm, build_xgb  # поменяй импорт под себя

def optimal_f1_threshold(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2*p*r/(p+r+1e-9)
    if len(thr)==0: return 0.5
    i = int(np.nanargmax(f1))
    i = min(i, len(thr)-1)
    return float(thr[i])

def compute_metrics(y_true, y_prob, thr):
    from sklearn.metrics import (
        f1_score, roc_auc_score, brier_score_loss,
        balanced_accuracy_score, matthews_corrcoef
    )
    y_pred = (y_prob >= thr).astype(int)
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "thr_f1": thr,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)           # твой датасет
    ap.add_argument("--textcol", default="url")
    ap.add_argument("--target",  default="label")
    ap.add_argument("--model",   default="lgbm", choices=["rf","lgbm","xgb"])
    ap.add_argument("--folds",   type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    X = df[[args.textcol]].rename(columns={args.textcol:"url"})
    y = df[args.target].astype(int).values
    groups = X["url"].map(registrable_domain).values

    # >>> выбери модель из твоих билдеров <<<
    if args.model == "rf":
        model = build_rf()     # твой RandomForest пайплайн
    elif args.model == "xgb":
        model = build_xgb()
    else:
        model = build_lgbm()

    gkf = GroupKFold(n_splits=args.folds)
    all_true, all_prob, rows = [], [], []
    for k, (tr, va) in enumerate(gkf.split(X, y, groups), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        model.fit(Xtr, ytr)
        yprob = model.predict_proba(Xva)[:,1]
        thr = optimal_f1_threshold(yva, yprob)
        met = compute_metrics(yva, yprob, thr)
        met["fold"] = k
        rows.append(met)
        all_true.append(yva); all_prob.append(yprob)

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    thr = optimal_f1_threshold(y_true, y_prob)
    summary = compute_metrics(y_true, y_prob, thr)

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/fold_metrics_group.csv", index=False)
    with open("results/metrics_group_summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Saved results to results/")

if __name__ == "__main__":
    main()
