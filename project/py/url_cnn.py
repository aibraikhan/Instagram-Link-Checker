"""
╔══════════════════════════════════════════════════════════════╗
║         Character-level CNN — Malicious URL Classifier       ║
║         Classes: benign | defacement | malware | phishing    ║
║         Optimised for Apple M2 (MPS) · PyTorch               ║
╚══════════════════════════════════════════════════════════════╝

USAGE
─────
  # Install deps (once)
  pip install torch pandas scikit-learn matplotlib seaborn

  # Train
  python url_cnn.py --mode train

  # Predict single URL
  python url_cnn.py --mode predict --url "free-iphone.ru/login?ref=abc"

  # Predict from file (one URL per line)
  python url_cnn.py --mode predict --url_file my_urls.txt

  # Evaluate saved model on test.csv
  python url_cnn.py --mode eval
"""

# ══════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


# ══════════════════════════════════════════════════════════════
#  CONFIG  ← edit here if needed
# ══════════════════════════════════════════════════════════════

class Config:
    # ── Files ────────────────────────────────────────────────
    TRAIN_CSV   = "train.csv"
    TEST_CSV    = "test.csv"
    URL_COL     = "url"           # column name in CSV
    LABEL_COL   = "type"          # column name in CSV
    SAVE_PATH   = "url_cnn_best.pt"

    # ── Preprocessing ────────────────────────────────────────
    # URLs in dataset have no scheme (e.g. "google.com/path")
    # We keep them as-is; the model learns from raw characters.
    MAX_URL_LEN = 256             # truncate/pad to this length

    # Character vocabulary — everything that can appear in a URL
    ALPHABET = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "-_.~:/?#[]@!$&'()*+,;=%\\"
    )

    # ── Model ────────────────────────────────────────────────
    EMBED_DIM    = 64
    NUM_FILTERS  = 256                  # per kernel size
    KERNEL_SIZES = [2, 3, 4, 5]  # n-gram detectors
    DROPOUT      = 0.5

    # ── Training ─────────────────────────────────────────────
    BATCH_SIZE       = 512     # safe for 16 GB RAM + M2
    EPOCHS           = 50      # early stopping will kick in before this
    LR               = 3e-3
    WEIGHT_DECAY     = 1e-4
    EARLY_STOP_PAT   = 8       # stop if val_acc stalls for N epochs
    PHISHING_W_BOOST = 1.0     # no boost — 2.0 killed precision

    # ── Device ───────────────────────────────────────────────
    @staticmethod
    def device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")   # Apple Silicon GPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


# ══════════════════════════════════════════════════════════════
#  VOCABULARY  (character → integer index)
# ══════════════════════════════════════════════════════════════

class CharVocab:
    PAD = 0   # padding index
    UNK = 1   # unknown character index

    def __init__(self, alphabet: str):
        # index 0 = PAD, 1 = UNK, then each char
        self._c2i: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        for ch in alphabet:
            self._c2i[ch] = len(self._c2i)

    def __len__(self) -> int:
        return len(self._c2i)

    def encode(self, url: str, max_len: int) -> list[int]:
        """Convert URL string to list of indices, truncated/padded to max_len."""
        ids = [self._c2i.get(ch, self.UNK) for ch in url[:max_len]]
        ids += [self.PAD] * (max_len - len(ids))   # right-pad
        return ids


# ══════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════

class URLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: CharVocab,
                 label_enc: LabelEncoder, max_len: int):
        self.vocab    = vocab
        self.max_len  = max_len
        self.urls     = df[Config.URL_COL].astype(str).tolist()
        self.labels   = label_enc.transform(df[Config.LABEL_COL]).tolist()

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx):
        x = torch.tensor(self.vocab.encode(self.urls[idx], self.max_len),
                          dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ══════════════════════════════════════════════════════════════
#  MODEL  —  Character-level CNN
# ══════════════════════════════════════════════════════════════

class URLCharCNN(nn.Module):
    """
    Pipeline:
        URL string
          → character embeddings          (batch, seq, embed_dim)
          → parallel Conv1d (k=2,3,4,5)  (batch, filters, seq)
          → global max-pool per branch    (batch, filters)
          → concat all branches           (batch, filters * 4)
          → dropout → FC → ReLU → FC     (batch, num_classes)

    Why this works for URLs:
        Short n-gram filters detect suspicious sub-strings such as
        "login", ".exe", "//", "base64", "bit.ly", etc. regardless
        of where they appear in the URL.
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 num_filters: int, kernel_sizes: list[int],
                 num_classes: int, dropout: float):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embed_dim,
            padding_idx    = CharVocab.PAD
        )

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(num_filters),
                nn.GELU(),
            )
            for k in kernel_sizes
        ])

        total = num_filters * len(kernel_sizes)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L)
        emb = self.embedding(x).permute(0, 2, 1)          # (B, E, L)
        pooled = [F.adaptive_max_pool1d(conv(emb), 1).squeeze(-1)
                  for conv in self.convs]                  # each (B, F)
        return self.head(torch.cat(pooled, dim=1))         # (B, num_classes)


# ══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def run_epoch(model, loader, device, optimizer=None,
              scheduler=None, criterion=None):
    training = optimizer is not None
    model.train(training)

    loss_sum, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

            loss_sum += loss.item() * len(y)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += len(y)

    return loss_sum / total, correct / total


def train(cfg: Config):
    device = cfg.device()
    print(f"\n{'─'*55}")
    print(f"  Device : {device}")

    # ── Load CSVs ──────────────────────────────────────────
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    test_df  = pd.read_csv(cfg.TEST_CSV)

    for col in [cfg.URL_COL, cfg.LABEL_COL]:
        assert col in train_df.columns, \
            f"Column '{col}' missing in {cfg.TRAIN_CSV}. " \
            f"Available: {list(train_df.columns)}"

    print(f"  Train  : {len(train_df):,} rows")
    print(f"  Test   : {len(test_df):,} rows")
    print(f"\n  Label distribution (train):")
    counts = train_df[cfg.LABEL_COL].value_counts()
    for label, cnt in counts.items():
        print(f"    {label:<15} {cnt:>8,}  ({cnt/len(train_df):.1%})")

    # ── Label encoder ─────────────────────────────────────
    le = LabelEncoder().fit(train_df[cfg.LABEL_COL])
    class_names  = list(le.classes_)
    num_classes  = len(class_names)
    print(f"\n  Classes ({num_classes}): {class_names}")

    # ── Vocab & Datasets ───────────────────────────────────
    vocab = CharVocab(cfg.ALPHABET)
    print(f"  Vocab size: {len(vocab)}")

    train_ds = URLDataset(train_df, vocab, le, cfg.MAX_URL_LEN)
    test_ds  = URLDataset(test_df,  vocab, le, cfg.MAX_URL_LEN)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # ── Class-weighted loss (handles imbalance) ────────────
    total = len(train_df)
    weights = torch.tensor(
        [total / (num_classes * counts[cls]) for cls in class_names],
        dtype=torch.float, device=device
    )
    # Boost phishing weight to improve its low precision
    if "phishing" in class_names:
        idx = class_names.index("phishing")
        weights[idx] *= cfg.PHISHING_W_BOOST
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"\n  Class weights (after phishing boost):")
    for cls, w in zip(class_names, weights.cpu()):
        print(f"    {cls:<15} {w:.3f}")

    # ── Model ──────────────────────────────────────────────
    model = URLCharCNN(
        vocab_size   = len(vocab),
        embed_dim    = cfg.EMBED_DIM,
        num_filters  = cfg.NUM_FILTERS,
        kernel_sizes = cfg.KERNEL_SIZES,
        num_classes  = num_classes,
        dropout      = cfg.DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Parameters: {n_params:,}")
    print(f"{'─'*55}\n")

    # ── Optimizer + scheduler ──────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=cfg.LR,
                           steps_per_epoch=len(train_loader),
                           epochs=cfg.EPOCHS, pct_start=0.1)

    # ── Training loop ──────────────────────────────────────
    history      = {"tr_loss": [], "tr_acc": [], "vl_loss": [], "vl_acc": []}
    best_acc     = 0.0
    no_improve   = 0          # early stopping counter
    header    = f"{'Epoch':>6}  {'Tr Loss':>9} {'Tr Acc':>8}  {'Vl Loss':>9} {'Vl Acc':>8}  {'Time':>6}"
    print(header)
    print("─" * len(header))

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device,
                                    optimizer, scheduler, criterion)
        vl_loss, vl_acc = run_epoch(model, test_loader,  device,
                                    criterion=criterion)
        elapsed = time.time() - t0

        history["tr_loss"].append(tr_loss)
        history["tr_acc"].append(tr_acc)
        history["vl_loss"].append(vl_loss)
        history["vl_acc"].append(vl_acc)

        flag = ""
        if vl_acc > best_acc:
            best_acc   = vl_acc
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "vocab_size":  len(vocab),
            }, cfg.SAVE_PATH)
            flag = "  ✓ saved"
        else:
            no_improve += 1

        print(f"{epoch:>6}  {tr_loss:>9.4f} {tr_acc:>7.4f}  "
              f"{vl_loss:>9.4f} {vl_acc:>7.4f}  {elapsed:>5.1f}s{flag}")

        if no_improve >= cfg.EARLY_STOP_PAT:
            print(f"\n  Early stopping: no improvement for {cfg.EARLY_STOP_PAT} epochs.")
            break

    print(f"\nBest val accuracy : {best_acc:.4f}")
    print(f"Model saved to    : {cfg.SAVE_PATH}")

    # ── Final evaluation ───────────────────────────────────
    _evaluate(model, test_loader, device, class_names, criterion)
    _plot_history(history)


# ══════════════════════════════════════════════════════════════
#  EVALUATION  (also called after training)
# ══════════════════════════════════════════════════════════════

def _evaluate(model, loader, device, class_names, criterion):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    print("\n── Classification Report ─────────────────────────────")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved: confusion_matrix.png")


def _plot_history(history: dict):
    ep = range(1, len(history["tr_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ep, history["tr_loss"], label="Train")
    ax1.plot(ep, history["vl_loss"], label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(ep, history["tr_acc"], label="Train")
    ax2.plot(ep, history["vl_acc"], label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("Saved: training_history.png")


# ══════════════════════════════════════════════════════════════
#  EVAL-ONLY MODE  (load saved model, run on test.csv)
# ══════════════════════════════════════════════════════════════

def eval_only(cfg: Config):
    checkpoint = torch.load(cfg.SAVE_PATH, map_location=cfg.device())
    class_names = checkpoint["class_names"]

    model = URLCharCNN(
        vocab_size   = checkpoint["vocab_size"],
        embed_dim    = cfg.EMBED_DIM,
        num_filters  = cfg.NUM_FILTERS,
        kernel_sizes = cfg.KERNEL_SIZES,
        num_classes  = len(class_names),
        dropout      = 0.0,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(cfg.device()).eval()

    vocab    = CharVocab(cfg.ALPHABET)
    test_df  = pd.read_csv(cfg.TEST_CSV)
    le       = LabelEncoder().fit(test_df[cfg.LABEL_COL])
    test_ds  = URLDataset(test_df, vocab, le, cfg.MAX_URL_LEN)
    loader   = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE,
                          shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    _evaluate(model, loader, cfg.device(), class_names, criterion)


# ══════════════════════════════════════════════════════════════
#  INFERENCE  (predict one or many URLs)
# ══════════════════════════════════════════════════════════════

def predict(cfg: Config, urls: list[str]) -> list[dict]:
    device = cfg.device()
    checkpoint  = torch.load(cfg.SAVE_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = URLCharCNN(
        vocab_size   = checkpoint["vocab_size"],
        embed_dim    = cfg.EMBED_DIM,
        num_filters  = cfg.NUM_FILTERS,
        kernel_sizes = cfg.KERNEL_SIZES,
        num_classes  = len(class_names),
        dropout      = 0.0,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    vocab = CharVocab(cfg.ALPHABET)

    x = torch.tensor(
        [vocab.encode(u, cfg.MAX_URL_LEN) for u in urls],
        dtype=torch.long, device=device
    )
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).cpu().numpy()

    results = []
    for url, prob in zip(urls, probs):
        top = int(prob.argmax())
        results.append({
            "url":        url,
            "prediction": class_names[top],
            "confidence": f"{prob[top]:.2%}",
            "all_scores": {c: f"{p:.2%}" for c, p in zip(class_names, prob)},
        })
    return results


# ══════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Character-level CNN — URL classifier"
    )
    parser.add_argument(
        "--mode", choices=["train", "eval", "predict"],
        default="train", help="train | eval | predict"
    )
    parser.add_argument("--url",      type=str, default=None,
                        help="Single URL to predict")
    parser.add_argument("--url_file", type=str, default=None,
                        help="Path to .txt file with one URL per line")
    args = parser.parse_args()

    cfg = Config()

    if args.mode == "train":
        train(cfg)

    elif args.mode == "eval":
        eval_only(cfg)

    elif args.mode == "predict":
        if args.url:
            urls = [args.url]
        elif args.url_file:
            with open(args.url_file) as f:
                urls = [l.strip() for l in f if l.strip()]
        else:
            parser.error("--predict requires --url or --url_file")

        results = predict(cfg, urls)
        print()
        for r in results:
            print(f"URL        : {r['url']}")
            print(f"Prediction : {r['prediction']}  ({r['confidence']})")
            print(f"All scores : {r['all_scores']}")
            print()


if __name__ == "__main__":
    main()