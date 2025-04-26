#!/usr/bin/env python3
"""
PhishSentry — scalable TF-IDF + Logistic-Regression phishing-email demo
Adds an `evaluate` command that generates confusion-matrix, ROC and
precision-recall plots.
"""

import argparse, csv, json, re, os
from pathlib import Path
from typing import Iterator

import joblib
import pandas as pd
from tqdm import tqdm
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    PrecisionRecallDisplay, average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# ─── constants ─────────────────────────────────────────────────────────────
csv.field_size_limit(2_000_000_000)           # allow huge cells
ENCODINGS = ("utf-8", "cp1252", "latin1")
MODEL_FILE = "model.joblib"

LABEL_MAP = {"safe email": 0, "phishing email": 1}
INV_LABEL = {v: k.title() for k, v in LABEL_MAP.items()}

URL_PATTERN   = re.compile(r"http\S+")
PUNCT_PATTERN = re.compile(r"[^\w\s]")
SPACE_PATTERN = re.compile(r"\s+")

# ─── preprocessing helpers ────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    text = URL_PATTERN.sub("", text)
    text = PUNCT_PATTERN.sub(" ", text)
    text = text.lower()
    return SPACE_PATTERN.sub(" ", text).strip()

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        c = " ".join(col.lower().split())
        if "email text" in c:
            mapping[col] = "text"
        elif "email type" in c:
            mapping[col] = "label"
    return df.rename(columns=mapping)

def drop_blank_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text"])
    return df[df["text"].str.strip().astype(bool)]

def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = (df["label"]
                   .astype(str)
                   .str.strip()
                   .str.lower()
                   .map(LABEL_MAP))
    if df["label"].isna().any():
        bad = df.loc[df["label"].isna(), "label"].unique()
        raise ValueError(f"Unmapped label values: {bad!r}")
    return df

# ─── CSV loader with encoding fallback ────────────────────────────────────
def read_csv_any(path: str, chunk: int | None = None) -> Iterator[pd.DataFrame]:
    for enc in ENCODINGS:
        try:
            reader = pd.read_csv(path, encoding=enc, chunksize=chunk,
                                 engine="python")
            if chunk is None:
                yield reader
            else:
                for c in tqdm(reader, desc="Reading CSV", unit="chunks"):
                    yield c
            return
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode {path}")

# ─── model pipeline ───────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess_text,
                                  max_features=50_000,
                                  ngram_range=(1, 2),
                                  sublinear_tf=True)),
        ("clf",  LogisticRegression(max_iter=2000,
                                    solver="saga",
                                    class_weight="balanced",
                                    n_jobs=-1,
                                    verbose=1))
    ])

# ─── training ─────────────────────────────────────────────────────────────
def train_model(csv_path: str, model_dir: str,
                test_size: float, chunk_size: int):

    if chunk_size == 0:
        df_raw = next(read_csv_any(csv_path))
        df = normalise_labels(drop_blank_text(standardise_columns(df_raw)))
    else:
        raw_chunks = (standardise_columns(c) for c in read_csv_any(csv_path, chunk_size))
        clean_chunks = (drop_blank_text(c) for c in raw_chunks)
        df_chunks = [normalise_labels(c) for c in clean_chunks]
        df = pd.concat(df_chunks, ignore_index=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=test_size, stratify=df["label"], random_state=42
    )

    pipe = build_pipeline().fit(X_train, y_train)
    print(classification_report(
        y_test, pipe.predict(X_test), target_names=INV_LABEL.values()))

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, Path(model_dir) / MODEL_FILE)
    print(f"[+] Model saved → {Path(model_dir) / MODEL_FILE}")

# ─── evaluation (NEW) ─────────────────────────────────────────────────────
def evaluate_model(csv_path: str, model_dir: str, out_dir: str, chunk_size: int):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # load & clean
    if chunk_size == 0:
        df = normalise_labels(
            drop_blank_text(
                standardise_columns(next(read_csv_any(csv_path)))))
    else:
        raw_chunks = (standardise_columns(c) for c in read_csv_any(csv_path, chunk_size))
        clean_chunks = (drop_blank_text(c) for c in raw_chunks)
        df_chunks = [normalise_labels(c) for c in clean_chunks]
        df = pd.concat(df_chunks, ignore_index=True)

    model = joblib.load(Path(model_dir) / MODEL_FILE)
    y_true = df["label"].astype(int).values
    y_prob = model.predict_proba(df["text"])[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4,3))
    ConfusionMatrixDisplay(cm, display_labels=["Safe","Phish"]).plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title("Confusion Matrix")
    fig_cm.tight_layout()
    fig_cm.savefig(out_path / "confusion_matrix.png", dpi=300)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(4,3))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0,1],[0,1],'--', lw=0.8)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve"); ax_roc.legend()
    fig_roc.tight_layout()
    fig_roc.savefig(out_path / "roc_curve.png", dpi=300)

    # PR
    fig_pr, ax_pr = plt.subplots(figsize=(4,3))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax_pr, name=None)
    ap = average_precision_score(y_true, y_prob)
    ax_pr.set_title(f"Precision-Recall (AP = {ap:.3f})")
    fig_pr.tight_layout()
    fig_pr.savefig(out_path / "precision_recall.png", dpi=300)

    print(f"[+] Plots saved to {out_path.resolve()}")

# ─── inference helpers ────────────────────────────────────────────────────
def load_model(model_dir: str):
    return joblib.load(Path(model_dir) / MODEL_FILE)

def predict_prob(model, text: str):
    p = model.predict_proba([text])[0, 1]
    return INV_LABEL[int(p >= 0.5)], p

# ─── Flask API ────────────────────────────────────────────────────────────
def create_app(model_dir: str) -> Flask:
    app = Flask(__name__)
    model = load_model(model_dir)

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True)
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text'"}), 400
        label, prob = predict_prob(model, data["text"])
        return jsonify({"prediction": label, "probability": prob})
    return app

# ─── CLI helpers ──────────────────────────────────────────────────────────
def predict_cli(model_dir: str, text: str | None, file: str | None):
    if file: text = Path(file).read_text(encoding="utf-8")
    if not text: raise ValueError("Provide --text or --file")
    label, prob = predict_prob(load_model(model_dir), text)
    print(json.dumps({"prediction": label, "probability": prob}, indent=2))

# ─── argparse setup ───────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(description="PhishSentry demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True)
    tr.add_argument("--model", required=True)
    tr.add_argument("--test-size", type=float, default=0.2)
    tr.add_argument("--chunksize", type=int, default=0)

    ev = sub.add_parser("evaluate", help="Generate confusion, ROC, PR plots")
    ev.add_argument("--data", required=True)
    ev.add_argument("--model", required=True)
    ev.add_argument("--out",   required=True, help="Output directory for PNGs")
    ev.add_argument("--chunksize", type=int, default=0)

    sv = sub.add_parser("serve")
    sv.add_argument("--model", required=True)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=5000)

    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True)
    g = pr.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file")
    return p

# ─── main ────────────────────────────────────────────────────────────────
def main():
    args = build_parser().parse_args()

    if args.cmd == "train":
        train_model(args.data, args.model, args.test_size, args.chunksize)

    elif args.cmd == "evaluate":
        evaluate_model(args.data, args.model, args.out, args.chunksize)

    elif args.cmd == "serve":
        app = create_app(args.model)
        print(f"[+] Serving on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port)

    elif args.cmd == "predict":
        predict_cli(args.model, args.text, getattr(args, "file", None))

if __name__ == "__main__":
    main()
