import glob
import html
import os
import re

import joblib
import torch
from flask import Flask, jsonify, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


def first_dir(*cands):
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return BASE


MODEL_DIR = first_dir(os.environ.get("MODEL_DIR"), os.path.join(BASE, "models"), BASE)
PAGE_DIR = first_dir(*[d for d in (os.path.join(BASE, "templates"), BASE)
                       if os.path.exists(os.path.join(d, "index.html"))])
print("model folder:", MODEL_DIR)
print("page folder :", PAGE_DIR)


def find(pattern: str) -> str:
    """Match by prefix so slightly different file names still work."""
    hits = sorted(glob.glob(os.path.join(MODEL_DIR, pattern)))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern} in {MODEL_DIR}")
    return hits[0]


def load_bundle(pattern: str):
    """joblib.load with a clear error if the file is a Git LFS pointer."""
    path = find(pattern)
    with open(path, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"{os.path.basename(path)} is a Git LFS pointer, not the real model "
            f"({os.path.getsize(path)} bytes). Run: git lfs install && git lfs pull")
    return joblib.load(path)


def clean_text(text: str) -> str:          # identical to the training-time cleaning
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ----------------------------------------------------------------- model loaders
def load_baseline():
    b = load_bundle("baseline*.joblib")
    vec, clf = b["vec"], b["model"]
    return lambda t: float(clf.predict_proba(vec.transform([t]))[0, 1])


def load_embedding():
    from sentence_transformers import SentenceTransformer
    b = load_bundle("embedding*.joblib")
    enc = SentenceTransformer(b["encoder_name"], device="cpu")   # downloads MiniLM on first run
    enc.max_seq_length = b.get("max_seq_length", 256)
    clf = b["model"]
    return lambda t: float(clf.predict_proba(
        enc.encode([t], normalize_embeddings=True))[0, 1])


def load_distilbert():
    import transformers
    b = load_bundle("distilbert*.joblib")
    saved = b.get("versions", {})
    if saved:
        print(f"distilbert saved with torch {saved.get('torch')}, "
              f"transformers {saved.get('transformers')} | "
              f"here: torch {torch.__version__}, transformers {transformers.__version__}")
    model, tok, max_len = b["model"].cpu().eval(), b["tokenizer"], b["max_len"]

    def predict(t):
        enc = tok([t], truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            return float(torch.softmax(model(**enc).logits, dim=1)[0, 1])
    return predict


PREDICTORS, LOAD_ERRORS = {}, {}
for key, loader in [("baseline", load_baseline), ("embedding", load_embedding),
                    ("distilbert", load_distilbert)]:
    try:
        PREDICTORS[key] = loader()
        print(f"loaded {key}")
    except Exception as e:                   
        LOAD_ERRORS[key] = f"{type(e).__name__}: {e}"
        print(f"COULD NOT LOAD {key} -> {LOAD_ERRORS[key]}")


# ----------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(PAGE_DIR, "index.html")


@app.post("/predict")
def predict():
    text = clean_text((request.get_json(silent=True) or {}).get("text", ""))[:20000]
    if len(text.split()) < 3:
        return jsonify({"error": "Write at least a few words."}), 400
    results = {}
    for key in ["baseline", "embedding", "distilbert"]:
        if key in PREDICTORS:
            p = PREDICTORS[key](text)
            results[key] = {"p_positive": round(p, 4),
                            "label": "positive" if p >= 0.5 else "negative"}
        else:
            results[key] = {"error": LOAD_ERRORS.get(key, "model not loaded")}
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
