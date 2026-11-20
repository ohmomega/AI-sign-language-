"""Train a small landmark classifier and export it for the web app.

Usage:  python tools/train.py data/landmarks.json <dataset/train dir> app/
Writes app/model.json (weights for app.js) and app/signs/<A-Z>.jpg reference photos.
The feature function MUST stay identical to `features()` in app/app.js.
"""
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

RNG = np.random.default_rng(42)


def features(pts):
    """21x3 landmark array -> 63 normalized features (wrist origin, unit palm size)."""
    p = np.asarray(pts, dtype=np.float64).copy()
    p -= p[0]
    scale = np.linalg.norm(p[9, :2]) or 1.0  # wrist -> middle-finger knuckle
    p /= scale
    return p.reshape(-1)


def variants(pts):
    """Mirror variants so left/right hands and front/back views all look alike."""
    p = np.asarray(pts, dtype=np.float64)
    out = []
    for sx, sz in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        q = p.copy()
        q[:, 0] *= sx
        q[:, 2] *= sz
        out.append(q)
    return out


def jitter(pts):
    """Small random in-plane rotation + per-point noise."""
    p = np.asarray(pts, dtype=np.float64).copy()
    a = RNG.uniform(-0.25, 0.25)
    c, s = np.cos(a), np.sin(a)
    p[:, :2] = p[:, :2] @ np.array([[c, -s], [s, c]]).T
    p += RNG.normal(0, 0.01 * np.linalg.norm(p[9, :2] - p[0, :2]), p.shape)
    return p


def build(records, augment):
    X, y = [], []
    for r in records:
        for v in variants(r["lm"]):
            X.append(features(v))
            y.append(r["label"])
            if augment:
                for _ in range(3):
                    X.append(features(jitter(v)))
                    y.append(r["label"])
    return np.array(X), np.array(y)


def main():
    data = json.loads(pathlib.Path(sys.argv[1]).read_text())
    src = pathlib.Path(sys.argv[2])
    app = pathlib.Path(sys.argv[3])
    recs = data["records"]
    labels = sorted({r["label"] for r in recs})
    print(f"{len(recs)} hands, {len(data['missed'])} images with no hand found")

    train_r, test_r = train_test_split(
        recs, test_size=0.2, random_state=42, stratify=[r["label"] for r in recs])
    Xtr, ytr = build(train_r, augment=True)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=400,
                        early_stopping=True, random_state=42)
    clf.fit(Xtr, ytr)

    # Honest evaluation: held-out photos, original orientation only.
    Xte = np.array([features(r["lm"]) for r in test_r])
    yte = np.array([r["label"] for r in test_r])
    pred = clf.predict(Xte)
    print(f"held-out accuracy: {accuracy_score(yte, pred):.3f}  ({len(test_r)} photos never seen in training)")
    print(classification_report(yte, pred, digits=2, zero_division=0))
    cm = confusion_matrix(yte, pred, labels=labels)
    pairs = [(cm[i, j], labels[i], labels[j]) for i in range(len(labels))
             for j in range(len(labels)) if i != j and cm[i, j]]
    print("most confused (true -> predicted):", sorted(pairs, reverse=True)[:8])

    # Final model on all photos.
    Xall, yall = build(recs, augment=True)
    clf.fit(Xall, yall)
    model = {
        "labels": [str(c) for c in clf.classes_],
        "layers": [{"W": W.round(5).tolist(), "b": b.round(5).tolist()}
                   for W, b in zip(clf.coefs_, clf.intercepts_)],
    }
    (app / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    print("wrote", app / "model.json")

    # Reference photo per letter: the most confidently-recognized, largest hand.
    signs = app / "signs"
    signs.mkdir(parents=True, exist_ok=True)
    probs = clf.predict_proba(np.array([features(r["lm"]) for r in recs]))
    idx = {c: i for i, c in enumerate(clf.classes_)}
    for letter in labels:
        best = max((r for r in range(len(recs)) if recs[r]["label"] == letter),
                   key=lambda r: probs[r][idx[letter]] + 0.0001 * (recs[r]["bbox"][2] - recs[r]["bbox"][0]))
        r = recs[best]
        img = Image.open(src / r["file"])
        img.draft("RGB", (960, 960))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((960, 960))
        x0, y0, x1, y1 = r["bbox"]
        W, H = img.size
        half = min(max(x1 - x0, y1 - y0) * 0.7, W / 2, H / 2)
        # keep the square inside the photo so there are no black bars
        cx = min(max((x0 + x1) / 2, half), W - half)
        cy = min(max((y0 + y1) / 2, half), H - half)
        crop = img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
        crop.resize((320, 320), Image.LANCZOS).save(signs / f"{letter}.jpg", quality=82)
    print("wrote reference photos to", signs)


if __name__ == "__main__":
    main()
