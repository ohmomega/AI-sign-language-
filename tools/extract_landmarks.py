"""Extract MediaPipe hand landmarks from the sign-language photo dataset.

Usage:  python tools/extract_landmarks.py <dataset/train dir> <out.json> [limit]
Each record: label, file, handedness, 21 landmarks as [x_px, y_px, z_px]
(x/y in pixels of the downscaled image, z scaled like x), and the hand bbox.
"""
import json
import pathlib
import sys
import time

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps

MODEL = pathlib.Path(__file__).resolve().parent.parent / "app" / "hand_landmarker.task"
MAX_SIDE = 960


def load(path):
    img = Image.open(path)
    img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # fast JPEG downscale on decode
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def main():
    src = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    files = sorted(p for p in src.glob("*/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if limit:
        files = files[::max(1, len(files) // limit)][:limit]

    records, missed = [], []
    t0 = time.time()
    with vision.HandLandmarker.create_from_options(opts) as det:
        for i, f in enumerate(files):
            img = load(f)
            w, h = img.size
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img)))
            if not res.hand_landmarks:
                missed.append(str(f.relative_to(src)))
                continue
            lm = res.hand_landmarks[0]
            pts = [[p.x * w, p.y * h, p.z * w] for p in lm]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            records.append({
                "label": f.parent.name,
                "file": str(f.relative_to(src)).replace("\\", "/"),
                "hand": res.handedness[0][0].category_name,
                "size": [w, h],
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "lm": [[round(v, 2) for v in p] for p in pts],
            })
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(files)}  {time.time() - t0:.0f}s", flush=True)

    out.write_text(json.dumps({"records": records, "missed": missed}))
    print(f"done: {len(records)} hands, {len(missed)} missed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
