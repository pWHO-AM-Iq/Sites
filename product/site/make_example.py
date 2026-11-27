"""Run the NMS-free ONNX on an original illustration; export normalized public boxes."""
import argparse
import hashlib
import json
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--annotations", required=True, type=Path)
args = parser.parse_args()
if args.annotations.suffix == ".zip":
    with ZipFile(args.annotations) as archive:
        annotations = json.loads(archive.read("annotations/instances_val2017.json"))
else:
    annotations = json.loads(args.annotations.read_text("utf-8"))
labels = [c["name"] for c in sorted(annotations["categories"], key=lambda c: c["id"])]
image_path = root / "main/dist/assets/crossing.jpg"
image = Image.open(image_path).convert("RGB")
w, h = image.size
ratio = min(640 / w, 640 / h)
nw, nh = round(w * ratio), round(h * ratio)
left, top = (640 - nw) // 2, (640 - nh) // 2
canvas = Image.new("RGB", (640, 640), (114, 114, 114))
canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
options = ort.SessionOptions()
options.intra_op_num_threads = 1
weights = root / "models/custom_detector/custom_detector_nms_free.onnx"
session = ort.InferenceSession(str(weights), options, providers=["CPUExecutionProvider"])
boxes, scores = session.run(["boxes", "scores"], {"images": tensor})
scores = scores[0]
anchors = np.argsort(-scores.max(axis=1), kind="stable")[:300]
subset = scores[anchors]
results = []
for flat in np.argsort(-subset.reshape(-1), kind="stable")[:300]:
    row, cls = np.unravel_index(flat, subset.shape)
    score = float(subset[row, cls])
    if score < 0.5:
        break
    box = (boxes[0, anchors[row]] - [left, top, left, top]) / [nw, nh, nw, nh]
    results.append({"label": labels[cls], "score": round(score, 4), "xyxy": np.clip(box, 0, 1).round(5).tolist()})
summary = {"variant": "nms_free", "threshold": 0.5, "image_size": [w, h],
           "input_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
           "onnx_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "detections": results}
(root / "main/dist/data/example.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print([(r["label"], r["score"]) for r in results])
