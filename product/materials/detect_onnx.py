"""Run person detection on a local image using the supplied ONNX model."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

SIZE = 640
LIMIT = 300
PERSON = 0


def prepare(image):
    width, height = image.size
    gain = min(SIZE / width, SIZE / height)
    resized_width = max(1, round(width * gain))
    resized_height = max(1, round(height * gain))
    x = (SIZE - resized_width) // 2
    y = (SIZE - resized_height) // 2
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    canvas.paste(image.resize((resized_width, resized_height), Image.Resampling.BILINEAR), (x, y))
    pixels = np.array(canvas, dtype=np.float32).transpose(2, 0, 1) / 255
    return pixels[None], (x, y, resized_width / width, resized_height / height)


def nms(boxes, scores, threshold=0.7):
    order = np.argsort(-scores, kind="stable")
    areas = np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    keep = []
    while order.size and len(keep) < LIMIT:
        first, remaining = order[0], order[1:]
        keep.append(first)
        lo = np.maximum(boxes[first, :2], boxes[remaining, :2])
        hi = np.minimum(boxes[first, 2:], boxes[remaining, 2:])
        overlap = np.prod(np.maximum(0, hi - lo), axis=1)
        union = areas[first] + areas[remaining] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        order = remaining[iou <= threshold]
    return keep


def decode_people(boxes, scores, mode, threshold, geometry, image_size):
    if mode == "nms":
        people = np.flatnonzero((scores.argmax(axis=1) == PERSON) & (scores[:, PERSON] > threshold))
        confidences = scores[people, PERSON]
        keep = nms(boxes[people], confidences)
        selected_boxes, confidences = boxes[people[keep]].copy(), confidences[keep]
    else:
        candidates = np.argsort(-scores.max(axis=1), kind="stable")[:LIMIT]
        candidate_scores = scores[candidates].ravel()
        top = np.argsort(-candidate_scores, kind="stable")[:LIMIT]
        rows, classes = np.divmod(top, scores.shape[1])
        people = (classes == PERSON) & (candidate_scores[top] > threshold)
        selected_boxes = boxes[candidates[rows[people]]].copy()
        confidences = candidate_scores[top[people]]
    x, y, scale_x, scale_y = geometry
    width, height = image_size
    selected_boxes[:, 0::2] = np.clip((selected_boxes[:, 0::2] - x) / scale_x, 0, width)
    selected_boxes[:, 1::2] = np.clip((selected_boxes[:, 1::2] - y) / scale_y, 0, height)
    return np.column_stack((selected_boxes, confidences)).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--mode", choices=("nms", "nms_free"), default="nms_free")
    parser.add_argument("--score", type=float, default=0.35)
    args = parser.parse_args()
    if not 0 <= args.score <= 1:
        parser.error("--score must be between 0 and 1")
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    tensor, geometry = prepare(image)
    model = Path(__file__).parent / f"custom_detector_{args.mode}.onnx"
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    boxes, scores = session.run(["boxes", "scores"], {"images": tensor})
    detections = decode_people(boxes[0], scores[0], args.mode, args.score, geometry, image.size)
    print(json.dumps({"mode": args.mode, "people_xyxy_score": detections}, allow_nan=False))


if __name__ == "__main__":
    main()
