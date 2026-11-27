"""Run distributed detector models on the original COCO 2017 validation archives."""
import argparse
import io
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

INPUT_SIZE = 640
SCORE_FLOOR = 0.001
NMS_IOU = 0.7
CANDIDATE_LIMIT = 300


def prepare(data):
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        width, height = image.size
        ratio = INPUT_SIZE / max(width, height)
        new_width, new_height = round(width * ratio), round(height * ratio)
        x = (INPUT_SIZE - new_width) // 2
        y = (INPUT_SIZE - new_height) // 2
        canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (114, 114, 114))
        canvas.paste(image.resize((new_width, new_height), Image.Resampling.BILINEAR), (x, y))
    tensor = np.ascontiguousarray(np.asarray(canvas).transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255
    return tensor, ((width, height), (x, y), (new_width / width, new_height / height))


def top_indices(scores, count):
    subset = np.argpartition(-scores, count - 1)[:count] if count < len(scores) else np.arange(len(scores))
    return subset[np.lexsort((subset, -scores[subset]))]


def decode(boxes, scores, mode, geometry):
    if mode == "nms_free":
        candidate = top_indices(scores.max(axis=1), CANDIDATE_LIMIT)
        subset = scores[candidate]
        flat = top_indices(subset.ravel(), CANDIDATE_LIMIT)
        row, classes = np.unravel_index(flat, subset.shape)
        confidence = subset[row, classes]
        indices = candidate[row]
        keep = confidence > SCORE_FLOOR
        indices, classes, confidence = indices[keep], classes[keep], confidence[keep]
    else:
        classes = scores.argmax(axis=1)
        confidence = scores[np.arange(len(classes)), classes]
        indices = np.flatnonzero(confidence > SCORE_FLOOR)
        boxes_for_nms = boxes[indices].copy()
        if not len(indices):
            return boxes_for_nms, classes[indices], confidence[indices]
        # OpenCV offsets class coordinates by their maximum, so ensure all pre-NMS coordinates are nonnegative.
        boxes_for_nms -= min(0, float(boxes_for_nms.min()))
        boxes_for_nms[:, 2:] -= boxes_for_nms[:, :2]
        keep = np.asarray(cv2.dnn.NMSBoxesBatched(
            boxes_for_nms.tolist(), confidence[indices].tolist(), classes[indices].tolist(), SCORE_FLOOR, NMS_IOU,
        ), dtype=np.int64).ravel()[:CANDIDATE_LIMIT]
        indices = indices[keep]
        classes, confidence = classes[indices], confidence[indices]
    size, offset, scale = geometry
    xyxy = (boxes[indices] - np.tile(offset, 2)) / np.tile(scale, 2)
    xyxy = np.clip(xyxy, 0, np.tile(size, 2))
    xyxy[:, 2:] -= xyxy[:, :2]
    valid = np.all(xyxy[:, 2:] > 0, axis=1)
    return xyxy[valid], classes[valid], confidence[valid]


def make_onnx_runner(args):
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if args.cuda is not None:
        ort.preload_dlls()
        providers.insert(0, ("CUDAExecutionProvider", {"device_id": args.cuda, "use_tf32": 0}))
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=providers)
    if args.cuda is not None and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("The CUDA provider was not initialized")

    def infer(tensor):
        return session.run(["boxes", "scores"], {"images": tensor})

    return infer


def make_torch_runner(args):
    import torch

    sys.path.insert(0, str(args.torch_source.resolve()))
    from core import load_torch_model

    torch.set_num_threads(args.threads)
    if args.cuda is not None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    device = "cpu" if args.cuda is None else f"cuda:{args.cuda}"
    model = load_torch_model(args.model, args.mode, device=device).float()

    @torch.inference_mode()
    def infer(tensor):
        outputs = model(torch.from_numpy(tensor).to(device=device), branch=args.mode)[args.mode]
        return outputs["boxes"].float().cpu().numpy(), outputs["logits"].float().sigmoid().cpu().numpy()

    return infer


def run(args):
    if not 1 <= args.limit <= 5000:
        raise ValueError("--limit must be between 1 and 5000")
    infer = make_onnx_runner(args) if args.torch_source is None else make_torch_runner(args)
    with ZipFile(args.dataset / "annotations_trainval2017.zip") as archive:
        gt = json.loads(archive.read("annotations/instances_val2017.json"))
    categories = sorted(cat["id"] for cat in gt["categories"])
    assert len(gt["images"]) == 5000 and len(categories) == 80
    images = sorted(gt["images"], key=lambda record: record["id"])[:args.limit]
    start = time.monotonic()
    with ZipFile(args.dataset / "val2017.zip") as archive, args.predictions.open("w", encoding="utf-8") as stream:
        for count, record in enumerate(images, start=1):
            image_id = record["id"]
            tensor, geometry = prepare(archive.read("val2017/" + record["file_name"]))
            assert geometry[0] == (record["width"], record["height"])
            boxes, scores = infer(tensor)
            if boxes.shape != (1, 8400, 4) or scores.shape != (1, 8400, 80):
                raise RuntimeError(f"Unexpected output shape in image {image_id}: {boxes.shape}, {scores.shape}")
            if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
                raise RuntimeError(f"Nonfinite output in image {image_id}")
            if not (scores.min() >= 0 and scores.max() <= 1):
                raise RuntimeError(f"Nonprobability score in image {image_id}")
            xywh, labels, confidence = decode(boxes[0], scores[0], args.mode, geometry)
            detections = [{"image_id": image_id, "category_id": categories[label],
                           "bbox": box.tolist(), "score": float(score)}
                          for box, label, score in zip(xywh, labels, confidence)]
            stream.write(json.dumps([image_id, detections], separators=(",", ":"), allow_nan=False) + "\n")
            if count % 100 == 0 or count == len(images):
                stream.flush()
                print(f"{args.mode} {count}/{len(images)} elapsed={time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["nms", "nms_free"])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--cuda", type=int)
    parser.add_argument("--torch-source", type=Path, help="Use verified project-side Torch loader from this directory")
    run(parser.parse_args())
