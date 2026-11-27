"""Summarize complete per-image predictions with the official COCO bbox evaluator."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from predict_coco import CANDIDATE_LIMIT, INPUT_SIZE, NMS_IOU, SCORE_FLOOR

OPERATING_POINTS = (0.10, 0.25, 0.35, 0.50, 0.75)
STAT_NAMES = (
    "ap", "ap50", "ap75", "apSmall", "apMedium", "apLarge", "ar1", "ar10", "ar100",
    "arSmall", "arMedium", "arLarge",
)


def mean_valid(values):
    valid = values[values >= 0]
    return float(valid.mean()) if len(valid) else None


def operating_stats(tp, fp, ground_truth):
    fn = ground_truth - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / ground_truth if ground_truth else None
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def class_metrics(evaluation, truth):
    params = evaluation.params
    precision = evaluation.eval["precision"]
    recall = evaluation.eval["recall"]
    all_areas = params.areaRngLbl.index("all")
    max_det = params.maxDets.index(100)
    iou50 = int(np.flatnonzero(np.isclose(params.iouThrs, 0.50))[0])
    iou75 = int(np.flatnonzero(np.isclose(params.iouThrs, 0.75))[0])
    # evalImgs is ordered category -> area range -> image. Matching rows apply official crowd handling.
    image_count = len(params.imgIds)
    category_block = len(params.areaRng) * image_count
    classes = []
    for index, category in enumerate(params.catIds):
        start = index * category_block + all_areas * image_count
        matches = [record for record in evaluation.evalImgs[start:start + image_count] if record is not None]
        ground_truth = int(sum((~record["gtIgnore"].astype(bool)).sum() for record in matches))
        scores = np.concatenate([record["dtScores"] for record in matches])
        true_positive = np.concatenate([record["dtMatches"][iou50] > 0 for record in matches])
        keep = np.concatenate([~record["dtIgnore"][iou50] for record in matches])
        operating_points = []
        for threshold in OPERATING_POINTS:
            selected = (scores > threshold) & keep
            tp = int(np.count_nonzero(selected & true_positive))
            fp = int(np.count_nonzero(selected & ~true_positive))
            operating_points.append({"threshold": threshold, **operating_stats(tp, fp, ground_truth)})
        precisions = precision[iou50, :, index, all_areas, max_det]
        classes.append({
            "id": int(category), "name": truth.cats[category]["name"], "groundTruth": ground_truth,
            "ap": mean_valid(precision[:, :, index, all_areas, max_det]),
            "ap50": mean_valid(precisions),
            "ap75": mean_valid(precision[iou75, :, index, all_areas, max_det]),
            "ar100": mean_valid(recall[:, index, all_areas, max_det]),
            "operatingPoints": operating_points,
            "pr50": [float(value) if value >= 0 else None for value in precisions],
        })
    return classes


def score(truth, predictions):
    seen = set()
    valid_categories = set(truth.getCatIds())
    results = []
    expected = set(truth.getImgIds())
    if predictions.suffix == ".gz":
        stream_source = gzip.open(predictions, "rt", encoding="utf-8")
    else:
        stream_source = predictions.open(encoding="utf-8")
    with stream_source as stream:
        for line in stream:
            image_id, detections = json.loads(line)
            if image_id in seen or image_id not in expected or len(detections) > CANDIDATE_LIMIT:
                raise ValueError(f"Invalid image or candidate count {image_id}")
            seen.add(image_id)
            image = truth.imgs[image_id]
            for detection in detections:
                x, y, width, height = detection["bbox"]
                if (detection["image_id"] != image_id or detection["category_id"] not in valid_categories
                    or not SCORE_FLOOR < detection["score"] <= 1 or not 0 <= x < x + width <= image["width"] + 1e-7
                    or not 0 <= y < y + height <= image["height"] + 1e-7):
                    raise ValueError(f"Invalid detection {detection}")
            results.extend(detections)
    if seen != expected:
        raise ValueError(f"Incomplete predictions, scored {len(seen)}/{len(expected)} images")
    if results:
        detection_set = truth.loadRes(results)
    else:
        detection_set = COCO()
        detection_set.dataset = {
            "images": truth.dataset["images"], "categories": truth.dataset["categories"], "annotations": [],
        }
        detection_set.createIndex()
    evaluation = COCOeval(truth, detection_set, "bbox")
    evaluation.params.imgIds = sorted(expected)
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    classes = class_metrics(evaluation, truth)
    micro = []
    ground_truth = sum(item["groundTruth"] for item in classes)
    for index, threshold in enumerate(OPERATING_POINTS):
        tp = sum(item["operatingPoints"][index]["tp"] for item in classes)
        fp = sum(item["operatingPoints"][index]["fp"] for item in classes)
        micro.append({"threshold": threshold, **operating_stats(tp, fp, ground_truth)})
    return {
        "status": "measured",
        "imagesEvaluated": len(seen),
        "detections": len(results),
        "aggregate": dict(zip(STAT_NAMES, map(float, evaluation.stats))),
        "operatingPoints": micro,
        "classes": classes,
    }


def main(args):
    with ZipFile(args.dataset / "annotations_trainval2017.zip") as archive:
        payload = archive.read("annotations/instances_val2017.json")
    truth = COCO()
    truth.dataset = json.loads(payload)
    truth.createIndex()
    image_count = len(truth.imgs)
    category_count = len(truth.cats)
    if image_count != 5000 or category_count != 80:
        raise ValueError("Expected the complete COCO 2017 validation set")
    models = {}
    output = {
        "schemaVersion": 1,
        "dataset": {
            "name": "COCO 2017", "split": "val2017", "imageCount": image_count,
            "annotationCount": len(truth.anns), "categoryCount": category_count,
            "crowdAnnotationCount": sum(bool(a["iscrowd"]) for a in truth.anns.values()),
            "annotationsSha256": hashlib.sha256(payload).hexdigest(),
        },
        "protocol": {
            "task": "bbox", "inputSize": [INPUT_SIZE, INPUT_SIZE], "scoreFloor": SCORE_FLOOR,
            "nmsIou": NMS_IOU, "candidateLimitPerImage": CANDIDATE_LIMIT, "maxDets": [1, 10, 100],
            "ious": np.linspace(0.50, 0.95, 10).round(2).tolist(),
            "recallPoints": np.linspace(0.0, 1.0, 101).round(2).tolist(),
            "operatingPoints": list(OPERATING_POINTS), "operatingPointIou": 0.50,
            "pycocotoolsVersion": importlib.metadata.version("pycocotools"),
        },
        "models": models,
    }
    specs = json.loads(args.specs.read_text(encoding="utf-8"))["models"] if args.specs else {}
    if args.nms is not None:
        models["nms"] = {**score(truth, args.nms), "specs": specs.get("nms")}
    if args.nms_free is not None:
        models["nms_free"] = {**score(truth, args.nms_free), "specs": specs.get("nms_free")}
    if not models:
        raise ValueError("Supply one or more predictions files")
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--nms", type=Path)
    parser.add_argument("--nms-free", type=Path)
    parser.add_argument("--specs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
