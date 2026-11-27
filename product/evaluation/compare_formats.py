"""Compare COCO score files for the same predictions protocol after re-export."""
import argparse
import json
from pathlib import Path

METADATA_KEY = "inference"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(onnx, torch):
    onnx_protocol = {key: value for key, value in onnx["protocol"].items() if key != METADATA_KEY}
    torch_protocol = {key: value for key, value in torch["protocol"].items() if key != METADATA_KEY}
    if onnx["dataset"] != torch["dataset"] or onnx_protocol != torch_protocol:
        raise ValueError("COCO dataset and postprocessing protocol must match")
    result = {"schemaVersion": 1, "dataset": onnx["dataset"], "protocol": onnx_protocol, "models": {}}
    for mode in sorted(torch["models"]):
        original = torch["models"][mode]
        export = onnx["models"][mode]
        if original["imagesEvaluated"] != export["imagesEvaluated"]:
            raise ValueError(f"Evaluated image counts differ in {mode}")
        original_classes = {item["id"]: item for item in original["classes"]}
        if {item["id"] for item in export["classes"]} != original_classes.keys():
            raise ValueError(f"Evaluated category IDs differ in {mode}")
        category_deltas = []
        for item in export["classes"]:
            reference = original_classes[item["id"]]
            if (item["name"], item["groundTruth"]) != (reference["name"], reference["groundTruth"]):
                raise ValueError(f"Category metadata differs in {mode} for {item['id']}")
            category_deltas.append({
                "id": item["id"], "name": item["name"], "torchAp": reference["ap"], "onnxAp": item["ap"],
                "onnxMinusTorchApPoints": 100 * (item["ap"] - reference["ap"]),
            })
        result["models"][mode] = {
            "imagesEvaluated": original["imagesEvaluated"],
            "torchDetections": original["detections"], "onnxDetections": export["detections"],
            "torchAggregate": original["aggregate"], "onnxAggregate": export["aggregate"],
            "onnxMinusTorchPoints": {key: 100 * (val - original["aggregate"][key])
                                     for key, val in export["aggregate"].items()},
            "torchOperatingPoints": original["operatingPoints"], "onnxOperatingPoints": export["operatingPoints"],
            "classes": category_deltas,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--torch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(load(args.onnx), load(args.torch))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
