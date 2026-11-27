"""Summarize batch-1 runtime runs and compare their decoded detections."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(left, right):
    left = left["steady_state"]["sample_detections"]
    right = right["steady_state"]["sample_detections"]
    if set(left) != set(right):
        raise ValueError("Runtime reports need the same sample set for parity comparison")
    paired, unmatched_left, unmatched_right = 0, 0, 0
    max_coord_delta, max_confidence_delta = None, None
    for name in left:
        a = np.asarray(left[name]).reshape(-1, 6)
        b = np.asarray(right[name]).reshape(-1, 6)
        available = set(range(len(b)))
        for row in a:
            best_iou, best_index = 0.5, None
            for j in sorted(available):
                other = b[j]
                if row[5] != other[5]:
                    continue
                wh = np.maximum(0, np.minimum(row[2:4], other[2:4]) - np.maximum(row[:2], other[:2]))
                inter = wh.prod()
                union = (row[2:4] - row[:2]).prod() + (other[2:4] - other[:2]).prod() - inter
                iou = inter / max(union, 1e-12)
                if iou >= best_iou:
                    best_iou, best_index = iou, j
            if best_index is None:
                unmatched_left += 1
                continue
            other = b[best_index]
            available.remove(best_index)
            paired += 1
            coord_delta = float(np.abs(row[:4] - other[:4]).max())
            confidence_delta = abs(float(row[4] - other[4]))
            max_coord_delta = coord_delta if max_coord_delta is None else max(max_coord_delta, coord_delta)
            max_confidence_delta = confidence_delta if max_confidence_delta is None else max(max_confidence_delta,
                                                                                              confidence_delta)
        unmatched_right += len(available)
    return {"images": len(left), "matched_by_class_and_iou_0_5": paired,
            "unmatched_reference": unmatched_left, "unmatched_candidate": unmatched_right,
            "max_matched_coordinate_delta_px": max_coord_delta,
            "max_matched_score_delta": max_confidence_delta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = [{"backend": backend, "mode": mode} for backend in ("ort-cpu", "ort-cuda", "ort-trt",
                                                                "torch-cpu", "torch-fp32", "torch-fp16")
             for mode in ("nms", "nms_free")]
    inventory_path = args.directory / "environment.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else {}
    reports = {}
    for pair in pairs:
        name = f'{pair["backend"]}-{pair["mode"]}'
        path = args.directory / f"{name}.json"
        if path.exists():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            reports[name] = pair | {"status": "not_measured"}
            if pair["backend"].startswith("torch") and inventory.get("pytorch_model_source_available") is False:
                reports[name]["reason"] = "Original model definition and checkpoint loader are pending handoff."
            if pair["backend"] == "ort-trt" and inventory.get("tensorrt_runtime_loadable") is False:
                reports[name]["reason"] = "TensorRT runtime DLLs could not be resolved in the local environment."
    parity = {}
    pytorch_parity = {}
    for mode in ("nms", "nms_free"):
        cpu = reports[f"ort-cpu-{mode}"]
        cuda = reports[f"ort-cuda-{mode}"]
        if cpu["status"] == cuda["status"] == "completed":
            compared = compare(cpu, cuda)
            compared["unmatched_cpu"] = compared.pop("unmatched_reference")
            compared["unmatched_cuda"] = compared.pop("unmatched_candidate")
            parity[mode] = compared
        if cpu["status"] == "completed":
            for backend in ("torch-cpu", "torch-fp32", "torch-fp16"):
                key = f"{backend}-{mode}"
                if reports[key]["status"] == "completed":
                    pytorch_parity[key] = compare(cpu, reports[key])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema_version": 1, "runs": reports,
                                       "cpu_cuda_parity": parity, "pytorch_vs_onnx_cpu_parity": pytorch_parity,
                                       "environment": inventory}, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
