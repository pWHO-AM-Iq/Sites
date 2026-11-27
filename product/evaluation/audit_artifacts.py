"""Audit split identity, checkpoint metadata and ONNX graph statistics without model code."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import onnx
import torch
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

MODEL_MODES = {"nms": 195, "nms_free": 205}


def split_audit(root):
    with ZipFile(root / "annotations_trainval2017.zip") as archive:
        annotation = json.loads(archive.read("annotations/instances_val2017.json"))
    images = annotation["images"]
    expected_val_names = {f'val2017/{image["file_name"]}' for image in images}
    category_ids = sorted(item["id"] for item in annotation["categories"])
    with ZipFile(root / "train2017.zip") as train, ZipFile(root / "val2017.zip") as val:
        train_files = [item for item in train.infolist() if item.filename.lower().endswith(".jpg")]
        val_files = [item for item in val.infolist() if item.filename.lower().endswith(".jpg")]
        train_ids = {int(PurePosixPath(item.filename).stem) for item in train_files}
        val_ids = {int(PurePosixPath(item.filename).stem) for item in val_files}
        candidates = defaultdict(list)
        for info in train_files:
            candidates[(info.file_size, info.CRC)].append(info)
        exact_duplicates = 0
        for info in val_files:
            matches = candidates.get((info.file_size, info.CRC), [])
            if matches:
                image_bytes = val.read(info)
                exact_duplicates += sum(train.read(candidate) == image_bytes for candidate in matches)
    annotations_valid = all(item["image_id"] in val_ids and item["category_id"] in category_ids
                            for item in annotation["annotations"])
    return {
        "trainArchiveImages": len(train_files), "validationArchiveImages": len(val_files),
        "trainUniqueIds": len(train_ids), "validationUniqueIds": len(val_ids),
        "overlappingImageIds": len(train_ids & val_ids),
        "exactByteDuplicatePairs": exact_duplicates,
        "validationArchiveMatchesAnnotationFilenames": expected_val_names == {i.filename for i in val_files},
        "validationArchiveMatchesAnnotationIds": val_ids == {i["id"] for i in images},
        "validationAnnotationReferencesValid": annotations_valid,
        "validationCategoryIds": category_ids,
        "trainingRecordsAvailable": False,
    }


def value_shape(value):
    return [dimension.dim_value or None for dimension in value.type.tensor_type.shape.dim]


def operation_counts(model):
    graph = SymbolicShapeInference.infer_shapes(model, auto_merge=True, guess_output_rank=True).graph
    shapes = {value.name: value_shape(value) for value in [*graph.input, *graph.output, *graph.value_info]}
    shapes.update((value.name, list(value.dims)) for value in graph.initializer)
    macs = Counter()
    counted = Counter()
    missed = Counter()
    for node in graph.node:
        if node.op_type not in {"Conv", "MatMul"}:
            continue
        weights = shapes.get(node.input[1], [])
        result = shapes.get(node.output[0], [])
        if len(weights) < 2 or not result or None in weights or None in result:
            missed[node.op_type] += 1
            continue
        contraction = math.prod(weights[1:]) if node.op_type == "Conv" else weights[-2]
        macs[node.op_type] += math.prod(result) * contraction
        counted[node.op_type] += 1
    return {
        "macs": sum(macs.values()), "convolutionMacs": macs["Conv"], "matmulMacs": macs["MatMul"],
        "multiplyAddAsTwoFlops": 2 * sum(macs.values()), "operatorsCounted": dict(counted),
        "operatorsWithUnknownShapes": dict(missed),
        "scope": "ONNX Conv and MatMul at the exported fixed input shape; fused bias and other operators excluded",
    }


def file_info(path):
    with path.open("rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": fingerprint}


def specs(root, mode, category_ids):
    graph_path = root / f"custom_detector_{mode}.onnx"
    graph = onnx.load(str(graph_path))
    graph_shapes = {value.name: value_shape(value) for value in [*graph.graph.input, *graph.graph.output]}
    weights_path = root / f"custom_detector_{mode}.pt"
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    assert checkpoint["category_ids"] == category_ids and checkpoint["mode"] == mode
    counts = checkpoint["parameter_counts"]
    parameters = sum(tensor.numel() for key, tensor in checkpoint["model"].items()
                     if not key.endswith(("running_mean", "running_var", "num_batches_tracked")))
    assert parameters == counts[f"{mode}_inference"]
    return {
        "architectureId": checkpoint["architecture_id"], "upstreamCommit": checkpoint["upstream_commit"],
        "selectedEpochReadme": MODEL_MODES[mode], "trainingParametersMetadata": counts["training"],
        "inferenceParametersMetadata": counts[f"{mode}_inference"], "sharedParametersMetadata": counts["shared"],
        "headParametersMetadata": counts[f"{mode}_head"],
        "inferenceParameterElementsVerified": parameters,
        "onnxInitializerElements": sum(math.prod(item.dims) for item in graph.graph.initializer),
        "onnxOperators": dict(Counter(node.op_type for node in graph.graph.node)),
        "onnxOpset": next(entry.version for entry in graph.opset_import if not entry.domain),
        "onnxExportFramework": f"{graph.producer_name} {graph.producer_version}",
        "onnxShapes": graph_shapes,
        "complexity": operation_counts(graph),
        "checkpointCategoryOrderVerified": True,
        "files": {"onnx": file_info(graph_path), "pt": file_info(weights_path)},
    }


def compare_states(root):
    torch.set_num_threads(2)
    states = [torch.load(root / f"custom_detector_{mode}.pt", map_location="cpu", weights_only=True)["model"]
              for mode in MODEL_MODES]
    first, second = states
    common = first.keys() & second.keys()
    identical = sum(torch.equal(first[key], second[key]) for key in common)
    return {
        "commonTensorKeys": len(common), "commonDifferentTensors": len(common) - identical,
        "commonIdenticalTensors": identical, "keysOnlyNms": len(first.keys() - second.keys()),
        "keysOnlyNmsFree": len(second.keys() - first.keys()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = split_audit(args.dataset)
    report = {mode: specs(args.models, mode, audit["validationCategoryIds"]) for mode in MODEL_MODES}
    result = {"datasetAudit": audit, "models": report, "checkpointStateComparison": compare_states(args.models)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
