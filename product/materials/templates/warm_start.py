"""Initialize the dual detector's shared features from a compatible branch export."""

import argparse
import json
from pathlib import Path

import torch

from detector.model import ARCHITECTURE_ID, UPSTREAM_COMMIT, DualDetectorM


def load_shared_features(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or (
        checkpoint.get("architecture_id"), checkpoint.get("upstream_commit")
    ) != (ARCHITECTURE_ID, UPSTREAM_COMMIT):
        raise ValueError("Checkpoint architecture and upstream revision must match this trainer")
    branch = checkpoint.get("mode")
    state = checkpoint.get("model")
    if branch not in ("nms", "nms_free") or not isinstance(state, dict):
        raise ValueError("Expected a Custom Detector single-branch export")
    prefix = "model."
    source = {key.removeprefix(prefix): value for key, value in state.items()
              if isinstance(key, str) and key.startswith(prefix)}
    expected = model.model.state_dict()
    if source.keys() != expected.keys():
        raise ValueError("Checkpoint does not cover exactly the shared feature tensors")
    for key, destination in expected.items():
        value = source[key]
        if not isinstance(value, torch.Tensor) or value.shape != destination.shape or value.dtype != destination.dtype:
            raise ValueError(f"Incompatible feature tensor: {key}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite feature tensor: {key}")
    model.model.load_state_dict(source, strict=True)
    return {
        "source_branch": branch,
        "target_class_count": model.num_classes,
        "transferred_feature_tensors": len(source),
        "heads": "fresh initialization for both branches",
    }


def main():
    parser = argparse.ArgumentParser(description="Run a read-only CPU warm-start compatibility check.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-classes", type=int, default=80)
    args = parser.parse_args()
    if args.num_classes < 1:
        parser.error("--num-classes must be positive")
    model = DualDetectorM(args.num_classes)
    report = load_shared_features(model, args.checkpoint)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
