---
language:
  - en
  - ja
pipeline_tag: object-detection
tags:
  - onnx
  - pytorch
---

# Custom Detector

Custom Detector is a 640 × 640 object detector with 80 COCO 2017 categories. The NMS and NMS-free variants both have 20,435,260 active parameters. The ONNX exports return `boxes` with shape `[1, 8400, 4]` and `scores` with shape `[1, 8400, 80]`. The postprocessing mode determines how these outputs are filtered. A separately maintained Python example isolates person detections and projects bounding boxes back to original-image pixel coordinates.

## Training and provenance

The archived training configuration, source, and startup receipt support random initialization with seed `20261103`, with no external pretrained weights or resumed training checkpoint. That record lacks a standalone hash of the initial weights. The documented model definition adapts Ultralytics YOLO26 at commit `ae725d5cb20c2d849b864cd34fc15f5c55f877a5`. The archived training and inference notices identify the adapted parts as AGPL-3.0. The license text matches the recorded upstream copy. Exact downstream licensing scope and distribution terms remain under review.

The COCO train2017 split used 116,287 images for training and 2,000 for checkpoint selection. Final evaluation used the separate 5,000-image val2017 split. The handoff records EMA checkpoint selection at epochs 195 for NMS and 205 for NMS-free. [COCO](https://cocodataset.org/) · [Ultralytics YOLO26 paper](https://arxiv.org/abs/2606.03748)

## Evaluation and limitations

The local COCO val2017 bounding-box evaluation covered all 5,000 images and 80 classes with the official COCO evaluator. The NMS model recorded 48.52 AP50–95 and 65.94 AP50. The NMS-free model recorded 48.76 AP50–95 and 66.10 AP50. Scores are percentages. Evaluation used batch 1, 640 × 640 letterboxed input, a confidence floor of 0.001, and up to 300 candidates per image.

See the [evaluation report](../../materials/BENCHMARK.md) and [interface specification](../../materials/SPEC.md) for input/output, methodology, and measured results. General-image COCO performance may vary on domain-specific images, particularly small or partially occluded objects or low-light scenes. Model outputs should be checked against the target deployment data.

## Documentation and artifacts

This card is prepared for a documentation release. Model artifacts are pending a distribution decision. Before publishing this card, copy the benchmark and specification into the model repository and update the local links. Add the exact license tag, file manifest, notices, and instructions only when the matching artifacts and permissions are confirmed.

## 日本語

COCO 2017 の80クラスに対応する640 × 640固定入力の物体検出モデルの技術資料です。引き継ぎ資料にランダム初期化からの学習記録があり、モデル定義は Ultralytics YOLO26 を翻案しています。モデル・実装の配布条件は確認中です。対象データで検出漏れ・誤検出と処理時間を確認してください。
