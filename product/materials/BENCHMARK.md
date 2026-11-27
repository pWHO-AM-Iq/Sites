# ベンチマーク報告 / Benchmark report

## COCO 2017 val2017

COCO 2017 val2017 · 5,000 images · 36,781 annotations · 80 classes.

| Model | Status | AP50–95 % | AP50 % | Active parameters |
| --- | --- | --- | --- | --- |
| Custom Detector NMS | measured | 48.52 | 65.94 | 20435260 |
| Custom Detector NMS-free | measured | 48.76 | 66.10 | 20435260 |

| Model | AP75 | AP-S | AP-M | AP-L | AR1 | AR10 | AR100 | AR-S | AR-M | AR-L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Custom Detector NMS | 52.43 | 30.80 | 53.64 | 64.99 | 36.24 | 59.29 | 63.99 | 45.13 | 69.41 | 78.66 |
| Custom Detector NMS-free | 52.95 | 31.15 | 53.35 | 64.63 | 37.24 | 61.91 | 68.08 | 50.89 | 72.77 | 82.30 |

公式COCO APIのbbox評価を使用します。AP50–95はIoU 0.50から0.95、AP50はIoU 0.50でのAPです。表の精度は百分率です。COCO APは検証セットと評価時の後処理に依存します。クラス別の結果、面積別指標とARは評価JSONやCSVを参照してください。

The official COCO bbox evaluator computes AP at IoU 0.50 through 0.95 and AP50 at IoU 0.50, expressed here as percentages. The validation split and evaluation-time postprocessing govern these results. The evaluation JSON and CSV files provide the available class-level, object-size and recall results.

## ローカル対照評価 / Local control evaluation

| Model | AP50–95 % | AP50 % | AP75 % | AR100 % |
| --- | --- | --- | --- | --- |
| YOLO26m E2E | 52.50 | 69.70 | 57.28 | 72.12 |

同じ COCO val2017 全5,000枚、640 × 640 RGBレターボックス、スコア下限0.001、最大300候補、COCOeval bbox を使用しました。YOLO26m E2E は RTX 3080 で ONNX Runtime CUDA を使用し、上位候補の選択はグラフ内で行いました。

The local control used the same 5,000 COCO val2017 images, 640 × 640 RGB letterbox, 0.001 score floor, 300-candidate limit and COCOeval bbox. YOLO26m E2E ran on an RTX 3080 with ONNX Runtime CUDA and graph-embedded Top-K. The separate four-image CPU trial timed graph execution and used a different latency scope from the end-to-end measurements below.

Repository evidence · `product/comparisons/local-control.json` and `product/comparisons/local-control.md`.


## 元学習記録の評価 / Archived training-run evaluation

| Model | Model selection AP50–95 % | Final archived AP50–95 % |
| --- | --- | --- |
| Custom Detector NMS | 50.34 | 49.17 |
| Custom Detector NMS-free | 49.81 | 48.71 |

受領した元学習記録のAPを百分率で示します。train2017から分離した2,000枚で重みを選定し、COCO val2017全5,000枚で最終評価しました。上段のONNX再評価とは別の評価実行です。 / These percentages come from the original training record. Model selection used 2,000 held-out train2017 images and final evaluation used all 5,000 val2017 images. The ONNX rerun above is a separate evaluation run.

## ローカル実行 / Local runtime

| Model | Backend | Status | Iterations | P50 ms | P95 ms | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| Custom Detector NMS | ort-cpu | completed | 100 | 437.45 | 611.75 | 2.19 |
| Custom Detector NMS-free | ort-cpu | completed | 100 | 365.82 | 465.59 | 2.68 |
| Custom Detector NMS | ort-cuda | completed | 100 | 33.16 | 49.44 | 29.96 |
| Custom Detector NMS-free | ort-cuda | completed | 100 | 38.46 | 51.59 | 26.37 |
| Custom Detector NMS | ort-trt | not_measured | — | — | — | — |
| Custom Detector NMS-free | ort-trt | not_measured | — | — | — | — |
| Custom Detector NMS | torch-cpu | completed | 100 | 358.57 | 439.83 | 2.69 |
| Custom Detector NMS-free | torch-cpu | completed | 100 | 432.64 | 889.78 | 2.01 |
| Custom Detector NMS | torch-fp16 | completed | 100 | 28.69 | 36.41 | 33.66 |
| Custom Detector NMS-free | torch-fp16 | completed | 100 | 28.37 | 32.11 | 34.80 |
| Custom Detector NMS | torch-fp32 | completed | 100 | 26.25 | 29.93 | 37.10 |
| Custom Detector NMS-free | torch-fp32 | completed | 100 | 26.96 | 29.13 | 36.52 |

バッチ1、640 × 640。E2Eは画像のデコード、前処理、転送込み推論、後処理を含みます。セッション初期化とウォームアップは別集計です。CPU、GPU、バックエンド、入力数、スレッド設定と各集計区間は `runtime.csv` および元のランタイムJSONで確認できます。

Measurements use batch 1 at 640 × 640. End-to-end timing includes image decoding, preprocessing, the forward pass with transfers, and postprocessing. Initialization and warmup are tracked independently. Runtime CSV and source JSON record the environment, thread count, sample count and timing stages.

ローカルのONNX実行は Windows、Core i9-11900、GeForce RTX 3080、ONNX Runtime 1.22.0。CUDAはFP32のグラフでTF32を許可しています。同じワークステーションで他の処理が動いていたため、速度は観測値として扱います。 / ONNX runs used Windows, a Core i9-11900, an RTX 3080 and ONNX Runtime 1.22.0. The CUDA provider allowed TF32 on the FP32 graph. Other work on the same machine may have affected timings.

PyTorch行は受領した元のモデル定義で測定しました。GPUメモリ欄のPyTorch値はCUDA allocatorのピーク値です。実行時の詳細、CPUとの照合、FP32・FP16の出力差は `product/evaluation/runtime/measurements.json` に記録しています。 / PyTorch timings use the original model definition from the handoff. PyTorch GPU memory reports CUDA allocator peaks. Backend details, CPU comparisons and FP32/FP16 output deltas are recorded in the sanitized measurements.

| Model | Backend | MP4 frames | Video | Empty input | Repeat mismatches |
| --- | --- | --- | --- | --- | --- |
| Custom Detector NMS | ort-cpu | 30 | completed | rejected_at_decode | 0 |
| Custom Detector NMS-free | ort-cpu | 30 | completed | rejected_at_decode | 0 |
| Custom Detector NMS | ort-cuda | 30 | completed | rejected_at_decode | 0 |
| Custom Detector NMS-free | ort-cuda | 30 | completed | rejected_at_decode | 0 |
| Custom Detector NMS | torch-cpu | 30 | completed | rejected_at_decode | 0 |
| Custom Detector NMS-free | torch-cpu | 30 | completed | rejected_at_decode | 0 |
| Custom Detector NMS | torch-fp16 | 30 | completed | rejected_at_decode | 0 |
| Custom Detector NMS-free | torch-fp16 | 30 | completed | rejected_at_decode | 0 |
| Custom Detector NMS | torch-fp32 | 30 | completed | rejected_at_decode | 0 |
| Custom Detector NMS-free | torch-fp32 | 30 | completed | rejected_at_decode | 0 |

元READMEにある単一画像・単回の参考値（ONNX約156 ms、PyTorch約172 ms）はアーカイブ情報です。 / The original README’s single-image, single-run figures (approximately 156 ms ONNX and 172 ms PyTorch) are archived context.

## 公式公表値 / Official reference values

測定ハードウェア、設定、モデルの定義は各公表元の資料を参照してください。 / See each source for its hardware, settings and model definitions.

- YOLO26n · 40.9% AP50–95. [Official source](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)
- YOLO26s · 48.6% AP50–95. [Official source](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)
- YOLO26m · 53.1% AP50–95. [Official source](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)
- YOLO26l · 55% AP50–95. [Official source](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)
- YOLO26x · 57.5% AP50–95. [Official source](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)
- YOLO11n · 39.5% AP50–95. [Official source](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)
- YOLO11s · 47% AP50–95. [Official source](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)
- YOLO11m · 51.5% AP50–95. [Official source](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)
- YOLO11l · 53.4% AP50–95. [Official source](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)
- YOLO11x · 54.7% AP50–95. [Official source](https://docs.ultralytics.com/compare/yolo26-vs-yolo11)
- RT-DETR-R18 · 46.5% AP50–95. [Official source](https://github.com/lyuwenyu/RT-DETR)
- RT-DETR-R34 · 48.9% AP50–95. [Official source](https://github.com/lyuwenyu/RT-DETR)
- RT-DETR-R50 · 53.1% AP50–95. [Official source](https://github.com/lyuwenyu/RT-DETR)
- RT-DETR-R101 · 54.3% AP50–95. [Official source](https://github.com/lyuwenyu/RT-DETR)
- D-FINE-N · 42.8% AP50–95. [Official source](https://github.com/Peterande/D-FINE)
- D-FINE-S · 48.5% AP50–95. [Official source](https://github.com/Peterande/D-FINE)
- D-FINE-M · 52.3% AP50–95. [Official source](https://github.com/Peterande/D-FINE)
- D-FINE-L · 54% AP50–95. [Official source](https://github.com/Peterande/D-FINE)
- D-FINE-X · 55.8% AP50–95. [Official source](https://github.com/Peterande/D-FINE)
- RF-DETR-N · 48.4% AP50–95. [Official source](https://github.com/roboflow/rf-detr)
- RF-DETR-S · 53% AP50–95. [Official source](https://github.com/roboflow/rf-detr)
- RF-DETR-M · 54.7% AP50–95. [Official source](https://github.com/roboflow/rf-detr)
- RF-DETR-L · 56.5% AP50–95. [Official source](https://github.com/roboflow/rf-detr)
- RF-DETR-XL · 58.6% AP50–95. [Official source](https://github.com/roboflow/rf-detr)
- RF-DETR-2XL · 60.1% AP50–95. [Official source](https://github.com/roboflow/rf-detr)

## 評価記録 / Evaluation record

- 版 / Release · `draft.1`
- 結果記録日時 / Results recorded at · `2026-11-27T19:19:47+09:00`
- 評価コードのコミット / Evaluation source commit · `fe236d2bb887abf30b486cfc47eccd1fcb4a223e`
- 評価の出所 / Evaluation record · `product/evaluation/run-record.json`
- 重みと ONNX の SHA-256 / Artifact hashes · [Original model README](../../models/custom_detector/README.md#ファイルの照合), [Artifact audit](../evaluation/artifact-audit.json), and the packaged `SHA256SUMS.txt`.
