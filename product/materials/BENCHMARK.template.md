# ベンチマーク報告 / Benchmark report

## COCO 2017 val2017

{{COVERAGE}}

{{ACCURACY}}

{{ACCURACY_DETAILS}}

公式COCO APIのbbox評価を使用します。AP50–95はIoU 0.50から0.95、AP50はIoU 0.50でのAPです。表の精度は百分率です。COCO APは検証セットと評価時の後処理に依存します。クラス別の結果、面積別指標とARは評価JSONやCSVを参照してください。

The official COCO bbox evaluator computes AP at IoU 0.50 through 0.95 and AP50 at IoU 0.50, expressed here as percentages. The validation split and evaluation-time postprocessing govern these results. The evaluation JSON and CSV files provide the available class-level, object-size and recall results.

## ローカル対照評価 / Local control evaluation

{{LOCAL_CONTROL}}

## 元学習記録の評価 / Archived training-run evaluation

{{ARCHIVED}}

受領した元学習記録のAPを百分率で示します。train2017から分離した2,000枚で重みを選定し、COCO val2017全5,000枚で最終評価しました。上段のONNX再評価とは別の評価実行です。 / These percentages come from the original training record. Model selection used 2,000 held-out train2017 images and final evaluation used all 5,000 val2017 images. The ONNX rerun above is a separate evaluation run.

## ローカル実行 / Local runtime

{{RUNTIME}}

バッチ1、640 × 640。E2Eは画像のデコード、前処理、転送込み推論、後処理を含みます。セッション初期化とウォームアップは別集計です。CPU、GPU、バックエンド、入力数、スレッド設定と各集計区間は `runtime.csv` および元のランタイムJSONで確認できます。

Measurements use batch 1 at 640 × 640. End-to-end timing includes image decoding, preprocessing, the forward pass with transfers, and postprocessing. Initialization and warmup are tracked independently. Runtime CSV and source JSON record the environment, thread count, sample count and timing stages.

ローカルのONNX実行は Windows、Core i9-11900、GeForce RTX 3080、ONNX Runtime 1.22.0。CUDAはFP32のグラフでTF32を許可しています。同じワークステーションで他の処理が動いていたため、速度は観測値として扱います。 / ONNX runs used Windows, a Core i9-11900, an RTX 3080 and ONNX Runtime 1.22.0. The CUDA provider allowed TF32 on the FP32 graph. Other work on the same machine may have affected timings.

PyTorch行は受領した元のモデル定義で測定しました。GPUメモリ欄のPyTorch値はCUDA allocatorのピーク値です。実行時の詳細、CPUとの照合、FP32・FP16の出力差は `product/evaluation/runtime/measurements.json` に記録しています。 / PyTorch timings use the original model definition from the handoff. PyTorch GPU memory reports CUDA allocator peaks. Backend details, CPU comparisons and FP32/FP16 output deltas are recorded in the sanitized measurements.

{{COMPATIBILITY}}

元READMEにある単一画像・単回の参考値（ONNX約156 ms、PyTorch約172 ms）はアーカイブ情報です。 / The original README’s single-image, single-run figures (approximately 156 ms ONNX and 172 ms PyTorch) are archived context.

## 公式公表値 / Official reference values

測定ハードウェア、設定、モデルの定義は各公表元の資料を参照してください。 / See each source for its hardware, settings and model definitions.

{{REFERENCES}}
