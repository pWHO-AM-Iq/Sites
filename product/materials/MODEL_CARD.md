# モデルカード / Model card

| モデル / Variant | Checkpoint | 選定エポック / Selected epoch | 後処理 / Postprocessing |
| --- | --- | ---: | --- |
| Custom Detector NMS | `custom_detector_nms.pt` | 195 | NMS |
| Custom Detector NMS-free | `custom_detector_nms_free.pt` | 205 | two-stage Top-K |

## 日本語

一般画像でCOCO 80クラスを検出する開発用モデルです。別領域、遠距離、遮蔽、小物体、低照度での性能は導入前に対象データで確認してください。精度と速度の評価範囲は [BENCHMARK.md](BENCHMARK.md) に記載しています。ONNX版のアクセラレータ別適合性は別途検証が必要です。

提供された学習記録では、外部の学習済み重みや学習再開用の重みを使わず、seed 20261103 のランダム初期化から300エポック学習しています。初期重みの独立した保存ファイルと自作学習プロジェクトのGitコミットは不明です。 モデル定義はUltralytics YOLO26を基にしており、チェックポイントは `yolo26m_ae725d5` と上流コミット `ae725d5cb20c2d849b864cd34fc15f5c55f877a5` を記録しています。

学習実行 `yolo26m_coco640_20261104_134911` では COCO train2017 の 116,287 枚で学習し、同splitの残り 2,000 枚でEMA重みを選定しました。最終評価は COCO val2017 の 5,000 枚です。

受領した翻案通知には AGPL-3.0 の記載があります。モデル、学習データ、翻案コードの配布条件と提供可否は確認中です。

## English

This development model detects 80 COCO categories in general images. Validate transfer to the target domain, including small or distant objects, occlusions and low-light inputs. [BENCHMARK.md](BENCHMARK.md) describes the scope of the measured results. Accelerator-specific ONNX compatibility requires separate verification.

The supplied training records document a 300-epoch run from random initialization (seed 20261103), with no external pretrained weights or resume checkpoint. A standalone initial-weight artifact and the original training repository commit are unknown. The model definition derives from Ultralytics YOLO26. Both checkpoints record `yolo26m_ae725d5` and upstream commit `ae725d5cb20c2d849b864cd34fc15f5c55f877a5`.

Run `yolo26m_coco640_20261104_134911` used 116,287 COCO train2017 images for fitting and 2,000 from the same split to select EMA weights. Final evaluation used the 5,000 COCO val2017 images.

The supplied adaptation notices identify AGPL-3.0. Distribution clearance for the model, training data and adapted code is pending.

## 来歴記録 / Provenance records

- 既存モデル README と SHA-256 / Original model usage and artifact hashes · `models/custom_detector/README.md`
- チェックポイントのメタデータと ONNX の監査結果 / Checkpoint metadata and ONNX audit · `product/evaluation/artifact-audit.json`
- 受領した起動・分割・学習記録の要約 / Summary of supplied launch, split and training records · `product/evaluation/origin-training.json`

## 評価記録 / Evaluation record

- 版 / Release · `draft.1`
- 結果記録日時 / Results recorded at · `2026-11-27T19:19:47+09:00`
- 評価コードのコミット / Evaluation source commit · `fe236d2bb887abf30b486cfc47eccd1fcb4a223e`
- 評価の出所 / Evaluation record · `product/evaluation/run-record.json`
- 重みと ONNX の SHA-256 / Artifact hashes · [Original model README](../../models/custom_detector/README.md#ファイルの照合), [Artifact audit](../evaluation/artifact-audit.json), and the packaged `SHA256SUMS.txt`.
