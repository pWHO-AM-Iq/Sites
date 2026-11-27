# モデルカード / Model card

| モデル / Variant | Checkpoint | 選定エポック / Selected epoch | 後処理 / Postprocessing |
| --- | --- | ---: | --- |
| {{PRODUCT}} NMS | `custom_detector_nms.pt` | {{NMS_EPOCH}} | NMS |
| {{PRODUCT}} NMS-free | `custom_detector_nms_free.pt` | {{NMS_FREE_EPOCH}} | two-stage Top-K |

## 日本語

一般画像でCOCO 80クラスを検出する開発用モデルです。別領域、遠距離、遮蔽、小物体、低照度での性能は導入前に対象データで確認してください。精度と速度の評価範囲は [BENCHMARK.md](BENCHMARK.md) に記載しています。ONNX版のアクセラレータ別適合性は別途検証が必要です。

{{TRAINING_JA}} モデル定義はUltralytics YOLO26を基にしており、チェックポイントは `{{ARCHITECTURE}}` と上流コミット `{{UPSTREAM_COMMIT}}` を記録しています。

学習実行 `{{TRAINING_RUN}}` では COCO train2017 の {{TRAIN_IMAGES}} 枚で学習し、同splitの残り {{SELECTION_IMAGES}} 枚でEMA重みを選定しました。最終評価は COCO val2017 の {{FINAL_IMAGES}} 枚です。

受領した翻案通知には AGPL-3.0 の記載があります。モデル、学習データ、翻案コードの配布条件と提供可否は確認中です。

## English

This development model detects 80 COCO categories in general images. Validate transfer to the target domain, including small or distant objects, occlusions and low-light inputs. [BENCHMARK.md](BENCHMARK.md) describes the scope of the measured results. Accelerator-specific ONNX compatibility requires separate verification.

{{TRAINING_EN}} The model definition derives from Ultralytics YOLO26. Both checkpoints record `{{ARCHITECTURE}}` and upstream commit `{{UPSTREAM_COMMIT}}`.

Run `{{TRAINING_RUN}}` used {{TRAIN_IMAGES}} COCO train2017 images for fitting and {{SELECTION_IMAGES}} from the same split to select EMA weights. Final evaluation used the {{FINAL_IMAGES}} COCO val2017 images.

The supplied adaptation notices identify AGPL-3.0. Distribution clearance for the model, training data and adapted code is pending.

## 来歴記録 / Provenance records

- 既存モデル README と SHA-256 / Original model usage and artifact hashes · `models/custom_detector/README.md`
- チェックポイントのメタデータと ONNX の監査結果 / Checkpoint metadata and ONNX audit · `product/evaluation/artifact-audit.json`
- 受領した起動・分割・学習記録の要約 / Summary of supplied launch, split and training records · `product/evaluation/origin-training.json`
