# 入出力仕様 / I/O specification

| 項目 / Field | 値 / Value |
| --- | --- |
| 種別 / task | 物体検出 / object detection, COCO 80 |
| ONNX | FP32, opset 17 |
| ONNX input | `images`, float32 RGB NCHW, `[1,3,640,640]` |
| ONNX boxes | `boxes`, `[1,8400,4]`, xyxy in 640 × 640 input pixels |
| ONNX scores | `scores`, `[1,8400,80]`, sigmoid probabilities |
| PyTorch | tensor checkpoints with architecture metadata, original model definition required |
| Architecture ID | `yolo26m_ae725d5` |
| Active inference parameters | 20,435,260 per variant, checkpoint metadata |
| Training parameters | 21,896,248 across the shared network and both heads, checkpoint metadata |
| Trainable parameters | 21,896,248 |
| ONNX initializer elements | 20,411,132 each, counted from ONNX graph |
| Approximate ONNX compute | 34.213 GMACs · 68.426 GFLOPs (one multiply-add as two FLOPs), counting Conv and MatMul at fixed input size |
| File size | NMS ONNX 81,908,059 B, NMS-free ONNX 81,910,251 B, NMS PT 82,131,375 B, NMS-free PT 82,132,015 B |


RGB画像をアスペクト比を保って640辺内にリサイズします。RGB 114のキャンバスに中央配置し、画素値を255で割ります。NMS版の人物例は最大スコアのクラスが人物で、しきい値を超える候補にIoU 0.7のNMSを適用し最大300件を返します。NMSフリー版は候補の最大確率で上位300件を取り、その候補とクラスの組み合わせから上位300件を取った後、人物としきい値で絞ります。モデルはどちらも生の候補を出力するため、後処理は呼び出し側が行います。余白を引いてリサイズ倍率を戻し、元画像の範囲に丸めます。


Resize the RGB image to fit inside 640 × 640 with aspect ratio preserved, center it on RGB 114 padding and divide pixels by 255. For NMS, the person demo selects candidates whose highest scoring class is person, filters by threshold, and applies IoU 0.7 suppression with a limit of 300. In NMS-free mode, it first takes the 300 candidates with highest maximum class probability, then the top 300 candidate–class pairs before filtering for people and threshold. Both ONNX variants emit raw candidates for external postprocessing. Reverse padding and resize and clip boxes to the original image extent. The demo is for person inference. Use the COCO evaluator for all classes.

## 評価記録 / Evaluation record

- 版 / Release · `draft.1`
- 結果記録日時 / Results recorded at · `2026-11-27T19:19:47+09:00`
- 評価コードのコミット / Evaluation source commit · `fe236d2bb887abf30b486cfc47eccd1fcb4a223e`
- 評価の出所 / Evaluation record · `product/evaluation/run-record.json`
- 重みと ONNX の SHA-256 / Artifact hashes · [Original model README](../../models/custom_detector/README.md#ファイルの照合), [Artifact audit](../evaluation/artifact-audit.json), and the packaged `SHA256SUMS.txt`.
