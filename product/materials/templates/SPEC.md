# 入出力仕様 / I/O specification

| 項目 / Field | 値 / Value |
| --- | --- |
| 種別 / task | 物体検出 / object detection, COCO 80 |
| ONNX | FP32, opset {{OPSET}} |
| ONNX input | `images`, float32 RGB NCHW, `[1,3,640,640]` |
| ONNX boxes | `boxes`, `[1,8400,4]`, xyxy in 640 × 640 input pixels |
| ONNX scores | `scores`, `[1,8400,80]`, sigmoid probabilities |
| PyTorch | tensor checkpoints with architecture metadata, original model definition required |
| Architecture ID | `{{ARCHITECTURE}}` |
| Active inference parameters | {{ACTIVE_PARAMS}} per variant, checkpoint metadata |
| Training parameters | {{TRAINING_PARAMS}} across the shared network and both heads, checkpoint metadata |
| Trainable parameters | {{TRAINABLE_PARAMS}} |
| ONNX initializer elements | {{INITIALIZERS}} each, counted from ONNX graph |
| Approximate ONNX compute | {{GMACS}} GMACs · {{GFLOPS}} GFLOPs (one multiply-add as two FLOPs), counting Conv and MatMul at fixed input size |
| File size | NMS ONNX {{NMS_ONNX_BYTES}} B, NMS-free ONNX {{NMS_FREE_ONNX_BYTES}} B, NMS PT {{NMS_PT_BYTES}} B, NMS-free PT {{NMS_FREE_PT_BYTES}} B |

<!-- ja -->
RGB画像をアスペクト比を保って640辺内にリサイズします。RGB 114のキャンバスに中央配置し、画素値を255で割ります。NMS版の人物例は最大スコアのクラスが人物で、しきい値を超える候補にIoU 0.7のNMSを適用し最大300件を返します。NMSフリー版は候補の最大確率で上位300件を取り、その候補とクラスの組み合わせから上位300件を取った後、人物としきい値で絞ります。モデルはどちらも生の候補を出力するため、後処理は呼び出し側が行います。余白を引いてリサイズ倍率を戻し、元画像の範囲に丸めます。

<!-- en -->
Resize the RGB image to fit inside 640 × 640 with aspect ratio preserved, center it on RGB 114 padding and divide pixels by 255. For NMS, the person demo selects candidates whose highest scoring class is person, filters by threshold, and applies IoU 0.7 suppression with a limit of 300. In NMS-free mode, it first takes the 300 candidates with highest maximum class probability, then the top 300 candidate–class pairs before filtering for people and threshold. Both ONNX variants emit raw candidates for external postprocessing. Reverse padding and resize and clip boxes to the original image extent. The demo is for person inference. Use the COCO evaluator for all classes.
