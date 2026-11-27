# 入出力仕様

| 項目 | 値 |
| --- | --- |
| 種別 | 物体検出, COCO 80 |
| ONNX | FP32, opset 17 |
| ONNX 入力 | `images`, float32 RGB NCHW, `[1,3,640,640]` |
| ONNX 出力・矩形 | `boxes`, `[1,8400,4]`, xyxy （640 × 640 入力上のピクセル） |
| ONNX 出力・スコア | `scores`, `[1,8400,80]`, sigmoid 適用済みクラス確率 |
| PyTorch | モデル構成メタデータ付きテンソルチェックポイント。元のモデル定義が必要 |
| モデル構成 ID | `yolo26m_ae725d5` |
| 有効な推論パラメータ数 | 20,435,260 （方式ごと。チェックポイント記載） |
| 学習時パラメータ数 | 21,896,248 （共有部と両ヘッドの合計。チェックポイント記載） |
| 学習可能パラメータ数 | 21,896,248 |
| ONNX 初期化テンソルの要素数 | 20,411,132 （方式ごと。ONNX グラフで集計） |
| ONNX 演算量（概算） | 34.213 GMACs · 68.426 GFLOPs （積和演算を2 FLOPとして計算。固定入力で Conv と MatMul を集計） |
| ファイルサイズ | NMS ONNX 81,908,059 B, NMS-free ONNX 81,910,251 B, NMS PT 82,131,375 B, NMS-free PT 82,132,015 B |

RGB画像をアスペクト比を保って640辺内にリサイズします。RGB 114のキャンバスに中央配置し、画素値を255で割ります。NMS版の人物例は最大スコアのクラスが人物で、しきい値を超える候補にIoU 0.7のNMSを適用し最大300件を返します。NMSフリー版は候補の最大確率で上位300件を取り、その候補とクラスの組み合わせから上位300件を取った後、人物としきい値で絞ります。モデルはどちらも生の候補を出力するため、後処理は呼び出し側が行います。余白を引いてリサイズ倍率を戻し、元画像の範囲に丸めます。

## 評価記録 / Evaluation record

- 版 / Release · `draft.1`
- 結果記録日時 / Results recorded at · `2026-11-27T19:19:47+09:00`
- 評価コードのコミット / Evaluation source commit · `fe236d2bb887abf30b486cfc47eccd1fcb4a223e`
- 評価の出所 / Evaluation record · `product/evaluation/run-record.json`
- 重みと ONNX の SHA-256 / Artifact hashes · [Original model README](../../models/custom_detector/README.md#ファイルの照合), [Artifact audit](../evaluation/artifact-audit.json), and the packaged `SHA256SUMS.txt`.
