# {{PRODUCT}} / 製品ドラフト

> ローカル検証用。公開・提供の承認待ちです。 / Local evaluation draft. Release approval is pending.

## 日本語

COCO 80クラスの検出器です。NMS版とNMSフリー版のONNXは、いずれも固定640 × 640入力です。人物検出のサンプルは入力画像上の矩形とスコアをJSONで返します。

同梱ZIPを展開したディレクトリで、以下を実行します。検証環境は Python 3.12、Windows、ONNX Runtime CPU です。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.\.venv\Scripts\python.exe detect_onnx.py input.jpg --mode nms_free
.\.venv\Scripts\python.exe detect_onnx.py input.jpg --mode nms
```

`input.jpg` は手元の画像に置き換えてください。パッケージに含まれる方式を指定します。出力の `people_xyxy_score` は表示向きに補正した元画像のピクセル座標と信頼度です。スコアの既定しきい値は0.35です。

仕様は [SPEC.md](SPEC.md)、来歴と制約は [MODEL_CARD.md](MODEL_CARD.md)、付属物は各ZIPの `PACKAGE.md` を参照してください。 `SHA256SUMS.txt` は展開ファイルの照合用です。PyTorchのチェックポイントを含む場合、元のモデル定義が必要です。

## English

{{PRODUCT}} covers the 80 COCO categories. Both ONNX variants accept a fixed 640 × 640 input. The bundled person-detection example runs with ONNX Runtime on CPU and emits a JSON list of boxes and scores. Install the pinned dependencies in a Python 3.12 virtual environment, then run the commands above with your own image and a variant included in the package.

`people_xyxy_score` contains pixel coordinates on the orientation-corrected input image and a confidence score. The default threshold is 0.35. See [SPEC.md](SPEC.md) for the interface, [MODEL_CARD.md](MODEL_CARD.md) for provenance and limitations, and each archive’s `PACKAGE.md` for its contents. Compare local hashes against `SHA256SUMS.txt`. PyTorch checkpoints require the original model definition.
