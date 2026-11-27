# ONNX Runtimeの物体検出で、実行前に入出力を点検する

物体検出モデルを別のアプリへつなぐときは、まず入力名、形状、型、出力がモデルに収録された段階を確認します。同じ物体検出器でも、グラフが候補の全行を返すものと、上位だけを選んだ結果を返すものがあります。

## 出力の名前・形状・型を表示する

Python環境に `onnxruntime` を導入して、確認したいモデルファイルを引数に渡します。

```python
import sys

import onnxruntime as ort

session = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
for label, definitions in (("input", session.get_inputs()), ("output", session.get_outputs())):
    for definition in definitions:
        print(label, definition.name, definition.shape, definition.type)
```

Custom DetectorのONNX版では、ドキュメントに記載された形状は以下の通りです。

| 種類 | 名前 | 形状 |
| --- | --- | --- |
| 入力 | images | [1, 3, 640, 640] |
| 出力 | boxes | [1, 8400, 4] |
| 出力 | scores | [1, 8400, 80] |

`boxes` は `[x1, y1, x2, y2]` で、640 × 640 入力上のピクセルです。 `scores` に sigmoid は適用済みです。JSONに出す座標が元画像のピクセルか、入力画像のピクセルかを下流のコードで確認します。

## 上位候補の位置に注意する

同じ系列の公式YOLO26mを Ultralytics 8.4.0 でE2E ONNXに変換したローカル実験では、出力が `[1, 300, 6]` になりました。このグラフでは上位300件がすでに選ばれています。モデルを差し替えるときは、セッション生成時に期待する出力名と形状を照合しておくと、後処理を取り違えた位置でエラーを出せます。

参考資料　[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html) · [Ultralytics YOLO26 検出モデル](https://docs.ultralytics.com/models/yolo26)
