# Custom Detector の COCO 2017 評価

公開対象の ONNX 2点を、ローカルの COCO 2017 val2017 全5,000枚と公式 Python COCO API の bbox 評価で測定した。結果は [quality.json](quality.json)、分割とモデルの照合は [artifact-audit.json](artifact-audit.json) を参照。

## 測定結果

| COCO bbox 指標 | NMS | NMS フリー |
| --- | ---: | ---: |
| AP 0.50–0.95 | 48.52% | 48.76% |
| AP50 | 65.94% | 66.10% |
| AP75 | 52.43% | 52.95% |
| AP 小 | 30.80% | 31.15% |
| AP 中 | 53.64% | 53.35% |
| AP 大 | 64.99% | 64.63% |
| AR@1 | 36.24% | 37.24% |
| AR@10 | 59.29% | 61.91% |
| AR@100 | 63.99% | 68.08% |
| AR 小 | 45.13% | 50.89% |
| AR 中 | 69.41% | 72.77% |
| AR 大 | 78.66% | 82.30% |

全80カテゴリーの micro 指標（IoU 0.50）

| スコアしきい値 | NMS P / R / F1 | NMS フリー P / R / F1 |
| ---: | ---: | ---: |
| 0.10 | 55.25% / 72.80% / 62.82% | 57.19% / 71.16% / 63.41% |
| 0.25 | 74.47% / 63.46% / 68.53% | 75.73% / 61.66% / 67.98% |
| 0.35 | 81.36% / 58.22% / 67.87% | 82.01% / 56.66% / 67.02% |
| 0.50 | 88.32% / 50.05% / 63.89% | 88.50% / 49.63% / 63.60% |
| 0.75 | 95.53% / 31.56% / 47.44% | 95.14% / 35.04% / 51.21% |

NMS フリー版の差は AP が +0.24ポイント、AR@100 が +4.09ポイント。スコア > 0.35 では NMS 版の F1 が +0.85ポイント。両モデルとも、小物体 AP は大物体 AP を33ポイント以上下回った。

カテゴリー別 AP は NMS 版では train 74.8%、NMS フリー版では bear 75.8% が最高。両方式の最低は hair drier（9.5% / 7.1%、非 crowd の正解11件）。book は正解1,129件でも AP が18.0% / 17.4%。用途で対象クラスが決まっている場合は、全体平均と併せて該当クラスの AP・PR を確認できる。

評価対象は画像5,000枚、80カテゴリー、注釈36,781件（うち crowd 446件）。COCO API に入力した検出は NMS 445,733件、NMS フリー571,129件。

## 評価条件

- 画像は元の JPEG を RGB として読み込み、Pillow のバイリニア補間でアスペクト比を保って 640 × 640 以内に拡大縮小。余白を各チャンネル 114 で中央に配置し、0–1 の NCHW FP32 に変換した。
- 位置は余白を除き、丸め後の横・縦の実リサイズ倍率で元画像に復元した。画像境界で切り、正の面積を持つ COCO xywh のみを出力した。ONNX スコアは sigmoid 適用済みであることを出力範囲でも確認した。PyTorch 重みのラベル順序は、公式注釈の昇順カテゴリーIDと一致した。
- NMS 版は各候補の最高得点クラスを選び、スコア > 0.001、クラス別 NMS の IoU 0.70、画像当たり最大300件とした。NMS フリー版は最大クラススコアで上位300候補を選び、候補×80クラスの得点から再度上位300件を採り、スコア > 0.001 を適用した。
- COCO AP は IoU 0.50–0.95 の10点、リコール101点、80カテゴリーの公式平均。面積別 AP/AR と maxDets 1/10/100 を公式設定で算出した。COCO API の maxDets は画像・カテゴリー単位。前処理後の300件制限はモデルの画像単位出力条件。
- 固定しきい値の Precision / Recall / F1 は、IoU 0.50、全サイズ、maxDets 100、COCO API のスコア順マッチングを使用。crowd 等で無視された検出を除き、80カテゴリーの TP/FP/FN を合算した micro 指標。カテゴリー別の PR カーブは公式 AP50 の補間済み精度をリコール0–1の101点で記録した。固定しきい値で検出0件のカテゴリーの Precision は JSON の null。
- 推論環境は Windows 11 Pro 10.0.26200、Core i9-11900、ONNX Runtime GPU 1.22.0 CUDA EP、NVIDIA ドライバー 610.88、PyTorch 配布の CUDA 12.6 / cuDNN 9.10.2 DLL。FP32、TF32 無効。NMS 版は RTX 2080、NMS フリー版は RTX 3080。速度は個別のランタイム測定を参照。
- 画像ID昇順の先頭100枚を CPU でも実行し、スコア > 0.35 の検出を CUDA と照合した。画像ごとの検出件数とカテゴリーIDの並びは両方式で一致。最大絶対差は NMS 版でスコア 2.15×10⁻⁶、xywh 0.000175 px（562検出）、NMS フリー版でスコア 1.96×10⁻⁵、xywh 0.000401 px（528検出）。

## データと来歴

- ローカルの train2017 は118,287枚、val2017 は5,000枚。IDの共通集合は空であり、JPEG の ZIP サイズと CRC で候補を抽出して内容を照合した結果、バイト同一の組は0件。val2017 の画像名とIDは公式注釈に一致し、注釈の画像・カテゴリー参照はすべて有効だった。実際の学習データ、画像変換後の近重複、分割の運用は、元の学習記録が必要。
- 開発者申告では COCO 2017 を用いて初期重みから学習した。配布 README は Ultralytics YOLO26 を基にしたモデル定義と記載し、チェックポイントはアーキテクチャID `yolo26m_ae725d5`、上流コミット `ae725d5cb20c2d849b864cd34fc15f5c55f877a5`、`Reconstructed YOLO26m branch export` を含む。学習コード、初期化設定、ログ、エポック選択基準は未照合。設計の来歴や独自性は元プロジェクトの確認後に記述を確定する。
- 配布 README の採用エポックは NMS 195、NMS フリー 205。双方の PT チェックポイントは推論パラメータ20,435,260要素を保持し、ONNX は初期化子20,411,132要素を保持。学習パラメータ21,896,248は PT メタデータの値。ONNX Conv / MatMul のみを数えると 34.21 GMAC、積和を2 FLOPsとして 68.43 GFLOPs。固定 640 × 640 での演算量であり、活性化、加算、コピー、後処理等は含めていない。
- 2方式に共通する PT テンソルキー528個のうち440個の値が異なる。得られた差は配布モデル全体の比較であり、後処理方式単独の効果推定には同一重みでの対照実験が必要。
- AP / AR はこの val2017 の再測定値。チェックポイント選択に val2017 が使われたかは不明。独立テストセットでの性能とエポック選択履歴は元の学習プロジェクトが必要。

## 再現

以下はリポジトリ直下の Windows PowerShell。環境には NumPy、Pillow、OpenCV、ONNX Runtime、pycocotools、ONNX、PyTorch が必要。実測は NumPy 1.26.4、Pillow 10.2.0、OpenCV 4.12.0、ONNX Runtime GPU 1.22.0、pycocotools 2.0.11、ONNX 1.18.0、PyTorch 2.8.0+cu126。CPU 実行では `--cuda` を省略し、CPU 用 ONNX Runtime を利用できる。

```powershell
$data = 'E:/dataset/coco'
$weights = 'models/custom_detector'
$quality = 'product/evaluation'
$nms = Join-Path $env:TEMP 'detector-nms.jsonl'
$free = Join-Path $env:TEMP 'detector-nms-free.jsonl'
python "$quality/audit_artifacts.py" --dataset $data --models $weights --output "$quality/artifact-audit.json"
python "$quality/predict_coco.py" nms --dataset $data --model "$weights/custom_detector_nms.onnx" --predictions $nms --threads 2 --cuda 1
python "$quality/predict_coco.py" nms_free --dataset $data --model "$weights/custom_detector_nms_free.onnx" --predictions $free --threads 2 --cuda 0
python "$quality/score_coco.py" --dataset $data --nms $nms --nms-free $free --specs "$quality/artifact-audit.json" --output "$quality/quality.json"
```

参照元は [COCO Detection Evaluation](https://cocodataset.org/#detection-eval) と [公式 COCO API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)。

このワークスペース内の再集計では、ローカル専用の `private/product-evaluation/coco-val2017` に保存した `detections-nms.jsonl.gz` と `detections-nms_free.jsonl.gz` も scorer に直接渡せる。
