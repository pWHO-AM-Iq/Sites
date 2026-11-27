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

開発者の引き継ぎ資料には、2026年11月4日〜6日の300エポック実行 `yolo26m_coco640_20261104_134911` が保存されている。COCO train2017 の116,287枚を学習用、同じ元分割のうち別の2,000枚を選定用に使い、COCO val2017 の全5,000枚を最終評価に使った。全60回の選定用評価を読み、各方式で採用された EMA 重みのエポックが AP 最大の回と一致することを確認した。元評価の記録と今回の配布 ONNX 再測定値は [origin-training.json](origin-training.json) に収録した。

| 方式 | 採用エポック | 元の選定用 AP | 元の最終 AP | 今回の配布 ONNX AP |
| --- | ---: | ---: | ---: | ---: |
| NMS | 195 | 50.34% | 49.17% | 48.52% |
| NMS フリー | 205 | 49.81% | 48.71% | 48.76% |

- ローカルの COCO アーカイブ3点と train/val 注釈2点の SHA-256 は元の記録と一致した。注釈から3分割を再計算し、保存された学習・選定・最終評価のIDリストと順序まで一致し、分割の組み合わせすべてで重複は0件だった。元の train2017 と val2017 の JPEG では、ZIP サイズと CRC で候補を抽出して内容を照合した結果、バイト同一の組も0件だった。
- 学習時の設定3点と起動時に記録されたソース17点の SHA-256 を照合した。モデル生成、学習設定、起動・初回更新記録は seed 20261103 のランダム初期化、外部事前学習重みと再開元なし、物理バッチ32、目標実効バッチ64、BF16、MuSGD を示す。最初の更新は epoch 1 で、初回の勾配累積は1、設定の目標累積は2。配布チェックポイントのアーキテクチャIDは `yolo26m_ae725d5`。保存モデル定義は上流の YOLO26m トポロジーを保ち、プロジェクト側の二分岐 API、学習・評価アダプターを持つ。上流コミットは `ae725d5cb20c2d849b864cd34fc15f5c55f877a5`。初期重み単体のハッシュと元プロジェクト自身の Git コミットは不明。
- 保存モデル定義を CPU で初期化し、学習可能パラメータ21,896,248、凍結パラメータ0、共有部18,974,272、各ヘッド1,460,988を確認した。学習中の NMS フリーヘッドは共有特徴を勾配から切り離す。評価用 EMA コピーの重みは凍結されている。配布 PT の各方式は推論パラメータ20,435,260要素、ONNX は初期化子20,411,132要素。ONNX Conv / MatMul のみを数えると 34.21 GMAC、積和を2 FLOPsとして 68.43 GFLOPs。固定 640 × 640 での演算量であり、活性化、加算、コピー、後処理等は含めていない。
- 配布 PT/ONNX 4点の SHA-256 と両 ONNX 計算グラフの SHA-256 は引き継ぎの照合記録に一致した。PT の方式別648テンソルは保存モデル定義の対応キー・形状・型と一致した。元チェックポイントと配布前 ONNX は引き継ぎ ZIP に含まれず、採用 EMA 重みと配布 PT のテンソル値完全一致は開発者側の照合記録に基づく。
- 2方式に共通する PT テンソルキー528個のうち440個の値が異なる。得られた差は配布モデル全体の比較であり、後処理方式単独の効果推定には同一重みでの対照実験が必要。
- 元の最終評価は RTX 5090 上の PyTorch EMA、BF16、バッチ16、OpenCV の線形補間とリサイズ寸法の切り上げを使用する。NMS 版は同一候補から複数クラスを出せる評価用設定と torchvision NMS を使用する。今回の配布 ONNX は FP32、バッチ1、Pillow のバイリニア補間と四捨五入、NMS 版は候補ごとの最高得点クラスと OpenCV NMS を使用した。両方で COCO API と IoU・最低スコア・候補数の設定を照合した。これらの条件差それぞれの AP への寄与は未分離。

## ONNX 変換前後の精度

val2017 全5,000枚を配布 PT でも推論し、同じ各方式の GPU、FP32、TF32 無効、バッチ1、前処理、後処理、COCO API を用いて配布 ONNX の結果と比較した。実行基盤は PyTorch 2.8.0+cu126 と ONNX Runtime GPU 1.22.0。PyTorch は引き継ぎで検証した推論ローダーを使用した。[変換比較の全指標](conversion-accuracy.json)

| 方式 | 配布 PT の AP | 配布 ONNX の AP | 差（ONNX − PT、ポイント） |
| --- | ---: | ---: | ---: |
| NMS | 48.522650% | 48.522685% | +0.000035 |
| NMS フリー | 48.757742% | 48.757734% | −0.000008 |

どちらも AP の差の絶対値は0.00004ポイント未満。スコア > 0.35 では5,000枚すべてで検出件数が一致し、カテゴリーIDの並びは NMS 版4,999枚、NMS フリー版5,000枚で一致した。この比較は同一条件下の配布形式と実行基盤の差を測っている。

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

変換比較を再現する場合は、`predict_coco.py` の `--model` を対応する配布 PT に切り替え、`--torch-source` に検証済みの `inference/person_camera` を指定する。PT の推論結果を `score_coco.py` で集計し、`compare_formats.py --onnx quality.json --torch <PTの集計JSON> --output <比較出力JSON>` で比較指標を計算する。
