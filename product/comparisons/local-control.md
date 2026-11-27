# ローカル対照試行　YOLO26m E2E

## CPU グラフの短時間試行

2026-11-27 に公式 YOLO26m 重みを取得し、同一マシン・同一ライブラリ・同一入力で Custom Detector NMS-free とグラフ実行を確認した。入力は COCO val2017 の4枚（139、190140、384527、581781）を同じ640 × 640の RGB レターボックスで前処理した FP32 テンソル。Intel Core i9-11900、ONNX Runtime 1.22.0 CPUExecutionProvider、逐次実行、intra-op 4/inter-op 1、各4ウォームアップ、各12計測でモデル実行順を反復ごとに反転。モデル読み込み、画像デコード、前処理、グラフ外の後処理は計測外。計測中は別のGPU評価ジョブが動作しており、そのホスト側処理とCPUリソースが競合し得る。

| モデル | グラフ出力 | 中央値 ms |
| --- | --- | ---: |
| Custom Detector NMS-free | 生の矩形と80クラス確率。Top-K は呼び出し側 | 303.52 |
| YOLO26m E2E | グラフ内 Top-K 適用済みの `[1, 300, 6]` | 315.52 |

出力の処理範囲が異なるため、E2E の速度優位やモデルの一般性能はこの試行から判断しない。精度は本試行の対象外。YOLO11m、RT-DETR-R18、D-FINE-M、RF-DETR は公式公開値の参照まで。

YOLO26m は [Ultralytics v8.4.0 の公式資産](https://github.com/ultralytics/assets/releases/tag/v8.4.0) の `yolo26m.pt`（44,255,705 bytes、SHA-256 `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`）。Python 3.12.0、PyTorch 2.8.0+cu126、Ultralytics 8.4.0、ONNX 1.18.0 で CPU・バッチ1・固定640・簡略化なし・opset 17 としてローカルエクスポートし、SHA-256 は `a20a286669500235414aa25ca0c6c076f30b0578557d2a630f0117c1ed2aa056`。公式配布の重みと変換後のグラフは Git 除外済み `private/comparators/` に保管した。同フォルダの `control-trial.py` と `control-trial.json` に手順と全計測値を記録した。

## COCO val2017　全5,000枚

検証セット、80クラス、Pillow の RGB・バイリニア・640 × 640 レターボックス、スコア下限（0.001超）、上位300件、COCOeval bbox（pycocotools 2.0.11、maxDets 100 の AP）が共通。ONNX Runtime GPU 1.22.0 CUDA EP、バッチ1、FP32、TF32無効、CPU 2スレッド。Ultralytics ONNX はグラフに含まれる上位選択結果を使用し、ホスト側で座標の復元・クリップ・スコア下限の適用を行った。Custom Detector の NMS-free はホストで上位を選択した。

| モデル | GPU | AP50–95 | AP50 | AP75 | AP-S | AP-M | AP-L | AR100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom Detector NMS | RTX 2080 | 48.52 | 65.94 | 52.43 | 30.80 | 53.64 | 64.99 | 63.99 |
| Custom Detector NMS-free | RTX 3080 | 48.76 | 66.10 | 52.95 | 31.15 | 53.35 | 64.63 | 68.08 |
| YOLO26m E2E | RTX 3080 | 52.50 | 69.70 | 57.28 | 35.87 | 56.93 | 68.64 | 72.12 |

数値は百分率。同じ RTX 3080 上で評価した YOLO26m E2E は、Custom Detector NMS-free を AP50–95 で3.75ポイント上回った。この差は2つの実アーティファクトの観測値。学習設定、初期化の独立検証状態、モデル設計とエクスポートの条件が異なるため、差を特定の設計変更に帰属できない。YOLO26m ジョブの5,000枚の予測・JSONL保存は100.4秒。アーカイブ読出し、前後処理、ファイル書込を含むため速度ランキングには使わない。

[集計と設定 JSON](local-control.json) · [Custom Detector 評価手順](../evaluation/README.md)。YOLO26m のアダプター、全予測 JSONL と評価原本は Git 除外済みの `private/comparators/` に保存した。
