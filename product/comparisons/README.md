# 物体検出モデルの公式公開値

2026-11-27 確認。COCOの検出 AP@[.50:.95] は百分率。解像度は正方形の辺、パラメータ数は百万単位。詳細数値は [official-baselines.json](official-baselines.json) に保存した。空欄（JSON の null）は採用ソースの該当列がないことを示す。比較対象の実測値として転記しない。

| 候補 | 入力 px | AP | パラメータ M | CPU ONNX ms | T4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO26n | 640 | 40.9 | 2.4 | 38.9 | 1.7 ms |
| YOLO26s | 640 | 48.6 | 9.5 | 87.2 | 2.5 ms |
| YOLO26m | 640 | 53.1 | 20.4 | 220 | 4.7 ms |
| YOLO26l | 640 | 55 | 24.8 | 286.2 | 6.2 ms |
| YOLO26x | 640 | 57.5 | 55.7 | 525.8 | 11.8 ms |
| YOLO11n | 640 | 39.5 | 2.6 | 56.1 | 1.5 ms |
| YOLO11s | 640 | 47 | 9.4 | 90 | 2.5 ms |
| YOLO11m | 640 | 51.5 | 20.1 | 183.2 | 4.7 ms |
| YOLO11l | 640 | 53.4 | 25.3 | 238.6 | 6.2 ms |
| YOLO11x | 640 | 54.7 | 56.9 | 462.8 | 11.3 ms |
| RT-DETR-R18 | 640 | 46.5 | 20 | — | 217 FPS |
| RT-DETR-R34 | 640 | 48.9 | 31 | — | 161 FPS |
| RT-DETR-R50 | 640 | 53.1 | 42 | — | 108 FPS |
| RT-DETR-R101 | 640 | 54.3 | 76 | — | 74 FPS |
| D-FINE-N | 640 | 42.8 | 4 | — | 2.12 ms |
| D-FINE-S | 640 | 48.5 | 10 | — | 3.49 ms |
| D-FINE-M | 640 | 52.3 | 19 | — | 5.62 ms |
| D-FINE-L | 640 | 54 | 31 | — | 8.07 ms |
| D-FINE-X | 640 | 55.8 | 62 | — | 12.89 ms |
| RF-DETR-N | 384 | 48.4 | 30.5 | — | 2.3 ms |
| RF-DETR-S | 512 | 53 | 32.1 | — | 3.5 ms |
| RF-DETR-M | 576 | 54.7 | 33.7 | — | 4.4 ms |
| RF-DETR-L | 704 | 56.5 | 33.9 | — | 6.8 ms |
| RF-DETR-XL | 700 | 58.6 | 126.4 | — | 11.5 ms |
| RF-DETR-2XL | 880 | 60.1 | 126.9 | — | 17.2 ms |

## データの読み方

- **Ultralytics** の比較表を採用。CPU は ONNX Runtime FP32（32コア Intel Xeon Skylake）、GPU は NVIDIA T4 の TensorRT 10 FP16。YOLO26 の通常 AP と E2E AP は別列で、例として n は 40.9 と 40.1。NMS-free の比較では E2E 分を対応付け、採用ハッシュと出力モードを記録する。 [公式比較](https://docs.ultralytics.com/compare/yolo26-vs-yolo11) · [YOLO26 検出表](https://docs.ultralytics.com/models/yolo26) · [公式実装](https://github.com/ultralytics/ultralytics)
- **RT-DETR** は著者実装の COCO 行、原版を採用。RTX などに読み替えず、T4 TensorRT FP16 の公開 FPS をそのまま掲載した。Objects365 の追加事前学習行と v2/v4 は別候補。 [公式実装](https://github.com/lyuwenyu/RT-DETR) · [CVPR 論文](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf)
- **D-FINE** は著者の COCO 行。T4、TensorRT 10.4.0、FP16、バッチ1、COCO val2017。640 の入力は公開ベンチ表でも照合。追加データ条件の Objects365+COCO 行は別扱い。 [著者実装](https://github.com/Peterande/D-FINE) · [入力サイズ照合](https://github.com/roboflow/rf-detr)
- **RF-DETR** は Roboflow のリポジトリ表。COCO val2017 の5,000枚、pycocotools、SAB。T4 TensorRT FP16、バッチ1。同じリポジトリの他社行は Roboflow による再測定のため、今回の他社著者値には混ぜていない。XL、2XL は PML 1.0、N/S/M/L は同表で Apache 2.0 と表示。 [公式実装](https://github.com/roboflow/rf-detr) · [論文](https://arxiv.org/abs/2511.09554)

## 選定と比較設計

- Custom Detector の有効パラメータは両方式とも 20,435,260。主比較は、同系統の YOLO26m（20.4M、640、NMS と E2E をそれぞれ対応）、前世代の YOLO11m（20.1M、640、NMS）、Transformer 系の RT-DETR-R18（20M、640）と D-FINE-M（19M、640、COCO 行）。RF-DETR-N（30.5M、384）と S（32.1M、512）はパラメータ数・入力解像度が異なる設計比較として扱う。
- ベンダー表は処理範囲やバージョンが揃っていない。自環境の計測は同一ランタイム・精度・端末、データ分割・カテゴリ対応・閾値・最大検出数、バッチ、前後処理と同期、ウォームアップ、測定回数を固定し、エンジン単体と画像入力から出力までを分ける。公称 FPS を逆算した ms をレイテンシ実測欄へ入れない。
- Custom Detector の NMS/NMS-free、PT/ONNX はそれぞれのハッシュに結果を紐づける。人物デモの画像1枚の時間は参考値とし、全80クラス COCO AP と区別する。同条件で得た AP/AP50/AP75 と速度の再計測結果だけを対照表の同じ区画に載せる。
- 比較モデルは公式配布先の条件を確認し、必要なモデルだけ別途取得する。この資料に他社重みやコードは含めない。公開する比較表示は検証結果と条件を併記する。 [消費者庁の比較広告ガイドライン](https://www.caa.go.jp/policies/policy/representation/fair_labeling/representation_regulation/comparative_advertising)

## 自環境の比較

[ローカル対照試行](local-control.md) に、公式 YOLO26m から変換した E2E グラフの COCO val2017 全5,000枚の AP、Custom Detector の対応結果、CPU グラフの短時間試行を記載した。
