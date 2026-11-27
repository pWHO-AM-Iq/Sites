# Custom Detector ランタイム評価

固定バッチ1、640 × 640、16枚の COCO val2017 JPEG を使用する。ZIP からの読み込みとモデル初期化を計測区間の外に置く。JPEG のデコード、レターボックス、推論、後処理、各処理を含む E2E の平均・中央値・P95 と、E2E 平均から FPS を計算する。GPU もホスト出力が得られるまでを推論区間とし、デバイス転送を含む。後処理は80クラスを対象に信頼度 0.35、最大300件とし、NMS 版はクラスごとに IoU 0.7 で抑制する。COCO AP 計測には別の評価スクリプトを使う。

```powershell
python -B product/evaluation/runtime/benchmark.py --backend ort-cpu --mode nms_free --input E:/dataset/coco/val2017.zip --output private/product-evaluation/runtime/ort-cpu-nms_free.json --threads 4 --warmup 10 --iterations 100 --compatibility
```

NMS 版は `--mode nms`、CUDA 版は CUDA 対応 ONNX Runtime 環境で `--backend ort-cuda --gpu 0`、TensorRT Execution Provider は対応 CUDA・TensorRT 環境で `--backend ort-trt` を指定する。GPU の EP が初期化されなければ失敗を記録する。グラフの一部は CPU に配置される可能性がある。ONNX Runtime の CUDA 既定設定は TF32 を許可する。使用したプロバイダーと設定は JSON に記録する。

PyTorch は信頼できる配布元の `core.py` と同梱モデル定義がある場合に `--core <配布ツールのフォルダ>` と `--backend torch-cpu`、`torch-fp32`、`torch-fp16` を使用できるようにした。ここでの PyTorch アダプターは配布元のモデル定義・チェックポイント読み込み関数を前提とする。

`--compatibility` は JPEG、PNG、極端な縦横比、黒画像、低照度、ノイズ、縮小配置、4枚のタイル、空バイト列、30フレームの MP4 を検査する。変換元は選択された JPEG の先頭1枚。推論出力の有限性と座標の順序、同一 JPEG の反復結果を確認する。検出件数は入力差への反応を示す参考値であり、各ケースの精度は COCO AP と合わせて判断する。

```powershell
python -B product/evaluation/runtime/summarize.py private/product-evaluation/runtime --output private/product-evaluation/runtime/summary.json
```

比較は同じ入力16枚を用い、クラスが同一で IoU が 0.5 以上の検出を1対1に対応させ、対応なし件数と座標・信頼度の最大差を記録する。メモリは各リクエスト終了時点で観測したプロセス RSS。ONNX Runtime のデバイス VRAM ピークは計測していない。連続処理の測定期間は設定された反復数と動画フレーム数に限る。
