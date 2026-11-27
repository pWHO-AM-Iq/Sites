# 学習・再学習 / Training and adaptation

> ローカル検証ドラフト。学習は実施していません。 / Local verification draft. No training was run.

ここでのコマンドは `training` ディレクトリから実行します。元の学習は Python 3.12、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、RTX 5090、pycocotools 2.0.11 です。互換性を確認した CUDA 用 PyTorch、torchvision、NumPy、Pillow、pycocotools を仮想環境に用意してください。権限を確認済みの COCO アーカイブ（train2017、val2017、annotations_trainval2017）を `data/archives` に置きます。

Run from the `training` directory. The original run used Python 3.12, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, an RTX 5090, and pycocotools 2.0.11. Set up an isolated environment with compatible CUDA builds of PyTorch and torchvision, NumPy, Pillow and pycocotools. Place authorized copies of the three COCO ZIP files in `data/archives`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe prepare_data.py --source data/archives --destination data/coco
.\.venv\Scripts\python.exe train.py --config config/train.json --smoke --smoke-steps 3
```

`prepare_data.py` は既存アーカイブを読み、分割IDを生成します。スモークは対象環境で実施してください。全学習を明示的に開始する場合に限り、元の起動コマンドを使います。 / Data preparation reads existing archives and generates split IDs. Run the bounded smoke mode on the target system. The original full-run entry point requires an explicit start flag.

```powershell
.\Start-Training.ps1 -StartTraining -Config .\config\train.json
```

## 既存学習の再開と新規データへの適応 / Resume and adaptation

元トレーナーの `--resume` は、同じ実行の完全なチェックポイントから、モデル・EMA・optimizer・scaler・乱数状態を復元します。このローカル版の `--warm-start` は推論用PTから共有特徴抽出部だけを読み込み、2つの検出ヘッド、optimizer、学習エポックを新規に初期化します。ラベル対応が変わる場合も両ヘッドは新規になります。付属のCPU用確認コマンドは重みの形式、アーキテクチャ、テンソル名と形状を検査し、学習更新を行いません。

The original trainer resumes from a complete checkpoint in the same run, including model, EMA, optimizer, scaler and random generator state. The local `--warm-start` option loads the shared feature extractor from a bundled inference checkpoint and initializes both detection heads, optimizer and epoch state afresh. The CPU check validates the checkpoint architecture and tensor names and shapes without any training step.

```powershell
.\.venv\Scripts\python.exe warm_start.py ..\custom_detector_nms_free.pt --num-classes 80
.\.venv\Scripts\python.exe train.py --config config/train.json --start-training --warm-start ..\custom_detector_nms_free.pt
```

80クラスと3クラスのモデル構成でCPU確認済みです。新規データではCOCO形式のカテゴリIDと `num_classes` を合わせ、分割・データ読み込み・評価経路を確認してから実行してください。顧客データ取り込みと実際の微調整は未検証です。 / CPU initialization was verified with 80 and 3 target classes. For a new dataset, align the COCO-style category IDs and `num_classes`, then validate the splits, loader and evaluation. Customer dataset import and training updates are unverified.

パッケージ用に相対パスとウォームスタート入口を加えています。元ファイルは引き継ぎ側に保管されています。 / Packaging adds relative paths and the warm-start entry point. The handoff preserves the originals.
