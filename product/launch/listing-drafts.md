# 掲載文の下書き

「Vision Model Lab」「Custom Detector」は仮称。`[[...]]` は差替欄。価格、利用条件、体制、掲載 URL、公開済みファイルを実物と一致させる。予約注文・先行決済は開始しない。掲載条件は [公開先と販売条件](platforms.md)、ファイルの確認手順は [権利レビュー](rights-review.md) を参照。

## BOOTH 商品説明（日本語）

**タイトル**　Custom Detector｜物体検出モデル・ローカル推論サンプル

画像から物体の位置を検出するモデルのダウンロードセットです。COCO 2017 の80クラスを対象とし、入力は640 × 640です。付属例では画像から人物を検出し、元画像上の矩形座標と信頼度を JSON で確認できます。学習時の設定・起動記録・保存ソースには、COCO 2017 を使い、外部の事前学習済み重みを読み込まずランダム初期化から学習した記録があります。モデル定義は Ultralytics YOLO26 を翻案しており、適用条件はライセンス欄をご覧ください。

- Standardの内容　NMSフリー版の `custom_detector_nms_free.pt` と `custom_detector_nms_free.onnx`、人物検出の実行例、導入手順、日英仕様書、第三者通知。
- 動作確認　Windows 11、Python 3.12、ONNX Runtime 1.22、PyTorch 2.8。COCO val2017の5,000枚でNMSフリー版のAP50–95は48.76%。詳しい条件は技術資料に掲載します。
- ライセンス　[[モデル・コード別の条項、商用利用と再配布可否、ライセンスへの導線]]
- 価格・提供時期　税込 [[価格]] 円。[[支払い・ダウンロード条件]]
- サポート　[[対象、期間、連絡先、返金・再提供条件]]
- 特定商取引法に基づく表記　[[実際の表記 URL]]

検出結果には誤検出や未検出が含まれます。速度と精度は対象画像、設定、実行環境によって変わります。検証条件は [[技術資料 URL]] をご覧ください。

掲載画像は [listing-ja.png](assets/listing-ja.png)。図は推論出力の形式を示す説明用イラストです。

## Gumroad product description (English)

**Title**　Custom Detector — Object Detection Model Files and Local Inference Examples

Run an object detector on your own images and inspect the output locally. The model targets the 80 COCO 2017 classes with a 640 × 640 input. The included example returns person detections as bounding boxes and confidence scores in JSON. The archived training configuration, source, and startup receipt support training on COCO 2017 from random initialization without external pretrained weights. The model definition adapts Ultralytics YOLO26. See the license section for the distribution terms.

**Standard download** includes the NMS-free checkpoint and ONNX export, a local person-detection example, setup instructions, bilingual specifications and third-party notices.

**Tested environment** was Windows 11 with Python 3.12, ONNX Runtime 1.22 and PyTorch 2.8. The NMS-free ONNX artifact scored 48.76% AP50–95 on all 5,000 COCO val2017 images.

**Price and delivery**　[[USD price and download instructions]]  
**License**　[[model and code terms, commercial use and redistribution terms]]  
**Support and refunds**　[[contact, support scope and policy]]  
**Technical documentation**　[[evaluation report and release URL]]

Accuracy and latency depend on the images, settings and hardware. See the technical report for the measured conditions. Downloadable standalone files are delivered through this listing.

Use [listing-en.png](assets/listing-en.png). The diagram illustrates bounding-box output.

## Hugging Face モデルカード

権利確定後の YAML 例。`license` は配布物に適用される許諾の実値に置き換える。

```yaml
---
language:
  - en
  - ja
pipeline_tag: object-detection
datasets:
  - detection-datasets/coco
license: "[[verified Hugging Face license identifier]]"
tags:
  - onnx
  - pytorch
---
```

**Custom Detector** is a 640 × 640 object detector targeting 80 COCO 2017 categories. The release documents the model variants and the person-detection example, along with the input tensor and bounding-box output formats.

- Artifacts and SHA-256　[[files, upstream provenance, license and checksums]]
- Training　Training records document random initialization, with no external pretrained weights or resume checkpoint. The model definition adapts Ultralytics YOLO26. The existing notice identifies the adapted code as AGPL-3.0. Link to the final release notices and terms once approved.
- Evaluation　[[COCO val2017 metric values and settings, or status if pending]]
- Inference　[[tested CPU environment, dependencies, reproducible command and expected JSON]]
- Limitations　Occlusion, small objects and changes in the image domain can affect detections. Evaluate the target use case with representative data.
- License and notices　[[verified model/code terms and bundled notices]]
- Source and issue reporting　[[technical repository and issue URL]]

上記 datasets タグは実際の学習元と一致する場合のみ使用し、COCO 公式配布から取得した場合は本文に公式 URL、分割、取得記録を示す。

## GitHub README / Release

**Custom Detector inference tools**

このリリースでは、640 × 640 の固定入力から人物検出の結果を JSON で取得する実行例を提供します。デモは COCO の人物クラスを選別します。モデル入力の作り方、NMS / NMS-free の出力処理、元画像への座標復元、確認済み環境は [[技術ドキュメント URL]] を参照してください。

- セットアップと起動コマンド　[[リリースと照合済みのコマンド]]
- 検出結果の例　[[自作・許諾済み画像を使った結果と期待形状]]
- リリース内容と整合性　[[対象ファイル・SHA-256・バージョン]]
- ソース・ライセンス　[[参照元の固定コミット・対応ソース・NOTICE]]
- 不具合報告　環境、バックエンド、モデル識別子、ログをお知らせください。画像を添付する場合は公開できる素材を使用してください。

## Zenn 技術記事

**タイトル**　640 × 640 固定入力の物体検出で、矩形を元画像へ戻すまで

縦長画像を物体検出モデルへ渡すとき、画像をリサイズした位置と余白を記録しておくと、モデルが返した矩形を元のピクセル座標へ戻せます。Custom Detector の推論例では、RGB画像を縦横比を保ってリサイズし、640 × 640 の中央に置いて推論しています。

横座標は余白の左端までの距離を引いて実際の横倍率で割り、縦座標は上端までの距離を引いて実際の縦倍率で割ります。リサイズ時に整数へ丸めるため、使用した横・縦の寸法から倍率を求めます。最後に元画像の範囲へ収めます。

[[著者が権利確認済みの縦長と横長の画像で再現し、元サイズ・リサイズ後・余白と実際の JSON を追記]]  
関連する実行例　[[推論リポジトリの文書 URL]]

## Qiita 技術記事

**タイトル**　ONNX Runtime の物体検出出力を点検する。boxes と scores の形状と座標系

固定解像度の検出器をつなぐとき、出力が画像上のピクセル値か、0〜1に正規化された値かを先に確認すると、描画時の拡大率を揃えやすくなります。対象の ONNX モデルは `boxes` が `[1, 8400, 4]` で、640 × 640 入力上の xyxy を返します。`scores` は `[1, 8400, 80]` の確率です。`scores` をそのまま閾値処理に渡します。

```python
import sys

import onnxruntime as ort

session = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
for output in session.get_outputs():
    print(output.name, output.shape, output.type)
```

[[著者が採用アーティファクトでコマンドを実行し、実際の出力・バージョン・採用した閾値を追記]]  
関連資料　[[ONNX 入出力仕様・検証資料 URL]]

## note 開発記事

**タイトル**　画像の中のどこに何があるかを調べる、小さな物体検出モデルの開発

物体検出では、写真に人物がいるかを判定するだけでなく、画像のどの位置に人物がいるかを矩形で記録します。今回の推論例では、座標と信頼度を JSON で取り出せるため、後から同じ位置へ枠を描いたり、画像の一部を切り出したりできます。

公開に向けて整えているのは、モデルファイルに加えて、画像を用意して推論結果を確認するまでの手順です。実行環境、画像サイズ、誤検出や見落としの例を技術資料にまとめています。

[[著者自身の開発時の判断と実際に確かめた例を1件追記]]  
技術資料　[[公開後の文書 URL]]

## X 告知

**日本語・公開準備中**  
物体検出モデル「Custom Detector」の公開準備を進めています。640 × 640 入力で、付属の人物検出例は矩形座標と信頼度を JSON に出力します。図は出力形式のイメージです。

**日本語・技術資料公開後**  
Custom Detector の技術資料を公開しました。人物検出の実行例、NMS の有無による出力処理、動作確認環境を掲載しています。[[公開済み技術資料 URL]]

**English · after documentation release**  
Custom Detector documentation is live. It covers local person detection, bounding-box output in JSON, and the tested Python runtimes. [[published documentation URL]]

## 受託学習の問い合わせ欄

手元の画像と対象クラスに合わせた検出モデルについて相談できます。対象物、設置環境、端末、希望納期を文章でお知らせください。素材の受け渡しは、利用条件・秘密保持・受け渡し方法に合意した後にご案内します。お問い合わせ先は [[承認済み連絡先]] です。

For custom detector training, tell us what you need to detect, your deployment environment, and your timeline. We will discuss data permissions, secure transfer, and acceptance criteria before requesting images. Contact [[approved business contact]].
