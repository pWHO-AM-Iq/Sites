# 物体検出モデルの製品化レビュー

2026年11月27日 · 製品資料 `draft.1` · ChatGPT Sites 保存版 v1。開発・試験・配布準備を実施し、公開と販売開始は承認待ちです。

## 1. 公式サイトとプレビュー

[日本語プレビュー](http://127.0.0.1:8765/) · [English preview](http://127.0.0.1:8765/en/) · [性能評価](http://127.0.0.1:8765/benchmarks/) · [Sites保存版の記録](site/review-version.json)

リポジトリ直下の `main/` が1サイトのルートです。Home、Models、モデル詳細、Benchmarks、Custom Training、Research、Documentation、Store、Legalの9ページを日英で生成しました。PC・モバイル幅、言語切替、検出枠の表示切替をブラウザで確認し、18ページ、514個の内部参照、7図、10件の速度測定の整合性を自動確認しました。ブランド「Vision Model Lab」と製品名「Custom Detector」は仮称です。[サイト設定](../main/config.json) · [再生成・起動方法](../README.md)

トップの交差点は内蔵ImageGenによる作例です。表示した8件の矩形・クラス・スコアは配布候補のONNXによる実際の出力です。[使用画像](../main/dist/assets/crossing.jpg) · [生成プロンプトと作成記録](site/asset-record.md) · [出力JSON](../main/dist/data/example.json)

## 2. 変更ファイル

[リポジトリ変更一覧](changed-files.txt)に起点コミットからの相対パスを収録しました。サイト、評価、比較、製品資料、販売準備を分けています。学習データ、元の配布用モデル、引き継ぎ原本、試験中間ファイルはGit管理対象外です。ローカル専用ZIPは[ハッシュ一覧](../private/packages/SHA256SUMS.txt)から確認できます。

## 3. 性能試験

このPCに元からあるCOCOの画像・注釈を使いました。val2017全5,000枚・80クラス、640 × 640、バッチ1。配布ONNXをFP32・TF32無効で評価し、公式pycocotoolsのbbox指標を算出しました。下表は百分率です。

| 方式 | AP50–95 | AP50 | AP75 | AP小 / 中 / 大 | AR@100 | Precision / Recall / F1 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| NMS | 48.52 | 65.94 | 52.43 | 30.80 / 53.64 / 64.99 | 63.99 | 81.36 / 58.22 / 67.87 |
| NMSフリー | 48.76 | 66.10 | 52.95 | 31.15 / 53.35 / 64.63 | 68.08 | 82.01 / 56.66 / 67.02 |

P/R/F1はIoU 0.50、信頼度 > 0.35、全クラスのmicro集計です。配布モデルの両方式は別エポックの重みを使います。今回のONNX評価と同じ前後処理で元のPyTorchを全件評価したAP差は、NMSが+0.000035ポイント、NMSフリーが−0.000008ポイント（ONNX−PyTorch）でした。[全指標・しきい値・クラス別PR](evaluation/quality.json) · [変換精度](evaluation/conversion-accuracy.json)

学習時の最終評価はNMSが49.17%、NMSフリーが48.71%です。元の評価はBF16、リサイズ方法とNMSの候補選択も製品評価とは異なります。学習116,287枚・選定2,000枚・最終5,000枚を公式注釈から再構成し、IDの重複0件を確認しました。[元記録との照合](evaluation/origin-training.json)

CPU（Core i9-11900）とGPU（RTX 3080）でPyTorch CPU FP32、CUDA FP32/FP16、ONNX Runtime CPU/CUDAを両方式で測定しました。例えばNMSフリー・PyTorch CUDA FP32は平均27.38 ms、P95 29.13 ms、36.52 FPSでした。E2EはJPEGデコード・前処理・転送込み推論・後処理を含みます。各10回ウォームアップ後100回測定し、一部区間はほかの評価と並行しました。全10構成でMP4の30フレーム処理を確認し、同じ画像の反復出力に不一致はありませんでした。[速度CSV](materials/runtime.csv) · [処理別時間・メモリ・互換性](evaluation/runtime/measurements.json)

## 4. 既存モデルとの比較

公式YOLO26m E2Eを製品から隔離した環境で測定しました。同じ5,000枚でAP50–95は52.50%、Custom Detector NMSフリーは48.76%。この条件で3.75ポイントの差でした。各モデルの学習履歴は異なります。YOLO11、RT-DETR、D-FINE、RF-DETRを含むメーカー・著者公表値は別表にまとめています。[自環境の対照実験](comparisons/local-control.md) · [5系列25モデルの公開値・出典](comparisons/README.md)

## 5. スペック表・仕様書

総・学習可能パラメータ数はいずれも21,896,248、推論時は各方式20,435,260。640 × 640のConvとMatMulを集計すると34.213 GMAC、積和を2 FLOPとして68.426 GFLOPsです。元のモデル定義、PTテンソル、ONNXグラフの照合範囲も記録しました。[日本語仕様書](materials/SPEC.ja.md) · [English specification](materials/SPEC.en.md) · [モデルカード](materials/MODEL_CARD.md) · [ベンチマーク報告](materials/BENCHMARK.md) · [更新履歴](materials/CHANGELOG.md) · [再生成手順](materials/BUILD.md)

## 6. 販売パッケージ

| ローカル配布候補 | 同梱内容と確認 |
| --- | --- |
| [Standard](../private/packages/custom-detector-standard-draft.1.zip) | NMSフリーのPT・ONNX、推論例、元の推論ツール、日英資料・通知。人物検出を確認。 |
| [Developer](../private/packages/custom-detector-developer-draft.1.zip) | 両方式のPT・ONNX、推論・学習コード、相対パスの設定、開発者手順。80/3クラスでウォームスタート初期化を確認。 |
| [Commercial](../private/packages/custom-detector-commercial-draft.1.zip) | 両方式と社内契約検討用資料。承認済みの商用許諾契約は未確定。 |
| [Custom Training](../private/packages/custom-detector-custom-training-draft.1.zip) | 受付票、オフライン検査、YOLO TXT→COCO JSON変換、納品・見積もりの運用案。合成データで検査・変換を確認。 |

4 ZIPのSHA-256、CRC、同梱資料の相対リンクを確認しました。既存ウェイトや学習画像の変更、新規学習は実施していません。[検収・データ管理・見積もり](launch/custom-training-operations.md)

## 7. 外部掲載資料

BOOTHの日本語商品説明・Gumroadの英語商品説明と画像、Hugging Faceモデルカード、GitHub推論例、Zenn・Qiita・note記事、X告知をローカルに用意しました。2026-11-27に確認した規約、AI関連方針、外部リンク条件、手数料と要確認事項を媒体別に記録しました。[媒体別原稿と画像](launch/README.md) · [利用条件・手数料と公式情報](launch/platforms.md)

## 8. ライセンス・来歴

元の学習記録はCOCOでのランダム初期化、外部事前学習ウェイト・再開元なしを裏づけます。学習・推論ソースと元の推論ZIPには、Ultralyticsから翻案した実装部分を列挙したAGPL-3.0通知がありました。スクラッチ学習は確認済みで、配布するコード・ウェイト・ONNXの許諾範囲は販売方式ごとに判断が必要です。COCO画像・アノテーション、依存物、参考論文、通知、特許候補を記録しました。[権利レビュー](launch/rights-review.md) · [特許文献の初期調査](launch/patent-search.md)

## 9. 未実施・未検証

TensorRTは必要DLLを解決できず、INT8、バッチ1以外の速度、ORTのプロセス別最大VRAMは未測定。長時間運用、実際の顧客データでの追加学習・精度・必要工数も未検証です。元の学習プロジェクト自身のGitコミット、学習開始時点の重みファイル、履歴上の変換前バイナリが引き継ぎに含まれない箇所は、保存ソースのハッシュと当時の照合記録を使いました。

## 10. 販売開始前の作業

1. 配布形態別の許諾、対応ソースの提供方法、必要な第三者通知、商用契約とサンプルのライセンスを確定する。
2. 運営者、連絡先、価格、提供時期、返品・キャンセル、データ保管・削除・秘密保持、販売方式に応じた表示を確定する。
3. 受託学習の受付上限・納期は、実案件に近い条件で承認後に必要工数を計測して設定する。元のRTX 5090学習ログには300エポックの学習ループ約45.15時間が記録されている。
4. BOOTH・Gumroad等の商品分類と掲載条件を確認し、購入先を確定する。サイトのデプロイ、GitHub Push、外部投稿、販売開始は個別の承認後に行う。
