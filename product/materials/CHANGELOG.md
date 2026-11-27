# 変更履歴 / Changelog

## 2026-11-27 · draft.1

- NMS・NMSフリーの既存ONNXとPyTorchチェックポイントの一致を元READMEのSHA-256で確認。 / Verified both existing ONNX models and checkpoints against the original README’s SHA-256 values.
- ONNX Runtimeによる人物検出サンプルと日英仕様を追加。 / Added a person detection example for ONNX Runtime and bilingual specifications.
- 評価データの生成手順と出所付きの表示資料を追加。 / Added a reproducible evaluation presentation pipeline with provenance.
- Standard、Developer、Commercial、Custom Trainingのローカルドラフトを作成。 / Prepared local drafts for Standard, Developer, Commercial and Custom Training.
- 受領した元学習記録を基に初期化、データ分割、元の評価値を文書化。 / Documented initialization, dataset splits and archived metrics from the original training records.

## 評価記録 / Evaluation record

- 版 / Release · `draft.1`
- 結果記録日時 / Results recorded at · `2026-11-27T19:19:47+09:00`
- 評価コードのコミット / Evaluation source commit · `fe236d2bb887abf30b486cfc47eccd1fcb4a223e`
- 評価の出所 / Evaluation record · `product/evaluation/run-record.json`
- 重みと ONNX の SHA-256 / Artifact hashes · [Original model README](../../models/custom_detector/README.md#ファイルの照合), [Artifact audit](../evaluation/artifact-audit.json), and the packaged `SHA256SUMS.txt`.
