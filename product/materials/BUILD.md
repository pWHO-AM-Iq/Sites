# 評価資料の再生成 / Rebuilding the evaluation materials

Python 3.12、Matplotlib 3.9.2 で実行します。リポジトリルートで以下を実行してください。 / Run from the repository root with Python 3.12 and Matplotlib 3.9.2.

```powershell
python product/materials/build_materials.py
python product/materials/build_packages.py
```

ローカル配布候補の再生成には、検証済み引き継ぎの展開先 `private/handoff/source/` を使います。別の保存場所では `--handoff <展開先>` を指定してください。元の推論ZIPとDeveloper向け学習ソースをマニフェストで照合します。 / Package builds use the verified extraction at `private/handoff/source/`. Use `--handoff <extracted-root>` to select a different extraction.

`build_materials.py` は `product/evaluation/quality.json`、`product/evaluation/runtime/measurements.json`、`product/evaluation/artifact-audit.json`、`product/comparisons/official-baselines.json` を読み込みます。パスを指定する場合は `--quality`、`--runtime`、`--audit`、`--references` を使います。COCO推論や学習はこの処理の外側です。パッケージはドキュメント、許可リストのファイル、SHA-256から構成し、`private/packages` に出力します。

既定のランタイム入力は、元の個別リクエストデータを省いた再生成用の集計です。元データを使う場合は `--runtime RUN.json ...`、集計を更新する場合は `--refresh-runtime SUMMARY.json` を指定してください。 / The default runtime input is the checked-in aggregate report. Use `--runtime RUN.json ...` to override it with raw runs or `--refresh-runtime SUMMARY.json` to regenerate it from a raw summary.

The materials builder also reads `product/comparisons/local-control.json` (override with `--local-control`) and checks that its annotation digest and image count match the Custom Detector evaluation. The command-line options can override the inputs. The package builder uses an allowlist, verifies the source model digests and every archived member, then writes the archives and SHA-256 file to `private/packages`.

`main/config.json` の製品名、`release.json` の版と学習来歴、評価アーティファクトの監査結果、`templates/` から README、日英の仕様書、モデルカード、変更履歴を生成します。 / The product config, versioned release record, artifact audit and document templates generate the README, Japanese and English specifications, model card and changelog.

## Web JSON v1

`main/dist/data/evaluation.json` contains:

- `schema_version` · integer 1
- `dataset` · available COCO split name and counts, empty until evaluation data is supplied
- `models[]` · `id`, `label`, `scope`, `status`, nullable `ap50_95` and `ap50` in percent, nullable `params` in count, nullable `latency_ms` for local ONNX CPU E2E median in milliseconds, `latency_scope`, `latency_provenance`, `latency_backend`, `provenance`, `params_provenance`, `source_url`
- `runtime[]` · local aggregate timing, backend, batch size, thread and warmup settings, tested input count and status
- `charts` · relative SVG paths for `accuracy`, `latency`, `parameters`, `class_ap`, `params_vs_ap`, `latency_vs_ap` and `pr_curve`

`official_validation` is local inference on the official COCO validation split, with `measured` or `unavailable` status. `local_control` is an independently measured comparator on the same validation dataset. `official_reference` rows carry vendor-reported results and attribution. Runtime medians carry their own `local_smoke` measurement scope. The export uses an explicit allowlist of fields. Raw detections, image records, system paths and checkpoints stay outside the website. Null metrics are missing measurements.

`models.csv`, `runtime.csv`, `class-ap.csv`, `coco-summary.csv`, `operating-points.csv` and `BENCHMARK.md` are regenerated alongside this site data. CSV nulls are empty cells. AP and AR are in percent. The score thresholds and precision, recall and F1 values in `operating-points.csv` are fractions.
