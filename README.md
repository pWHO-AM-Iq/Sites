# Sites

公式サイトは `main/` にあります。ローカルと GitHub は同じディレクトリ構成を使います。現在は1つの ChatGPT Sites 標準URLで運用し、将来サブドメインが必要になったときに `main/` と同じ階層へ専用フォルダを追加します。

| ディレクトリ | 用途 |
| --- | --- |
| `main/` | 公式サイト。公開対象は `dist/` の静的ファイル |
| `product/` | 製品評価、資料生成、販売前の確認・掲載原稿 |
| `models/` | 手元の評価用モデル。Git管理対象外 |
| `private/` | 配布候補ZIP・試験の中間ファイルなど。Git管理対象外 |

ブランド・製品名は `main/config.json` で設定します。ルートで `python main/build.py` を実行すると、日英ページが `main/dist/` に生成されます。プレビューは `python -m http.server 8765 --bind 127.0.0.1 --directory main/dist` で開けます。

評価結果を更新したときは、リポジトリ直下で次の順に実行します。

```powershell
python product/materials/build_materials.py
python product/site/build_public_details.py
python main/build.py
python product/site/verify_site.py
```

追加製品は `main/config.json` の `additional_products` 配列に登録できます。各項目には `slug`、種別 `kind`（`model`・`software`・`learning`）、日英の `name`、`summary`、`description`（各々 `ja`・`en` を持つオブジェクト）を設定します。ビルド時に同じURL配下に製品ページを作り、HomeとStoreからリンクします。

GitHubへのPush、Sitesへのデプロイ、外部への掲載と販売開始は個別承認後に行います。
