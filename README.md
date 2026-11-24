# Sites

ChatGPT Sites で公開するサイトを、サブドメインごとに `sites/<サブドメイン名>/` で管理します。

たとえば `blog.example.com` と `tools.example.com` を作る場合は、次の配置になります。

```text
sites/
├── blog/
└── tools/
```

各フォルダが、そのサイトのソース、画像、ビルド設定のルートです。ChatGPT Sites にはサイトごとに登録し、発行された `project_id` を各フォルダ内の `.openai/hosting.json` に保存します。

更新時は、対象のフォルダの内容だけをその Site に同期してデプロイします。たとえば `blog` の記事と画像を更新する場合、公開対象は `sites/blog/` です。
