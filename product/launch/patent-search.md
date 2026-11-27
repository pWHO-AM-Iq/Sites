# 物体検出に関係する特許文献の初期調査

2026-11-27、製品のbbox候補の絞り込みとNMSに関し、英語の「object detection」「non-maximum suppression」「bounding box」「candidate filtering」を組み合わせて公開公報を確認した。次の文献は請求項調査の候補で、製品への適用有無は未判定。

| 文献 | 照合すべき技術要素 |
| --- | --- |
| [EP 3876201 B1 · Object detection and candidate filtering system](https://data.epo.org/publication-server/rest/v1.2/publication-dates/2024-05-22/patents/EP3876201NWB1/document.pdf) | 候補のフィルタリング、判断手順と使用する情報を、実装の候補選択順・スコア処理と照合する。 |
| [US 12073612 B2 · Apparatus and method for processing detection boxes](https://patents.google.com/patent/US12073612B2/en) | 画像タイルと検出矩形の重なりを使う抑制手順。推論コードと出荷先ランタイムの内部実装を区分して調べる。 |
| [US 11798269 B2 · Fast non-maximum suppression algorithm for object detection](https://patents.google.com/patent/US11798269B2/en) | 生出力から候補を早期除外する条件と非線形処理の順序を、候補生成・確率変換・抑制の実際の順序と照合する。 |

販売国と販売形態の決定後、各国ファミリー、存続・失効、独立請求項とその限定、ライセンス関係を公的記録で追跡する。今回のキーワード検索で網羅性や実施自由は判断できない。[J-PlatPat](https://www.j-platpat.inpit.go.jp/) · [European Patent Register](https://register.epo.org/)
