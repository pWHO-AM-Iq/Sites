"""Build a bilingual, dependency-free static Site. Run with python main/build.py."""
import json
from html import escape
from pathlib import Path
from bench_view import render as render_benchmarks
from catalog import entries, localized

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
CFG = json.loads((ROOT / "config.json").read_text("utf-8"))
DETAILS = json.loads((DIST / "data/details.json").read_text("utf-8"))
BRAND, MODEL = (escape(CFG[k]) for k in ("brand", "product"))
PAGES = {
    "": ("ホーム", "Home"), "models": ("モデル", "Models"),
    "models/detector": ("モデル詳細", "Model details"), "benchmarks": ("性能評価", "Benchmarks"),
    "training": ("受託学習", "Custom Training"), "research": ("研究", "Research"),
    "docs": ("ドキュメント", "Documentation"), "store": ("ストア", "Store"), "legal": ("法務情報", "Legal")
}
ADDITIONS = entries(CFG)
for route, item in ADDITIONS.items():
    PAGES[route] = (localized(item,"name",False), localized(item,"name",True))
ENGLISH = False
def tr(ja, en):
    return en if ENGLISH else ja
def url(page, en=None):
    if en is None:
        en = ENGLISH
    if page not in PAGES:
        raise ValueError(page)
    return ("/en/" if en else "/") + (page + "/" if page else "")
def name(page):
    return tr(*PAGES[page])
def a(page, label=None, cls=""):
    return f'<a class="{cls}" href="{url(page)}">{label or name(page)}</a>'
def section(title, body):
    return f'<section class="entry"><h2>{title}</h2><div>{body}</div></section>'
def para(s):
    return f'<p>{s}</p>'
def table(rows, heads=None):
    th = "" if not heads else "<thead><tr>" + "".join(f'<th scope="col">{v}</th>' for v in heads) + "</tr></thead>"
    trs = "".join("<tr>" + f'<th scope="row">{r[0]}</th>' + "".join(f'<td>{v}</td>' for v in r[1:]) + "</tr>" for r in rows)
    return f'<div class="tablewrap" tabindex="0"><table>{th}<tbody>{trs}</tbody></table></div>'
def links(*pages):
    return '<div class="linklist">' + "".join(a(p, name(p) + " ↗") for p in pages) + "</div>"
def extra_products(kind=None):
    selected = [route for route, item in ADDITIONS.items() if kind is None or item["kind"] == kind]
    return links(*selected) if selected else ""
def button(page, txt, primary=False):
    return a(page, txt + ' <span aria-hidden="true">↗</span>', "button primary" if primary else "button")
def grid(items):
    return '<div class="cards">' + "".join(f'<div class="card"><h3>{x}</h3><p>{y}</p></div>' for x,y in items) + "</div>"
def figure():
    result = json.loads((DIST / "data/example.json").read_text("utf-8"))
    labels = {"car":"車","bicycle":"自転車","person":"人物","backpack":"リュック"}
    # All boxes are present in the full results. Limit labels on the image to reduce overlap.
    boxes = []
    for i, d in enumerate(result["detections"]):
        x,y,r,b = d["xyxy"]
        label = escape(d["label"] if ENGLISH else labels.get(d["label"],d["label"]))
        tag = f'<i>{label} {d["score"]:.2f}</i>' if i < 3 else ""
        boxes.append(f'<span class="bbox" style="left:{100*x:.3f}%;top:{100*y:.3f}%;width:{100*(r-x):.3f}%;height:{100*(b-y):.3f}%">{tag}</span>')
    n = len(boxes)
    return f'''<figure class="demo"><div class="demo-head"><span class="lamp"></span> INFERENCE VIEW <span class="variant">ONNX / NMS-FREE</span></div>
    <div class="scene"><img src="/assets/crossing.jpg" width="{result["image_size"][0]}" height="{result["image_size"][1]}" alt="{tr("生成した横断歩道の画像。青い車、自転車、歩行者が写っている。","Generated crossing scene with a blue car, a bicycle and pedestrians.")}"><div class="boxes" aria-hidden="true">{"".join(boxes)}</div></div>
    <figcaption><span>{tr("生成画像での実推論","Inference on a generated scene")} · {n} {tr("件","detections")}</span><button type="button" data-overlay aria-pressed="true">{tr("検出枠","Boxes")} <b>ON</b></button></figcaption></figure>'''
def nav(page):
    top = ("models","benchmarks","research","docs")
    other = ("training","store","legal")
    primary = "".join(f'<a href="{url(p)}"'+ (' aria-current="page"' if p == page else '')+f'>{name(p)}</a>' for p in top)
    menu = "".join(a(p) for p in (*top,*other))
    switch = f'<div class="language"><a lang="ja" hreflang="ja" href="{url(page,False)}" class="{"" if ENGLISH else "selected"}">JP</a><span>/</span><a lang="en" hreflang="en" href="{url(page,True)}" class="{"selected" if ENGLISH else ""}">EN</a></div>'
    return f'''<header><div class="wrap header-inner"><a class="brand" href="{url("")}"><span class="mark" aria-hidden="true"><i></i><i></i><i></i></span>{BRAND}</a><nav class="desktop">{primary}</nav>{switch}<details class="menu"><summary aria-label="{tr("メニュー","Menu")}"><span></span><span></span></summary><nav>{menu}</nav></details></div></header>'''
def footer():
    return f'<footer><div class="wrap footergrid"><div class="brand"><span class="mark" aria-hidden="true"><i></i><i></i><i></i></span>{BRAND}</div><nav>{"".join(a(p) for p in PAGES if p not in ("","models/detector"))}</nav></div></footer>'
def home():
    free = DETAILS["models"]["nms_free"]
    quality_link = a("benchmarks", f'{free["scores"]["ap"]:.2%} <em>AP ↗</em>')
    return f'''<section class="hero"><div class="wrap herogrid"><div class="herocopy"><div class="topic"><span></span> COMPUTER VISION</div><h1>{tr("物体検出の<br>研究と実装。","Object detection,<br>from the lab.")}</h1><p>{tr("COCO 2017で学習した検出モデルを、実際の推論結果、5,000枚の精度評価、実装資料から調べられます。","Explore a detector trained on COCO 2017, its inference outputs, evaluation protocols and implementation notes.")}</p><div class="actions">{button("models/detector",tr("モデルを見る","Explore the model"),True)}{button("benchmarks",tr("性能評価","Benchmarks"))}</div></div>{figure()}</div></section>
    <div class="band"><div class="wrap bandgrid"><div><small>NMS-FREE · VAL2017</small><strong>{quality_link}</strong></div><div><small>INPUT</small><strong>640 × 640 <em>px</em></strong></div><div><small>CLASSES</small><strong>{DETAILS["dataset"]["categoryCount"]}</strong></div><div><small>MODEL FORMATS</small><strong>PyTorch <em>/</em> ONNX</strong></div></div></div>
    <section class="wrap homebody"><div class="sechead"><div class="topic dark">MODEL LIBRARY</div><h2>{tr("モデルライブラリ","Model library")}</h2></div><a href="{url("models/detector")}" class="feature"><div class="iconmodel" aria-hidden="true"><i></i><i></i><i></i></div><div><small>OBJECT DETECTION</small><h3>{MODEL}</h3><p>{tr("NMS版とNMSフリー版の仕様、実際の出力、推論手順。","Specifications, sample outputs and inference instructions for the NMS and NMS-free variants.")}</p></div><b aria-hidden="true">↗</b></a><div class="sechead"><div class="topic dark">RESEARCH NOTES</div><h2>{tr("研究・技術情報","Research and engineering")}</h2></div>{links("research","benchmarks","training")}</section>'''
def specs():
    models = DETAILS["models"]
    spec = models["nms_free"]
    shape = spec["shapes"]
    return table([
        (tr("タスク・学習データ","Task and training data"),tr("物体検出 · COCO 2017","Object detection · COCO 2017")),
        (tr("入力","Input"),f'RGB · float32 · NCHW · {shape["images"]}'),
        (tr("クラス・出力候補数","Classes and candidates"),f'{shape["scores"][2]} / {shape["scores"][1]:,}'),
        (tr("ONNX出力","ONNX outputs"),f'boxes {shape["boxes"]} · scores {shape["scores"]}'),
        (tr("チェックポイント","Checkpoints"),f'NMS · epoch {models["nms"]["epoch"]} / NMS-free · epoch {spec["epoch"]}'),
        (tr("推論パラメータ数（各方式）","Active inference parameters per variant"),f'{spec["inference_parameter_count"]:,}'),
        (tr("学習時の総数 / 学習可能パラメータ","Total / trainable parameters during training"),f'{spec["training_parameter_count"]:,} / {spec["trainable_parameter_count"]:,}'),
        (tr("計算量（Conv・MatMulのみ）","Graph compute (Conv and MatMul only)"),f'{spec["macs"]/1e9:.3f} GMACs / {spec["flops"]/1e9:.3f} GFLOPs'),
        (tr("ファイル形式","File formats"),"PyTorch (.pt) / ONNX (.onnx)"),
    ])
def body(page):
    if page in ADDITIONS:
        entry = ADDITIONS[page]
        return section(name(page),para(localized(entry,"description",ENGLISH)))+section(tr("関連情報","Related information"),links("models","store","legal"))
    if page == "models":
        return section(tr("物体検出","Object detection"),f'<h3>{MODEL}</h3>'+para(tr("画像からCOCOの80クラスを検出し、位置と信頼度を返します。用途に応じて後処理の異なる2方式を比較できます。","Detects 80 COCO classes and returns locations and scores. Two variants expose different post-processing choices."))+button("models/detector",tr("仕様と推論例","Specifications and sample"),True))+section(tr("製品・サービス","Products and services"),links("store","training"))
    if page == "models/detector":
        data = json.loads((DIST / "data/example.json").read_text("utf-8"))
        results = [(escape(d["label"]),f'{d["score"]:.4f}',", ".join(str(round(x * data["image_size"][i%2])) for i,x in enumerate(d["xyxy"]))) for d in data["detections"]]
        return section(tr("モデル仕様","Specifications"),specs())+section(tr("推論例","Inference sample"),figure()+para(tr("NMSフリー版をONNX Runtime CPUで実行しました。入力は作例用の生成画像、表示閾値は0.50です。下表は元画像上のピクセル座標です。","Executed with the NMS-free ONNX model on CPU. The input is a generated scene. Score threshold 0.50. Coordinates below are in original image pixels."))+table(results,("Class","Score","x1, y1, x2, y2")))+section(tr("動作環境・提供状況","Runtime and availability"),para(tr("動作環境・性能は試験結果を参照してください。ライセンスと購入先は販売前の確認中です。","See the benchmark record for verified runtimes. License and sales channels are under review."))+links("benchmarks","docs","store","legal"))
    if page == "research":
        return section(tr("開発経緯","Development history"),para(tr("学習記録では、外部の学習済みウェイトを使わず、COCO 2017でランダム初期化から300エポック学習しています。モデル構成はYOLO26mを参照し、元のモデル定義と学習コードにはUltralyticsの実装を翻案した部分とAGPL-3.0の通知があります。","The training record documents 300 epochs from random initialization on COCO 2017, without external pretrained weights. The architecture follows YOLO26m. The original model and training source identify portions adapted from the Ultralytics implementation under AGPL-3.0.")))+section(tr("2つの検出方式","Two detection paths"),para(tr("NMS版はクラスとスコアを選び、重なる矩形をIoUで整理します。NMSフリー版は候補位置とクラスをスコアで2段階に絞ります。両モデルを同じデータで測ることで、後処理を含む動作を比較できます。","The NMS variant suppresses overlapping boxes by intersection over union. The NMS-free variant ranks candidate positions and class scores in two stages. Testing both on the same data makes their end-to-end behaviour observable.")))+section(tr("論文ノート · YOLO","Paper note · YOLO"),para(tr("原論文は画像全体を一度のネットワーク評価に入力し、位置とクラスを同時に予測する枠組みを示しました。近年の実装を比較する際には、候補選択と後処理の時間も測る必要があります。","The original YOLO paper predicts locations and classes from the whole image in one network evaluation. For a comparison of current implementations, candidate selection and post-processing also belong in the timing protocol."))+'<p><a href="https://arxiv.org/abs/1506.02640">Redmon et al. · You Only Look Once ↗</a></p>')+section(tr("実験結果と考察","Experiments and interpretation"),para(tr("train2017の116,287枚で学習し、別の2,000枚でエポックを選択しました。最終評価にはval2017の5,000枚を使用しています。学習時の最終評価と配布用ONNXの再評価では、画像リサイズ・精度設定・NMS候補の選び方が異なります。","Training used 116,287 train2017 images. A separate 2,000-image development split was used for epoch selection. Final evaluation used the 5,000 val2017 images. The historical evaluation and the distribution ONNX evaluation differ in resize rules, numerical precision and candidate selection for NMS."))+links("benchmarks","models/detector"))
    if page == "training":
        return section(tr("サービス内容","Service scope"),para(tr("アノテーション済みの顧客データを使ったファインチューニングの提供を準備しています。対象は物体検出です。受付開始日と問い合わせ先は確定後に掲載します。","A fine-tuning service for customer-annotated object detection datasets is in preparation. Intake dates and contact details will follow confirmation.")))+section(tr("受付から納品まで","From intake to delivery"),table([
            (tr("受付条件","Intake"),tr("データの利用権、ラベル定義、クラス対応表、利用目的を確認","Confirm data rights, label definitions, class mapping and intended use")),
            (tr("データ検査","Validation"),tr("COCO JSON / YOLO TXTを想定。画像対応、座標、分割と重複を点検","Planned inputs: COCO JSON / YOLO TXT. Inspect image pairing, coordinates, splits and duplicates.")),
            (tr("学習と評価","Training and evaluation"),tr("合意した学習セットと保留評価セットを分けて結果を記録","Use agreed training and holdout splits, then record results.")),
            (tr("納品予定","Proposed delivery"),tr("ウェイト、検証済みのONNX、評価レポート、導入手順","Weights, validated ONNX, evaluation report and setup guide."))
        ]))+section(tr("価格とデータの取り扱い","Pricing and data handling"),para(tr("料金・上限・納期は、検査工数、計算時間、納品形式に基づく個別見積もりを予定しています。達成精度の保証はありません。データの利用目的、保管場所、アクセス権、削除期限、秘密保持を受領前に合意します。","Pricing, limits and lead time will be quoted from validation effort, compute time and deliverables. No accuracy level is guaranteed. Purpose, storage, access, retention, deletion and confidentiality must be agreed before transfer.")))
    if page == "docs":
        return section(tr("導入と推論","Installation and inference"),para(tr("ONNX版のCPU推論ではPython環境にonnxruntime、numpy、pillowを用意します。配布候補のONNXを取得済みの場合の最小例です。画像をモデル入力へ変換した後、2つの出力を取得します。","For CPU inference, prepare Python with onnxruntime, numpy and pillow. This minimal example assumes you already have a candidate ONNX file. Pass the preprocessed image tensor to obtain both outputs."))+'''<pre><code>python -m venv .venv
# Activate the environment, then:
python -m pip install onnxruntime numpy pillow</code></pre>
<pre><code>import onnxruntime as ort
import numpy as np
from PIL import Image

image = Image.open("input.jpg").convert("RGB")
w, h = image.size
ratio = min(640 / w, 640 / h)
nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
canvas = Image.new("RGB", (640, 640), (114, 114, 114))
canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR),
             ((640 - nw) // 2, (640 - nh) // 2))
tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255

session = ort.InferenceSession(
    "custom_detector_nms_free.onnx",
    providers=["CPUExecutionProvider"],
)
boxes, scores = session.run(["boxes", "scores"], {"images": tensor})</code></pre>''')+section(tr("入力と出力","Input and output"),specs()+para(tr("RGB画像を縦横比を保って縮尺変更し、RGB 114の余白で中央に配置します。255で割ってNCHWに変換します。出力矩形から余白を引き、各軸の実リサイズ倍率で元画像に戻します。ONNXのscoresにはsigmoidが適用済みです。","Resize in RGB preserving aspect ratio, centre-pad with RGB 114, divide by 255 and transpose to NCHW. Subtract padding from boxes and divide by each actual resize factor. ONNX scores already include sigmoid.")))+section(tr("ファインチューニングとFAQ","Fine-tuning and FAQ"),para(tr("元の学習コードはCOCO形式の注釈とクラス対応を読み込みます。追加学習では、対象ラベルを対応付け、学習・検証セットを分割し、出力ヘッドと初期化元を設定してから、保持した評価セットで確認します。受入上限と学習レシピは提供前の検証対象です。","The original trainer uses COCO annotations and a class mapping. Fine-tuning requires mapping target labels, making training and validation splits, configuring the detection head and initial weights, then checking a held-out set. Intake limits and the fine-tuning recipe are under pre-release validation."))+'<details class="faq"><summary>'+tr("GPUやFP16を利用できますか？","Can I use GPU or FP16?")+'</summary>'+para(tr("プロバイダー、精度、実際のデバイスを性能評価ページで照合してください。","Check the provider, precision and actual device listed in Benchmarks."))+'</details>'+links("benchmarks","training"))
    if page == "store":
        return section(tr("提供予定の製品","Planned products"),grid([
            ("Standard",tr("学習済みウェイト、推論コード、ONNX、導入ガイド。","Trained weights, inference code, ONNX and setup guide.")),
            ("Developer",tr("Standardに加え、学習コード、設定、ファインチューニング資料。","Standard plus training code, configuration and fine-tuning documentation.")),
            ("Commercial",tr("契約で定める利用範囲と再配布条件を個別に確認。","Usage scope and redistribution terms to be agreed by contract.")),
            (tr("技術教材","Learning materials"),tr("前処理、推論結果の読み方、再現可能な評価の教材を計画中。","Planned guides to preprocessing, reading outputs and reproducible evaluation."))
        ]))+section(tr("購入・問い合わせ","Purchasing and enquiries"),para(tr("販売準備中です。価格、販売条件、外部購入リンクは承認後に掲載します。受託学習の内容はサービスページで確認できます。","Sales are in preparation. Prices, terms and official purchase links will be added after approval. See the service page for custom training."))+links("training","legal"))
    if page == "legal":
        return section(tr("利用規約とライセンス","Terms and licensing"),para(tr("元の実装の一部にはUltralyticsのAGPL-3.0通知があります。ソース、ウェイト、ONNXごとに提供条件と再配布条件を確認しています。製品の使用許諾、サポート、免責は販売前に確定します。","Portions of the original implementation carry an Ultralytics AGPL-3.0 notice. Distribution rights for source code, weights and ONNX artifacts are under review. Product licensing, support and liability terms will be finalized before sales.")))+section(tr("プライバシー","Privacy"),para(tr("本サイトのアプリには登録、決済、データのアップロード機能はありません。ホスティングと将来の販売先に関する処理、運営者、問い合わせ手段、データの保持期間は公開前の確認事項です。","The application has no account registration, payment or dataset upload feature. Hosting and future sales-platform processing, controller identity, contact details and retention are pre-release review items.")))+section(tr("特定商取引法に基づく表示","Japanese commercial disclosures"),para(tr("販売方式に応じ、事業者の表示、支払方法、対価・追加費用、提供時期、返品・キャンセル条件を整備します。確認が完了するまで注文は受け付けていません。","Seller information, payment methods, pricing and additional fees, delivery timing, cancellation and return terms will be completed for the chosen sales method. Orders are not open.")))
    raise ValueError(page)
SUB = {
    "models":("提供を準備している検出モデル。","Explore the detector in development."),
    "models/detector":("モデル仕様とサンプル出力。","Model specifications and sample outputs."),
    "benchmarks":("数値を、測定条件とともに。","Results alongside the conditions that produced them."),
    "training":("顧客データに合わせた物体検出モデルの調整。","Adapt an object detector to an annotated dataset."),
    "research":("論文から実装、評価までの技術ノート。","Technical notes on papers, implementation and evaluation."),
    "docs":("入力の準備から推論結果の読み取りまで。","From preparing the input to interpreting detections."),
    "store":("配布パッケージと提供状況。","Distribution packages and their availability."),
    "legal":("公開・販売前の確認事項。","Disclosures under review before release.")
}
def build(en):
    global ENGLISH
    ENGLISH = en
    for p in PAGES:
        title = MODEL if p == "models/detector" else name(p)
        page_body = render_benchmarks(DIST, tr, table, section, para, MODEL) if p == "benchmarks" else (body(p) if p else "")
        if p in ("store","models"):
            page_body += extra_products("model" if p == "models" else None)
        if p:
            description = localized(ADDITIONS[p],"summary",en) if p in ADDITIONS else tr(*SUB[p])
            content = f'<section class="pagehead"><div class="wrap"><div class="topic"><span></span>{PAGES[p][1].upper()}</div><h1>{title}</h1><p>{description}</p></div></section><div class="wrap entries">{page_body}</div>'
        else:
            content = home()
            if ADDITIONS:
                content += f'<section class="wrap">{extra_products()}</section>'
        lang = "en" if en else "ja"
        html = f'''<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><title>{title} — {BRAND}</title><link rel="stylesheet" href="/assets/site.css"><script src="/assets/site.js" defer></script></head><body><a class="skip" href="#content">{tr("本文へ","Skip to content")}</a>{nav(p)}<main id="content">{content}</main>{footer()}</body></html>'''
        folder = DIST / url(p).strip("/")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "index.html").write_text(html, encoding="utf-8")
if __name__ == "__main__":
    for english in (False, True):
        build(english)
