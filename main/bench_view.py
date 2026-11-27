"""Render the public measurement allowlist as accessible HTML."""
from html import escape
import json

def number(value, precision=2):
    return "—" if value is None else f"{value:,.{precision}f}"

def render(dist, tr, table, section, para, product):
    data = json.loads((dist / "data/evaluation.json").read_text("utf-8-sig"))
    local = [r for r in data["models"] if r["scope"] == "official_validation"]
    official = [r for r in data["models"] if r["scope"] == "official_reference"]
    rows = [(product + (" NMS-free" if r["id"] == "nms_free" else " NMS"),
             number(r["ap50_95"]), number(r["ap50"]), number(r["params"], 0)) for r in local]
    result = section(tr("手元で測った検出精度","Locally measured detection accuracy"),
                     table(rows,("Model","AP50–95 (%)","AP50 (%)",tr("推論パラメータ","Active parameters")))+
                     para(tr("COCO val2017 · bbox · 640 × 640 · バッチ1。2つのチェックポイントは異なるエポックのものです。","COCO val2017 · bbox · 640 × 640 · batch 1. The two checkpoints come from different epochs.")))
    controls = [r for r in data["models"] if r["scope"] == "local_control"]
    if controls:
        control_rows = [(escape(r["label"]),number(r["ap50_95"]),number(r["ap50"])) for r in controls]
        result += section(tr("同じ5,000枚での対照実験","Independent control on the same 5,000 images"),
                          table(control_rows,("Model","AP50–95 (%)","AP50 (%)"))+
                          para(tr("公式ウェイトを独立環境でONNXに変換し、同じ入力と評価コードで測定しました。YOLO26m E2Eの候補選択はグラフ内にあり、学習条件も各モデルで異なります。","The official weights were exported in an isolated environment. Images, preprocessing and scoring matched the product evaluation. YOLO26m E2E selects candidates inside the graph and training histories differ.")))
    detail = json.loads((dist / "data/details.json").read_text("utf-8"))
    export_rows = []
    for mode, record in detail["models"].items():
        torch_ap, onnx_ap = record["matched_torch_ap"],record["scores"]["ap"]
        export_rows.append((escape(mode),number(100*record["historical_final_ap"],2),number(100*torch_ap,6),number(100*onnx_ap,6),f'{100*(onnx_ap-torch_ap):+.6f}'))
    result += section(tr("学習時の記録とONNX変換の検証","Historical results and ONNX conversion"),
                      table(export_rows,("Model",tr("学習時 AP (%)","Historical AP (%)"),"FP32 PT (%)","FP32 ONNX (%)","ONNX − PT (pp)"))+
                      para(tr("学習時の最終評価はRTX 5090 / BF16で、リサイズ規則とNMS候補選択も異なります。右の3列は同じ前後処理、同じ画像、同じ方式ごとのGPU、FP32、TF32無効で再評価した値です。","The historical final run used BF16 on an RTX 5090, different resize rules and different NMS candidate selection. The three rightmost columns use matched preprocessing and postprocessing, identical images and the same GPU per variant, in FP32 with TF32 disabled.")))
    ops = []
    for mode, record in detail["models"].items():
        op = next(row for row in record["operating_points"] if row["threshold"] == 0.35)
        ops.append((escape(mode), *(number(100 * op[k]) for k in ("precision","recall","f1"))))
    result += section(tr("評価条件と固定閾値での結果","Protocol and fixed-threshold results"),
                      para(tr("画像全体で最大300候補、スコア下限0.001。COCOevalはIoU 0.50–0.95、101点補間、maxDets 1/10/100で評価しました。下表はIoU 0.50、スコア > 0.35、全80クラスのmicro集計です。","Up to 300 candidates per image, score floor 0.001. COCOeval uses IoU 0.50–0.95, 101 recall points and maxDets 1/10/100. Below: micro averages across all 80 classes at IoU 0.50 and score > 0.35."))+
                      table(ops,("Model","Precision (%)","Recall (%)","F1 (%)"))+
                      '<p><a href="/data/coco-summary.csv">AP / AR CSV ↓</a> · <a href="/data/class-ap.csv">Class AP CSV ↓</a> · <a href="/data/operating-points.csv">P / R / F1 CSV ↓</a></p>')
    engines = {"ort-cpu":"ONNX Runtime · CPU FP32","ort-cuda":"ONNX Runtime · RTX 3080 FP32¹","torch-cpu":"PyTorch · CPU FP32","torch-fp32":"PyTorch · RTX 3080 FP32","torch-fp16":"PyTorch · RTX 3080 FP16"}
    timing, stages = [], []
    for r in detail["timings"]:
        engine = escape(engines.get(r["backend"],r["backend"]))
        ms = r["ms"]
        timing.append((r["variant"],engine,number(ms["e2e"]["mean"]),number(ms["e2e"]["p50"]),number(ms["e2e"]["p95"]),number(r["fps"])))
        allocated = (r.get("device_memory") or {}).get("cuda_peak_allocated_bytes")
        stages.append((r["variant"],engine,*(number(ms[k]["mean"]) for k in ("decode","preprocess","forward_with_transfers","postprocess")),number(r["peak_rss_sampled_bytes"]/2**20,1),number(allocated/2**20,1) if allocated else "—"))
    result += section(tr("CPU・GPUでの推論時間","CPU and GPU timing"),
                    table(timing,("Model","Runtime","E2E mean (ms)","P50 (ms)","P95 (ms)","FPS"))+
                    para(tr("Core i9-11900、RTX 3080。バッチ1、4スレッド、ウォームアップ10回、16枚のJPEGを計100回処理。¹ ONNX Runtime CUDAはTF32を許可。GPU計時は結果のホスト転送・同期を含みます。ほかの処理と負荷が重なった区間があり、速度はこの測定時の参考値です。","Core i9-11900 and RTX 3080. Batch 1, four threads, ten warm-ups, 100 runs over 16 JPEGs. ¹ ONNX Runtime CUDA allowed TF32. GPU timings include synchronization and result transfer to the host. Concurrent work affected parts of the measurement window."))+
                    '<details class="expander"><summary>'+tr("処理別の平均時間とメモリ","Mean stage timings and memory")+'</summary>'+table(stages,("Model","Runtime","Decode ms","Prepare ms","Forward ms","Post ms","RSS MiB","Torch allocated MiB"))+para(tr("RSSは各リクエスト終了時のプロセスメモリの最大観測値。Torch allocatedはCUDAアロケータのピーク割当量です。ORTのプロセス別最大VRAMは未測定です。","RSS is the largest sampled host process value at request boundaries. Torch allocated is the peak CUDA allocator allocation. ORT per-process peak VRAM was not measured."))+'</details>')
    selected = {"yolo26m","yolo11m","rt-detr-r18","d-fine-m","rf-detr-n"}
    official_rows = [(f'<a href="{escape(r["source_url"],quote=True)}">{escape(r["label"])} ↗</a>', number(r["ap50_95"],1), number(r["params"]/1e6,1)) for r in official if r["id"] in selected]
    result += section(tr("比較候補の公式公表値","Reference models · vendor reports"),
                      table(official_rows,("Model","AP50–95 (%)","Parameters (M)"))+
                      para(tr("YOLO26m・YOLO11m・RT-DETR-R18・D-FINE-Mは約19–20百万パラメータ。RF-DETR-Nは30.5百万、標準入力384です。各名称から出典と測定設定を確認できます。YOLO26の表は通常APです。","YOLO26m, YOLO11m, RT-DETR-R18 and D-FINE-M have about 19–20 million parameters. RF-DETR-N has 30.5 million and uses 384 px inputs. Each model links to its source protocol. The YOLO26 row is standard AP.")))
    titles = {"accuracy":tr("精度比較","Accuracy"),"latency":tr("推論時間","Latency"),"parameters":tr("パラメータ数","Parameter counts"),"class_ap":tr("クラス別AP","Per-class AP"),"pr_curve":tr("PR曲線","Precision–recall"),"params_vs_ap":tr("パラメータ数と精度","Parameters and accuracy"),"latency_vs_ap":tr("レイテンシと精度","Latency and accuracy")}
    figs = []
    for key, rel in data.get("charts",{}).items():
        path = "/" + rel.lstrip("/")
        label = titles.get(key, key.replace("_", " ").title())
        figs.append(f'<figure><a href="{escape(path)}"><img loading="lazy" alt="{escape(label)}" src="{escape(path)}"></a><figcaption>{label}</figcaption></figure>')
    result += section(tr("グラフ・データ","Plots and data"), '<div class="charts">'+''.join(figs)+'</div><p><a href="/data/evaluation.json">Evaluation JSON ↓</a> · <a href="/data/details.json">Protocol &amp; timings JSON ↓</a></p>')
    return result
