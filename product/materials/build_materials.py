"""Build the bilingual benchmark report, CSVs and public-only chart data."""

import argparse
import csv
import json
from pathlib import Path
import re
from shutil import copyfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "product" / "evaluation"
RUNTIME_REPORT = RESULTS / "runtime" / "measurements.json"
PUBLIC = ROOT / "main" / "dist" / "data"
PRODUCT = json.loads((ROOT / "main" / "config.json").read_text(encoding="utf-8-sig"))["product"]
MODES = {"nms": f"{PRODUCT} NMS", "nms_free": f"{PRODUCT} NMS-free"}
AGGREGATES = ("ap", "ap50", "ap75", "apSmall", "apMedium", "apLarge", "ar1", "ar10", "ar100",
              "arSmall", "arMedium", "arLarge")
CHARTS = {key: f"data/{key.replace('_', '-')}.svg"
          for key in ("accuracy", "latency", "parameters", "class_ap", "params_vs_ap", "latency_vs_ap",
                      "pr_curve")}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "svg.hashsalt": "custom-detector-evaluation-v1",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#334155", "axes.titleweight": "bold",
    "text.color": "#0f172a", "xtick.color": "#334155", "ytick.color": "#334155",
})


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def evidence(record):
    return (
        "## 評価記録 / Evaluation record\n\n"
        f"- 版 / Release · `{record['release']}`\n"
        f"- 結果記録日時 / Results recorded at · `{record['result_file_written_at']}`\n"
        f"- 評価コードのコミット / Evaluation source commit · `{record['evaluation_source_commit']}`\n"
        "- 評価の出所 / Evaluation record · `product/evaluation/run-record.json`\n"
        "- 重みと ONNX の SHA-256 / Artifact hashes · "
        "[Original model README](../../models/custom_detector/README.md#ファイルの照合), "
        "[Artifact audit](../evaluation/artifact-audit.json), and the packaged `SHA256SUMS.txt`.\n"
    )


def render_documents(audit):
    models = audit["models"]
    nms, free = (models[mode] for mode in MODES)
    release = read(HERE / "release.json")
    training = read(RESULTS / "origin-training.json")["origin"]
    split = training["split"]
    footer = "\n\n" + evidence(read(RESULTS / "run-record.json"))
    flop = nms["complexity"]["multiplyAddAsTwoFlops"]
    values = {
        "PRODUCT": PRODUCT, "VERSION": release["version"], "RELEASE_DATE": release["release_date"],
        "ARCHITECTURE": nms["architectureId"], "UPSTREAM_COMMIT": nms["upstreamCommit"],
        "NMS_EPOCH": nms["selectedEpochReadme"], "NMS_FREE_EPOCH": free["selectedEpochReadme"],
        "ACTIVE_PARAMS": f"{nms['inferenceParametersMetadata']:,}",
        "TRAINING_PARAMS": f"{nms['trainingParametersMetadata']:,}",
        "TRAINABLE_PARAMS": f"{training['parameterCounts']['trainable']:,}",
        "INITIALIZERS": f"{nms['onnxInitializerElements']:,}",
        "GMACS": f"{nms['complexity']['macs'] / 1e9:.3f}", "GFLOPS": f"{flop / 1e9:.3f}",
        "OPSET": nms["onnxOpset"], "TRAINING_JA": release["training_initialization"]["ja"],
        "TRAINING_EN": release["training_initialization"]["en"],
        "TRAINING_RUN": training["runId"], "TRAIN_IMAGES": f"{split['training']['imageCount']:,}",
        "SELECTION_IMAGES": f"{split['development']['imageCount']:,}",
        "FINAL_IMAGES": f"{split['final']['imageCount']:,}",
    }
    for prefix, model in (("NMS", nms), ("NMS_FREE", free)):
        for ext in ("onnx", "pt"):
            values[f"{prefix}_{ext.upper()}_BYTES"] = f"{model['files'][ext]['bytes']:,}"
    for filename in ("README.md", "SPEC.md", "MODEL_CARD.md", "CHANGELOG.md"):
        text = (HERE / "templates" / filename).read_text(encoding="utf-8-sig")
        text = re.sub(r"\{\{([A-Z_]+)\}\}", lambda match: str(values[match[1]]), text)
        if filename == "SPEC.md":
            shared, details = text.split("<!-- ja -->")
            ja, en = details.split("<!-- en -->")
            japanese = shared
            translations = {
                "Trainable parameters": "学習可能パラメータ数",
                " / I/O specification": "", " / Field": "", " / Value": "",
                " / task": "", " / object detection": "", "ONNX input": "ONNX 入力",
                "ONNX boxes": "ONNX 出力・矩形", "ONNX scores": "ONNX 出力・スコア",
                "in 640 × 640 input pixels": "（640 × 640 入力上のピクセル）",
                "sigmoid probabilities": "sigmoid 適用済みクラス確率",
                "tensor checkpoints with architecture metadata, original model definition required":
                    "モデル構成メタデータ付きテンソルチェックポイント。元のモデル定義が必要",
                "Architecture ID": "モデル構成 ID", "Active inference parameters": "有効な推論パラメータ数",
                "per variant, checkpoint metadata": "（方式ごと。チェックポイント記載）",
                "Training parameters": "学習時パラメータ数",
                "across the shared network and both heads, checkpoint metadata":
                    "（共有部と両ヘッドの合計。チェックポイント記載）",
                "ONNX initializer elements": "ONNX 初期化テンソルの要素数",
                "each, counted from ONNX graph": "（方式ごと。ONNX グラフで集計）",
                "Approximate ONNX compute": "ONNX 演算量（概算）",
                "(one multiply-add as two FLOPs), counting Conv and MatMul at fixed input size":
                    "（積和演算を2 FLOPとして計算。固定入力で Conv と MatMul を集計）",
                "File size": "ファイルサイズ",
            }
            for phrase, localized in translations.items():
                japanese = japanese.replace(phrase, localized)
            english = shared.replace("入出力仕様 / ", "").replace("項目 / ", "").replace("値 / ", "")
            english = english.replace("種別 / task", "Task").replace("物体検出 / ", "")
            (HERE / "SPEC.ja.md").write_text(japanese.rstrip() + "\n\n" + ja.strip() + footer,
                                            encoding="utf-8", newline="\n")
            (HERE / "SPEC.en.md").write_text(english.rstrip() + "\n\n" + en.strip() + footer,
                                            encoding="utf-8", newline="\n")
            text = shared + ja + en
        (HERE / filename).write_text(text.rstrip() + footer, encoding="utf-8", newline="\n")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percent(value):
    return None if value is None or value < 0 else round(value * 100, 4)


def sanitize_runtime(summary):
    runs = {}
    for name, run in summary["runs"].items():
        metadata = {key: run[key] for key in (
            "mode", "backend", "status", "generated_utc", "batch", "size", "threads", "warmup",
            "model_file", "sha256", "session_init_ms", "rss_bytes_before_load",
            "rss_bytes_peak_at_step_boundaries", "rss_bytes_at_end") if key in run}
        if run["status"] == "completed":
            metadata["hardware"] = {key: run["hardware"][key] for key in ("os", "cpu", "logical_cpus", "python")}
            details = run["backend_details"]
            metadata["backend_details"] = {key: details[key] for key in
                                           ("version", "providers", "transfer_in_forward", "device", "gpu_name",
                                            "cuda_version", "precision", "tf32_allowed") if key in details}
            provider = details.get("provider_options", {}).get("CUDAExecutionProvider", {})
            if provider:
                metadata["backend_details"]["allow_tf32"] = provider.get("use_tf32") == "1"
            steady = run["steady_state"]
            metadata["steady_state"] = {key: steady[key] for key in
                                        ("iterations", "ms", "fps", "repeat_output_mismatches", "unique_inputs")}
            metadata["compatibility"] = {
                case: {key: result[key] for key in ("status", "detections", "size", "e2e_ms",
                                                   "expected_frames", "processed_frames", "performance")
                       if key in result}
                for case, result in run.get("compatibility", {}).items()
            }
            memory = run.get("device_memory") or {}
            metadata["device_memory"] = {key: memory[key] for key in
                                         ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes") if key in memory}
            parity = run.get("raw_onnx_cpu_parity")
            if parity:
                metadata["raw_onnx_cpu_parity"] = {key: parity[key] for key in (
                    "images", "reference_sha256", "box_coordinate_abs_delta_model_pixels",
                    "class_probability_abs_delta") if key in parity}
        runs[name] = metadata
    return {
        "schema_version": 1, "runs": runs, "cpu_cuda_parity": summary["cpu_cuda_parity"],
        "pytorch_vs_onnx_cpu_parity": summary.get("pytorch_vs_onnx_cpu_parity", {}),
        "hardware_notes": {"cpu": "Intel Core i9-11900", "cuda_gpu": "GeForce RTX 3080"},
        "conditions": {
            "input": "16 COCO val2017 JPEG images, batch 1, 640 x 640",
            "scope": "JPEG decode, letterbox, forward including host/device transfers, and postprocessing",
            "memory": "Host process RSS sampled at request boundaries. PyTorch CUDA allocator peaks are separate from process VRAM. ORT GPU VRAM peak was not measured.",
            "parity": "One-to-one matching by class and IoU >= 0.50 on the same 16 JPEG images",
            "limitations": ["Shared CPU resources may affect timing.", "ONNX CUDA uses FP32 with TF32 allowed.",
                            "TensorRT provider dependencies were unavailable."],
        },
    }


def runtime_rows(runs):
    rows = []
    for name, run in sorted(runs.items()):
        if run.get("mode") not in MODES or not run.get("backend"):
            continue
        steady = run.get("steady_state", {})
        stages = steady.get("ms", {})
        e2e = stages.get("e2e", {})
        hardware = run.get("hardware", {})
        details = run.get("backend_details", {})
        compatibility = run.get("compatibility", {})
        video = compatibility.get("video", {})
        memory = run.get("device_memory") or {}
        rows.append({
            "mode": run["mode"], "backend": run["backend"], "status": run.get("status"),
            "provenance": name, "cpu": hardware.get("cpu"),
            "logical_cpus": hardware.get("logical_cpus"), "python": hardware.get("python"),
            "device": details.get("device"), "batch": run.get("batch"),
            "threads": run.get("threads"), "warmup": run.get("warmup"),
            "iterations": steady.get("iterations"), "unique_inputs": steady.get("unique_inputs"),
            "e2e_mean_ms": e2e.get("mean"), "e2e_p50_ms": e2e.get("p50"), "e2e_p95_ms": e2e.get("p95"),
            "forward_p50_ms": stages.get("forward_with_transfers", {}).get("p50"),
            "fps": steady.get("fps"), "session_init_ms": run.get("session_init_ms"),
            "rss_peak_bytes": run.get("rss_bytes_peak_at_step_boundaries"),
            "repeat_output_mismatches": steady.get("repeat_output_mismatches"),
            "os": hardware.get("os"), "runtime_version": details.get("version"),
            "gpu_name": details.get("gpu_name"), "precision": details.get("precision"),
            "cuda_peak_allocated_bytes": memory.get("cuda_peak_allocated_bytes"),
            "cuda_peak_reserved_bytes": memory.get("cuda_peak_reserved_bytes"),
            "video_frames": video.get("processed_frames"), "video_status": video.get("status"),
            "empty_bytes_status": compatibility.get("empty_bytes", {}).get("status"),
        })
    return rows


def present_rows(quality, baseline, latency, audit):
    models = quality.get("models", {})
    local, classes = [], []
    for mode, label in MODES.items():
        model = models.get(mode, {})
        accuracy = model.get("aggregate") or {}
        metadata = audit.get("models", {}).get(mode, {})
        cpu = next((x for x in latency if x["mode"] == mode and x["backend"] == "ort-cpu"
                    and x["status"] == "completed"), {})
        local.append({
            "id": mode, "label": label, "scope": "official_validation",
            "status": model.get("status", "unavailable"),
            "ap50_95": percent(accuracy.get("ap")), "ap50": percent(accuracy.get("ap50")),
            "params": metadata.get("inferenceParametersMetadata"), "latency_ms": cpu.get("e2e_p50_ms"),
            "latency_scope": "local_smoke" if cpu else None,
            "latency_provenance": cpu.get("provenance"),
            "latency_backend": cpu.get("backend"),
            "provenance": "COCO 2017 val2017 / local ONNX, COCOeval" if accuracy
                          else "COCO 2017 val2017 / pending",
            "params_provenance": "active inference / checkpoint metadata",
            "source_url": None,
        })
        for category in model.get("classes", []):
            classes.append({
                "model_id": mode, "category_id": category["id"], "category": category["name"],
                "ground_truth": category.get("groundTruth"), "ap50_95": percent(category.get("ap")),
                "ap50": percent(category.get("ap50")), "ap75": percent(category.get("ap75")),
                "ar100_pct": percent(category.get("ar100")), "scope": "official_validation",
                "status": model["status"],
            })
    official = [{
        "id": re.sub(r"[^a-z0-9]+", "-", row["model"].lower()).strip("-"),
        "label": row["model"], "scope": "official_reference", "status": "reported",
        "ap50_95": row["ap50_95_pct"], "ap50": None,
        "params": round(row["params_m"] * 1_000_000) if row["params_m"] is not None else None,
        "latency_ms": None, "latency_scope": None, "latency_provenance": None,
        "latency_backend": None, "provenance": row["methodology_id"],
        "params_provenance": "vendor report / rounded", "source_url": row["source_url"],
    } for row in baseline.get("rows", [])]
    return local, official, classes


def local_control(quality, control):
    result = next((row for row in control.get("coco_results", []) if row["model"] == "YOLO26m E2E"), None)
    if result is None:
        return [], {}
    dataset = quality["dataset"]
    if (control["dataset"]["annotationsSha256"] != dataset["annotationsSha256"]
            or result["images"] != dataset["imageCount"]):
        raise ValueError("Local control must use the same complete COCO validation set")
    metrics = result["coco_metrics_0_to_1"]
    row = {
        "id": "yolo26m-local-control", "label": result["model"], "scope": "local_control",
        "status": "measured", "ap50_95": percent(metrics["ap"]), "ap50": percent(metrics["ap50"]),
        "params": None, "latency_ms": None, "latency_scope": None, "latency_provenance": None,
        "latency_backend": None, "provenance": f"COCO 2017 val2017 / local ONNX, COCOeval / {result['device']}",
        "params_provenance": None, "source_url": control.get("cpu_control", {}).get("original_asset"),
    }
    return [row], metrics


def save(fig, key):
    fig.savefig(PUBLIC / Path(CHARTS[key]).name, metadata={"Date": None}, bbox_inches="tight",
                facecolor="#f8fafc", pad_inches=0.28)
    plt.close(fig)


def bars(key, title, xlabel, rows, color="#2563eb", xmax=None):
    plotted = [(label, value) for label, value in rows if value is not None]
    fig, ax = plt.subplots(figsize=(7.3, max(2.35, 1.35 + 0.45 * len(plotted))))
    fig.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_title(title, loc="left", pad=13)
    if plotted:
        labels, values = zip(*plotted)
        bars = ax.barh(labels, values, height=0.55, color=color)
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.2f", padding=5)
        ax.set_xlim(0, xmax or max(values) * 1.18)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
    else:
        ax.text(0.02, 0.6, "Measurement unavailable", transform=ax.transAxes, fontsize=13)
        ax.set_axis_off()
    save(fig, key)


def class_chart(classes):
    categories = {item["category_id"]: item["category"] for item in classes if item["ap50_95"] is not None}
    if not categories:
        bars("class_ap", "Per-class COCO AP50–95", "AP (%)", [])
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(categories) * 0.125 + 1.5)))
    fig.set_facecolor("#f8fafc")
    fig.suptitle("Per-class AP50–95 | COCO val2017", x=0.09, ha="left", fontweight="bold")
    lookup = {(item["model_id"], item["category_id"]): item["ap50_95"] for item in classes}
    for ax, ids in zip(axes, [sorted(categories)[:len(categories) // 2], sorted(categories)[len(categories) // 2:]]):
        ax.set_facecolor("#f8fafc")
        for mode, shift, color in [("nms", -0.12, "#2563eb"), ("nms_free", 0.12, "#0f766e")]:
            points = [(i + shift, lookup.get((mode, category))) for i, category in enumerate(ids)]
            ax.scatter([x for _, x in points], [y for y, _ in points], s=15,
                       color=color, label=MODES[mode].removeprefix(PRODUCT + " "))
        ax.set_yticks(range(len(ids)), [categories[category] for category in ids], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("AP (%)")
        ax.grid(axis="x", alpha=0.22)
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.subplots_adjust(wspace=0.37, top=0.91)
    save(fig, "class_ap")


def scatter(key, title, panels):
    columns = len(panels) if len(panels) <= 2 else 2
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(5.75 * columns, 3.85 * rows))
    fig.set_facecolor("#f8fafc")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_axis_off()
    for ax, (label, xlabel, points) in zip(axes.flat, panels):
        ax.set_facecolor("#f8fafc")
        ax.set_title(label, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("AP50–95 (%)")
        ax.grid(alpha=0.2)
        grouped = {}
        for group, name, x, y in points:
            if x is not None and y is not None:
                grouped.setdefault(group, []).append((name, x, y))
        if not grouped:
            ax.text(0.5, 0.5, "Measurement unavailable", transform=ax.transAxes, ha="center")
            continue
        for group, values in grouped.items():
            names, xs, ys = zip(*values)
            ax.scatter(xs, ys, label=group, s=53, alpha=0.86)
            for name, x, y in zip(names, xs, ys):
                if name:
                    ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.margins(x=0.24, y=0.25)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    save(fig, key)


def joint_charts(local, official, runtime):
    local_params = [(x["id"], "", x["params"] / 1e6 if x["params"] is not None else None, x["ap50_95"])
                    for x in local]
    vendor_params = [(x["methodology_id"], "", x["params_m"], x["ap50_95_pct"]) for x in official]
    scatter("params_vs_ap", "Parameter count vs COCO AP50–95", [
        ("Local COCO validation", "Active inference parameters (M)", local_params),
        ("Vendor publications", "Reported parameters (M)", vendor_params),
    ])
    accuracy = {x["id"]: x["ap50_95"] for x in local}
    backends = list(dict.fromkeys(x["backend"] for x in runtime if x["status"] == "completed"))
    local_runs = [("Local E2E · " + backend, "E2E median (ms)",
                   [(x["mode"], "", x["e2e_p50_ms"], accuracy[x["mode"]]) for x in runtime
                    if x["backend"] == backend and x["status"] == "completed"])
                  for backend in backends]
    vendor_runs = [("Published · " + name, "Published latency (ms)",
                    [(x["methodology_id"], "", x[metric], x["ap50_95_pct"]) for x in official
                     if x.get(metric) is not None])
                   for name, metric in (("CPU ONNX", "cpu_onnx_ms"), ("T4", "t4_latency_ms"))]
    scatter("latency_vs_ap", "Inference latency vs COCO AP50–95", local_runs + vendor_runs)


def pr_chart(quality):
    protocol = quality.get("protocol", {})
    recall = protocol.get("recallPoints", [])
    if not recall:
        bars("pr_curve", "Precision vs recall at IoU 0.50", "Recall", [])
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.set_facecolor("#f8fafc")
    fig.suptitle("COCO val2017 precision–recall | IoU 0.50", fontweight="bold")
    for ax, (title, selector) in zip(axes, [("Macro mean · 80 classes", None), ("Person", 1)]):
        ax.set_facecolor("#f8fafc")
        ax.set_title(title, loc="left")
        ax.set(xlabel="Recall", ylabel="Interpolated precision", xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.2)
        for mode, model in quality.get("models", {}).items():
            curves = [item["pr50"] for item in model["classes"] if selector is None or item["id"] == selector]
            values = []
            for column in zip(*curves):
                valid = [value for value in column if value is not None]
                values.append(sum(valid) / len(valid) if valid else float("nan"))
            if values:
                ax.step(recall, values, where="post", linewidth=1.6, label=MODES[mode].removeprefix(PRODUCT + " "))
        ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "pr_curve")


def result_table(rows, column_names):
    header = ["| " + " | ".join(column_names) + " |", "| " + " | ".join(["---"] * len(column_names)) + " |"]
    output = []
    for row in rows:
        output.append("| " + " | ".join("—" if item is None else
                      (f"{item:.2f}" if isinstance(item, float) else str(item)) for item in row) + " |")
    return "\n".join(header + output)


def report(quality, local, controls, control_metrics, refs, runtime):
    dataset = quality.get("dataset", {})
    accuracy_table = result_table(((x["label"], x["status"], x["ap50_95"], x["ap50"], x["params"]) for x in local),
                                  ("Model", "Status", "AP50–95 %", "AP50 %", "Active parameters"))
    coverage = (f"COCO 2017 val2017 · {dataset['imageCount']:,} images · "
                f"{dataset['annotationCount']:,} annotations · {dataset['categoryCount']} classes."
                if dataset else "COCO 2017 val2017 results pending.")
    local_table = result_table(((MODES[x["mode"]], x["backend"], x["status"], x["iterations"],
                                 x["e2e_p50_ms"], x["e2e_p95_ms"], x["fps"]) for x in runtime),
                               ("Model", "Backend", "Status", "Iterations", "P50 ms", "P95 ms", "FPS"))
    compatibility_table = result_table(
        ((MODES[x["mode"]], x["backend"], x["video_frames"], x["video_status"], x["empty_bytes_status"],
          x["repeat_output_mismatches"]) for x in runtime if x["video_status"]),
        ("Model", "Backend", "MP4 frames", "Video", "Empty input", "Repeat mismatches"),
    )
    accuracy_details = result_table(
        ((label, *(percent(quality.get("models", {}).get(mode, {}).get("aggregate", {}).get(metric))
                   for metric in ("ap75", "apSmall", "apMedium", "apLarge", "ar1", "ar10", "ar100",
                                  "arSmall", "arMedium", "arLarge"))) for mode, label in MODES.items()),
        ("Model", "AP75", "AP-S", "AP-M", "AP-L", "AR1", "AR10", "AR100", "AR-S", "AR-M", "AR-L"),
    )
    archived = read(RESULTS / "origin-training.json")["models"]
    archive_table = result_table(
        ((label, percent(archived[mode]["originDevelopmentAp"]), percent(archived[mode]["originFinalAp"]))
         for mode, label in MODES.items()),
        ("Model", "Model selection AP50–95 %", "Final archived AP50–95 %"),
    )
    control_table = ""
    if controls:
        control_table = result_table(
            ((row["label"], row["ap50_95"], row["ap50"], percent(control_metrics["ap75"]),
              percent(control_metrics["ar100"])) for row in controls),
            ("Model", "AP50–95 %", "AP50 %", "AP75 %", "AR100 %"),
        ) + (
            "\n\n同じ COCO val2017 全5,000枚、640 × 640 RGBレターボックス、スコア下限0.001、"
            "最大300候補、COCOeval bbox を使用しました。YOLO26m E2E は RTX 3080 で ONNX Runtime CUDA "
            "を使用し、上位候補の選択はグラフ内で行いました。\n\n"
            "The local control used the same 5,000 COCO val2017 images, 640 × 640 RGB letterbox, "
            "0.001 score floor, 300-candidate limit and COCOeval bbox. YOLO26m E2E ran on an RTX 3080 "
            "with ONNX Runtime CUDA and graph-embedded Top-K. The separate four-image CPU trial timed "
            "graph execution and used a different latency scope from the end-to-end measurements below.\n\n"
            "Repository evidence · `product/comparisons/local-control.json` and `product/comparisons/local-control.md`.\n"
        )
    references = "\n".join(f"- {x['label']} · {x['ap50_95']}% AP50–95. [Official source]({x['source_url']})"
                           for x in refs if x["ap50_95"] is not None)
    template = (HERE / "BENCHMARK.template.md").read_text(encoding="utf-8-sig")
    data = {"COVERAGE": coverage, "ACCURACY": accuracy_table, "ACCURACY_DETAILS": accuracy_details,
            "LOCAL_CONTROL": control_table or "対照結果未入力 / Local control results unavailable.",
            "ARCHIVED": archive_table,
            "RUNTIME": local_table, "COMPATIBILITY": compatibility_table, "REFERENCES": references}
    for token, value in data.items():
        template = template.replace("{{" + token + "}}", value)
    footer = "\n\n" + evidence(read(RESULTS / "run-record.json"))
    (HERE / "BENCHMARK.md").write_text(template.rstrip() + footer, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, default=RESULTS / "quality.json")
    parser.add_argument("--audit", type=Path, default=RESULTS / "artifact-audit.json")
    parser.add_argument("--references", type=Path, default=ROOT / "product/comparisons/official-baselines.json")
    parser.add_argument("--local-control", type=Path, default=ROOT / "product/comparisons/local-control.json")
    parser.add_argument("--runtime", type=Path, nargs="*", help="Raw runs overriding the durable aggregate report")
    parser.add_argument("--refresh-runtime", type=Path, metavar="SUMMARY",
                        help="Refresh the durable aggregate report from a raw summary")
    args = parser.parse_args()
    quality = read(args.quality)
    if args.refresh_runtime:
        sanitized = sanitize_runtime(read(args.refresh_runtime))
        RUNTIME_REPORT.write_text(json.dumps(sanitized, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    runs = ({path.stem: read(path) for path in args.runtime} if args.runtime is not None
            else read(RUNTIME_REPORT).get("runs", {}))
    runtime = runtime_rows(runs)
    baseline = read(args.references)
    audit = read(args.audit)
    if audit:
        render_documents(audit)
    local, refs, classes = present_rows(quality, baseline, runtime, audit)
    controls, control_metrics = local_control(quality, read(args.local_control))
    PUBLIC.mkdir(parents=True, exist_ok=True)
    dataset = quality.get("dataset", {})
    payload = {
        "schema_version": 1, "models": local + controls + refs, "charts": CHARTS,
        "dataset": {key: dataset[key] for key in
                    ("name", "split", "imageCount", "annotationCount", "categoryCount") if key in dataset},
        "runtime": [{key: row[key] for key in ("mode", "backend", "status", "batch", "threads", "warmup",
                                               "iterations", "unique_inputs", "e2e_p50_ms", "e2e_p95_ms", "fps",
                                               "precision", "gpu_name")}
                    for row in runtime],
    }
    (PUBLIC / "evaluation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                                        allow_nan=False) + "\n", encoding="utf-8")
    model_accuracy = [(x["label"], x["ap50_95"]) for x in local]
    control_accuracy = [(f"{x['label']} (local control)", x["ap50_95"]) for x in controls]
    bars("accuracy", "Local measurements | COCO val2017", "AP50–95 (%)",
         model_accuracy + control_accuracy, xmax=100,
         color=["#2563eb"] * sum(value is not None for _, value in model_accuracy)
               + ["#0f766e"] * len(control_accuracy))
    bars("latency", "Local inference | E2E median", "P50 (ms), batch 1",
         [(f"{MODES[x['mode']]} · {x['backend']}", x["e2e_p50_ms"]) for x in runtime
          if x["status"] == "completed"], color="#0f766e")
    bars("parameters", "Active inference parameters", "Millions of parameters (checkpoint metadata)",
         [(x["label"], x["params"] / 1e6) for x in local if x["params"] is not None])
    class_chart(classes)
    joint_charts(local, baseline.get("rows", []), runtime)
    pr_chart(quality)
    write_csv(HERE / "models.csv", local + controls + refs, list(local[0]))
    write_csv(HERE / "class-ap.csv", classes,
              ["model_id", "category_id", "category", "ground_truth", "ap50_95", "ap50", "ap75",
               "ar100_pct", "scope", "status"])
    aggregates = []
    operating_points = []
    for mode, model in quality.get("models", {}).items():
        aggregates.append({"model_id": mode, "scope": "official_validation", "status": model["status"],
                           **{name: percent(model["aggregate"].get(name)) for name in AGGREGATES}})
        operating_points.extend({"model_id": mode, "category_id": None, "category": "micro", **point}
                                for point in model["operatingPoints"])
        for category in model["classes"]:
            operating_points.extend({"model_id": mode, "category_id": category["id"],
                                     "category": category["name"], **point}
                                    for point in category["operatingPoints"])
    aggregates.extend({"model_id": row["id"], "scope": row["scope"], "status": row["status"],
                       **{name: percent(control_metrics.get(name)) for name in AGGREGATES}} for row in controls)
    write_csv(HERE / "coco-summary.csv", aggregates, ["model_id", "scope", "status", *AGGREGATES])
    write_csv(HERE / "operating-points.csv", operating_points,
              ["model_id", "category_id", "category", "threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
    runtime_fields = ("mode backend status provenance cpu logical_cpus python device batch threads warmup "
                      "iterations unique_inputs e2e_mean_ms e2e_p50_ms e2e_p95_ms forward_p50_ms fps "
                      "session_init_ms rss_peak_bytes repeat_output_mismatches os runtime_version "
                      "gpu_name precision cuda_peak_allocated_bytes cuda_peak_reserved_bytes "
                      "video_frames video_status empty_bytes_status")
    write_csv(HERE / "runtime.csv", runtime, runtime_fields.split())
    for filename in ("class-ap.csv", "coco-summary.csv", "operating-points.csv"):
        copyfile(HERE / filename, PUBLIC / filename)
    report(quality, local, controls, control_metrics, refs, runtime)


if __name__ == "__main__":
    main()
