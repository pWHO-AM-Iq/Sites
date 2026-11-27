"""Project the verified local measurement record into public numerical fields."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUALITY = ROOT / "product/evaluation/quality.json"
DATA = ROOT / "main/dist/data"
quality = json.loads(QUALITY.read_text("utf-8"))
details = {"dataset": quality["dataset"], "protocol": quality["protocol"], "models": {}}
origin = json.loads((QUALITY.parent / "origin-training.json").read_text("utf-8"))
runtime = json.loads((QUALITY.parent / "runtime/measurements.json").read_text("utf-8"))
conversion = json.loads((QUALITY.parent / "conversion-accuracy.json").read_text("utf-8"))
details["training"] = {"split": origin["origin"]["split"], "epochs": origin["origin"]["trainingEpochs"],
                       "seed": origin["origin"]["initialization"]["seed"]}
details["record"] = {k:v for k,v in json.loads((QUALITY.parent/"run-record.json").read_text("utf-8")).items()
                     if k in ("release","result_file_written_at","evaluation_source_commit","training_source_commit")}
details["timings"] = []
for run in runtime["runs"].values():
    if run["status"] != "completed":
        continue
    details["timings"].append({"variant":run["mode"], "backend":run["backend"], "ms":run["steady_state"]["ms"],
                               "fps":run["steady_state"]["fps"], "device_memory":run.get("device_memory"),
                               "peak_rss_sampled_bytes":run["rss_bytes_peak_at_step_boundaries"]})
for variant, record in quality["models"].items():
    spec = record["specs"]
    details["models"][variant] = {
        "scores": record["aggregate"], "operating_points": record["operatingPoints"],
        "historical_final_ap": origin["models"][variant]["originFinalAp"],
        "matched_torch_ap": conversion["models"][variant]["torchAggregate"]["ap"],
        "training_parameter_count": spec["trainingParametersMetadata"],
        "trainable_parameter_count": origin["origin"]["parameterCounts"]["trainable"],
        "inference_parameter_count": spec["inferenceParameterElementsVerified"],
        "macs": spec["complexity"]["macs"], "flops": spec["complexity"]["multiplyAddAsTwoFlops"],
        "files": spec["files"], "shapes": spec["onnxShapes"], "opset": spec["onnxOpset"],
        "epoch": spec["selectedEpochReadme"],
    }
DATA.mkdir(exist_ok=True, parents=True)
(DATA / "details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
for name in ["class-ap.csv", "coco-summary.csv", "operating-points.csv"]:
    path = ROOT / "product/materials" / name
    if path.exists():
        # These generated CSVs contain only category/aggregate metrics.
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        with (DATA / name).open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
