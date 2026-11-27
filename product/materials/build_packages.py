"""Build local draft archives and verify every member against its SHA-256."""

import argparse
import csv
from hashlib import file_digest, sha256
import json
import re
from pathlib import Path
from posixpath import dirname, join, normpath
from shutil import copyfileobj
from zipfile import ZipFile, ZIP_STORED, ZipInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ROOT / "models" / "custom_detector"
DEST = ROOT / "private" / "packages"
RELEASE = json.loads((HERE / "release.json").read_text(encoding="utf-8-sig"))
VERSION = RELEASE["version"]

MODEL_HASHES = {
    "custom_detector_nms.pt": "9a6045ec73c577c5aa868e41a2b8baa8ed9f6f0bda2c32abd3cfd106f8e55a4c",
    "custom_detector_nms.onnx": "33dc5f8085888ea27a32fed4731d34ace5a819cf35d525b7adec13bf40cda120",
    "custom_detector_nms_free.pt": "410733f02fec78e9bb4992cea2d67a203023492781036fd8b98bf139249b836d",
    "custom_detector_nms_free.onnx": "34c59778d064ae888e67d9a075eebc9d4980c67f20afe89c1e07b67deee864f7",
}

COMMON = ("README.md", "SPEC.md", "SPEC.ja.md", "SPEC.en.md", "MODEL_CARD.md", "BENCHMARK.md",
          "CHANGELOG.md", "RIGHTS_STATUS.md", "class-ap.csv", "coco-summary.csv", "runtime.csv",
          "operating-points.csv")
INFERENCE = ("detect_onnx.py", "requirements-inference.txt")
ORIGINAL_INFERENCE = "inference/distribution/inference_tools.zip"
TRAINING_FILES = {
    "Start-Training.ps1", "train.py", "prepare_data.py", "config/train.json", "LICENSE.ultralytics",
    "THIRD_PARTY_NOTICES.txt", "predict.py", "summarize_completed_run.py",
}
PACKAGES = {
    "standard": (
        ("custom_detector_nms_free.onnx", "custom_detector_nms_free.pt"), INFERENCE,
        "NMSフリー版と人物推論例。 / NMS-free model, checkpoint and person inference demo.",
    ),
    "developer": (
        tuple(MODEL_HASHES), INFERENCE,
        "両方式、チェックポイント、ONNX推論例と入出力仕様。 / Both variants, checkpoints, ONNX demo and I/O specification.",
    ),
    "commercial": (
        tuple(MODEL_HASHES), INFERENCE,
        "商用条件を検討するための社内用。許諾・サポート契約・価格は承認待ち。 / Internal review copy. Grant, support agreement and pricing are pending.",
    ),
    "custom-training": (
        (), ("TRAINING_INTAKE.md",),
        "ヒアリングと受入条件のドラフト。個別のデータ・学習・モデル成果物は案件合意後。 / Scoping and acceptance draft. Customer data, training and model delivery follow a project agreement.",
    ),
}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Unexpected original training source near {old.strip()}")
    return source.replace(old, new, 1)


def handoff_sources(handoff):
    with (handoff / "FILE_MANIFEST.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {name.lower().strip(): name for name in reader.fieldnames}
        path_field = fields.get("archive_path") or fields.get("path") or fields.get("relative_path")
        hash_field = fields.get("sha256") or fields.get("copy_sha256")
        if not path_field or not hash_field:
            raise ValueError("The handoff manifest lacks path or copy SHA-256 columns")
        manifest = {row[path_field].replace("\\", "/"): row[hash_field].strip().lower() for row in reader}
    inference = {}
    training = {}
    for name, expected in manifest.items():
        if name == ORIGINAL_INFERENCE:
            archive_path = "inference_tools.zip"
            dest = inference
        elif name.startswith("training/"):
            relative = name.removeprefix("training/")
            if relative not in TRAINING_FILES and not (relative.startswith("detector/") and relative.endswith(".py")):
                continue
            archive_path = name
            dest = training
        else:
            continue
        source = (handoff / name).resolve()
        if not source.is_relative_to(handoff) or digest(source) != expected:
            raise ValueError(f"Original handoff member hash mismatch: {name}")
        dest[archive_path] = source
    required = {"training/train.py", "training/config/train.json", "training/LICENSE.ultralytics",
                "training/THIRD_PARTY_NOTICES.txt", "training/prepare_data.py", "training/Start-Training.ps1"}
    if not inference or not required.issubset(training):
        raise ValueError("Original inference or training package inputs are incomplete")
    with ZipFile(inference["inference_tools.zip"]) as archive:
        members = set(archive.namelist())
        if not {"detect.py", "core.py", "THIRD_PARTY_NOTICES.txt", "LICENSE.ultralytics"}.issubset(members):
            raise ValueError("The original inference archive lacks expected scripts or notices")
        if not any(name.startswith("model_source/") and name.endswith(".py") for name in members):
            raise ValueError("The original inference archive lacks model source")
        if archive.testzip() is not None:
            raise ValueError("The original inference archive failed CRC verification")
        notices = {f"notices/{name}": archive.read(name) for name in
                   ("THIRD_PARTY_NOTICES.txt", "LICENSE.ultralytics")}
    config_name = "training/config/train.json"
    config = json.loads(training[config_name].read_text(encoding="utf-8-sig"))
    config.update({"project_dir": ".", "dataset_dir": "data/coco", "splits_file": "data/splits.json",
                   "archive_dir": "data/archives"})
    training[config_name] = (json.dumps(config, indent=2) + "\n").encode("utf-8")
    for name in ("training/prepare_data.py", "training/train.py"):
        source = training[name].read_text(encoding="utf-8-sig")
        source = replace_once(source, "D:/datasets/coco", "data/archives")
        if name.endswith("/train.py"):
            source = replace_once(source, "    criterion = DetectionCriterion",
                                  "    if args.warm_start:\n"
                                  "        from warm_start import load_shared_features\n"
                                  "        report.metadata['warm_start'] = load_shared_features(model, args.warm_start)\n"
                                  "    criterion = DetectionCriterion")
            source = replace_once(source, "    parser.add_argument('--resume', type=Path)",
                                  "    checkpoint = parser.add_mutually_exclusive_group()\n"
                                  "    checkpoint.add_argument('--resume', type=Path)\n"
                                  "    checkpoint.add_argument('--warm-start', type=Path,\n"
                                  "                            help='Load shared features with fresh heads and optimizer')")
        training[name] = source.encode("utf-8")
    for name, source in training.items():
        content = source if isinstance(source, bytes) else source.read_bytes()
        if re.search(rb"(?<![A-Za-z])[A-Za-z]:[/\\]|/Users/|/home/", content):
            raise ValueError(f"Host path remains in packaged training input: {name}")
    training["training/GETTING_STARTED.md"] = HERE / "TRAINING_QUICKSTART.md"
    training["training/warm_start.py"] = HERE / "templates" / "warm_start.py"
    return inference, training, notices


def member_info(name):
    date = tuple(map(int, RELEASE["release_date"].split("-")))
    info = ZipInfo(name, (*date, 0, 0, 0))
    info.external_attr = 0o644 << 16
    return info


def build(name, weights, extras, contents, private_sources):
    package = (
        f"# Custom Detector {name} · {VERSION}\n\n"
        "> ローカル限定・提供承認待ち / Local only, release approval pending.\n\n"
        f"{contents}\n\n"
    )
    if "inference_tools.zip" in private_sources:
        package += (
            "元の `inference_tools.zip` をこのZIPの展開先に重ねて展開すると、`detect.py` とモデルが同じ階層になります。"
            "元のスクリプトは `python detect.py input.jpg --backend torch --mode nms_free` で実行できます。"
            "展開後の翻案通知とライセンスを参照してください。\n\n"
            "Extract the original `inference_tools.zip` alongside the model files and run "
            "`python detect.py input.jpg --backend torch --mode nms_free`. See the original notices and license inside.\n"
        )
    elif weights:
        package += "付属のONNX例でローカル確認ができます。 / Use the included ONNX example for local checks.\n"
    if name == "developer" and "training/train.py" in private_sources:
        package += (
            "\n[学習とウォームスタートの手順](training/GETTING_STARTED.md) と"
            "[相対パス版の学習設定](training/config/train.json) を参照してください。COCO画像と学習実行記録は別途管理しています。\n\n"
            "See [training and warm-start](training/GETTING_STARTED.md) and the "
            "[portable training configuration](training/config/train.json). COCO images and original run records are managed separately.\n"
        )

    sources = {file: HERE / file for file in (*COMMON, *extras)}
    sources.update({file: MODELS / file for file in weights})
    sources.update(private_sources)
    if not weights:
        sources.pop("README.md")
    evaluation = ROOT / "product" / "evaluation"
    sources["evidence/artifact-audit.json"] = evaluation / "artifact-audit.json"
    sources["evidence/origin-training.json"] = evaluation / "origin-training.json"
    sources["evidence/runtime-measurements.json"] = evaluation / "runtime" / "measurements.json"
    run_record = json.loads((evaluation / "run-record.json").read_text(encoding="utf-8-sig"))
    run_record = {key: run_record[key] for key in ("release", "result_file_written_at", "evaluation_source_commit",
                                                   "training_source_commit", "training_source_status",
                                                   "quality_environment")}
    sources["evidence/run-record.json"] = (json.dumps(run_record, indent=2) + "\n").encode("utf-8")
    for filename in COMMON:
        source = sources.get(filename)
        if source is None or not filename.endswith(".md"):
            continue
        body = source.read_text(encoding="utf-8-sig")
        body = body.replace("[Original model README](../../models/custom_detector/README.md#ファイルの照合)",
                            "[Archive checksums](SHA256SUMS.txt)")
        body = body.replace("../evaluation/artifact-audit.json", "evidence/artifact-audit.json")
        body = body.replace("`product/evaluation/run-record.json`", "[Run record](evidence/run-record.json)")
        body = body.replace("`product/evaluation/origin-training.json`",
                            "[Training provenance](evidence/origin-training.json)")
        body = body.replace("`product/evaluation/runtime/measurements.json`",
                            "[Runtime measurements](evidence/runtime-measurements.json)")
        sources[filename] = body.encode("utf-8")
    files = {file: sha256(source).hexdigest() if isinstance(source, bytes) else digest(source)
             for file, source in sources.items()}
    files["PACKAGE.md"] = sha256(package.encode("utf-8")).hexdigest()
    manifest = "".join(f"{value}  {file}\n" for file, value in sorted(files.items()))
    output = DEST / f"custom-detector-{name}-{VERSION}.zip"
    temp = output.with_suffix(".zip.part")
    try:
        with ZipFile(temp, "w", compression=ZIP_STORED) as archive:
            for file, source in sorted(sources.items()):
                if isinstance(source, bytes):
                    archive.writestr(member_info(file), source)
                else:
                    with source.open("rb") as stream, archive.open(member_info(file), "w") as target:
                        copyfileobj(stream, target)
            archive.writestr(member_info("PACKAGE.md"), package.encode("utf-8"))
            archive.writestr(member_info("SHA256SUMS.txt"), manifest.encode("utf-8"))
        with ZipFile(temp) as archive:
            members = set(archive.namelist())
            if members != files.keys() | {"SHA256SUMS.txt"}:
                raise ValueError(f"Unexpected archive contents: {output.name}")
            for file, value in files.items():
                with archive.open(file) as stream:
                    if file_digest(stream, "sha256").hexdigest() != value:
                        raise ValueError(f"Archive member hash mismatch: {output.name} / {file}")
            for file in members:
                if file.endswith(".md"):
                    body = archive.read(file).decode("utf-8-sig")
                    for target in re.findall(r"\]\(([^)]+)\)", body):
                        if "://" in target or target.startswith("#"):
                            continue
                        path = normpath(join(dirname(file), target.partition("#")[0]))
                        if path not in members:
                            raise ValueError(f"Unresolved archive link: {file} -> {target}")
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(f"{output.name}  {output.stat().st_size:,} bytes")
    return f"{digest(output)}  {output.name}\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=ROOT / "private/handoff/source",
                        help="Verified extraction root containing FILE_MANIFEST.csv")
    args = parser.parse_args()
    for file, expected in MODEL_HASHES.items():
        if digest(MODELS / file) != expected:
            raise ValueError(f"Original hash mismatch: {file}")
    DEST.mkdir(parents=True, exist_ok=True)
    if not (args.handoff / "FILE_MANIFEST.csv").is_file():
        parser.error("Rebuilding the four packages requires the original source handoff. Set --handoff to its extraction root.")
    inference, training, notices = handoff_sources(args.handoff.resolve())
    operations = (ROOT / "product" / "launch" / "custom-training-operations.md").read_text(encoding="utf-8-sig")
    operations = operations.replace("../materials/intake_dataset.py", "intake_dataset.py")
    operations = operations.replace("(rights-review.md)", "(RIGHTS_STATUS.md)")
    source_packages = {
        "standard": inference | notices, "developer": inference | training | notices,
        "commercial": notices,
        "custom-training": {
            "intake_dataset.py": HERE / "intake_dataset.py",
            "DATASET_OPERATIONS.md": operations.encode("utf-8"),
            "requirements-intake.txt": b"Pillow==10.2.0\n",
        },
    }
    for extras in source_packages.values():
        for source in extras.values():
            if isinstance(source, Path) and not source.is_file():
                raise FileNotFoundError(source)
    manifest = "".join(build(name, *spec, source_packages.get(name, {})) for name, spec in PACKAGES.items())
    (DEST / "SHA256SUMS.txt").write_text(manifest, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
