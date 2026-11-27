"""Offline checks for explicitly split COCO bbox or YOLO detection data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def entries(root, suffixes):
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes),
        key=lambda path: path.relative_to(root).as_posix(),
    )


class Audit:
    def __init__(self):
        self.issues = []
        self.files = defaultdict(list)
        self.hashes = defaultdict(list)

    def issue(self, split, path, message):
        self.issues.append({"split": split, "path": str(path), "message": message})

    def image(self, root, name, split):
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name or not rel.parts:
            self.issue(split, name, "Image name must be a relative POSIX path within its image root.")
            return None
        path = root.joinpath(*rel.parts)
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root.resolve()):
                raise ValueError("Path or symlink escapes the image root.")
            with Image.open(resolved) as image:
                width, height = image.size
                image.verify()
            with Image.open(resolved) as image:
                image.load()
            with resolved.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            self.issue(split, name, f"Cannot read image: {error}")
            return None
        identifier = {"split": split, "file_name": name}
        self.hashes[checksum].append(identifier)
        self.files[resolved].append(identifier)
        return width, height

    def duplicates(self):
        for source, label in ((self.files, "same physical path"), (self.hashes, "identical file bytes (SHA-256)")):
            for identifiers in source.values():
                if len(identifiers) < 2:
                    continue
                involved = {item["split"] for item in identifiers}
                description = "Train/validation intersection" if len(involved) > 1 else "Duplicate image"
                details = ", ".join(f"{item['split']}:{item['file_name']}" for item in identifiers)
                self.issue("/".join(sorted(involved)), details, f"{description}: {label}.")


def is_int(value):
    return type(value) is int


def bbox_numbers(values):
    return (
        isinstance(values, list)
        and len(values) == 4
        and all(type(value) in (float, int) and abs(value) <= 1e12 and math.isfinite(value) for value in values)
    )


def valid_box(box, width, height, epsilon=0.0):
    x, y, w, h = box
    return (
        w > 0 and h > 0 and x >= -epsilon and y >= -epsilon
        and x + w <= width + epsilon and y + h <= height + epsilon
    )


def yolo_split(audit, split, images_root, labels_root, classes):
    images, annotations = [], []
    image_keys = set()
    labels = {}
    for label in entries(labels_root, {".txt"}):
        key = label.relative_to(labels_root).with_suffix("").as_posix()
        if key in labels:
            audit.issue(split, label.relative_to(labels_root), "Multiple label files share the same stem.")
        labels[key] = label
    for path in entries(images_root, IMAGE_SUFFIXES):
        name = path.relative_to(images_root).as_posix()
        key = path.relative_to(images_root).with_suffix("").as_posix()
        if key in image_keys:
            audit.issue(split, name, "Multiple images share the same YOLO label stem.")
            continue
        image_keys.add(key)
        size = audit.image(images_root, name, split)
        if size is None:
            continue
        width, height = size
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})
        label_path = labels.get(key)
        if label_path is None:
            audit.issue(split, name, "Missing label file. Use an empty TXT for an intentional negative.")
            continue
        try:
            if not label_path.resolve().is_relative_to(labels_root.resolve()):
                raise ValueError("Label symlink escapes the label root.")
            lines = label_path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, ValueError) as error:
            audit.issue(split, name, f"Cannot read labels: {error}")
            continue
        seen = set()
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            location = f"{label_path.relative_to(labels_root).as_posix()}:{number}"
            fields = line.split()
            try:
                if len(fields) != 5:
                    raise ValueError("Expected class_id cx cy width height.")
                class_id = int(fields[0])
                cx, cy, w, h = coordinates = [float(field) for field in fields[1:]]
                if not 0 <= class_id < len(classes):
                    raise ValueError("Class index is outside the class list.")
                if not all(0 <= value <= 1 for value in coordinates):
                    raise ValueError("Normalized coordinates must be finite and between 0 and 1.")
                box = [cx - w / 2, cy - h / 2, w, h]
                if not valid_box(box, 1, 1, epsilon=1e-9):
                    raise ValueError("Box must have positive size and fit inside the image.")
            except ValueError as error:
                audit.issue(split, location, str(error))
                continue
            signature = (class_id, *coordinates)
            if signature in seen:
                audit.issue(split, location, "Duplicate detection label.")
                continue
            seen.add(signature)
            x1, y1 = max(0, box[0]) * width, max(0, box[1]) * height
            x2, y2 = min(1, box[0] + w) * width, min(1, box[1] + h) * height
            pixel_box = [x1, y1, x2 - x1, y2 - y1]
            if pixel_box[2] <= 0 or pixel_box[3] <= 0:
                audit.issue(split, location, "Box collapses to zero size in pixel coordinates.")
                continue
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": class_id + 1,
                                "bbox": pixel_box, "area": pixel_box[2] * pixel_box[3], "iscrowd": 0})
    for key in sorted(labels.keys() - image_keys):
        audit.issue(split, labels[key].relative_to(labels_root), "Orphan label file.")
    categories = [{"id": index, "name": name} for index, name in enumerate(classes, start=1)]
    return {"info": {}, "images": images, "annotations": annotations, "categories": categories}


def coco_split(audit, split, images_root, json_path):
    data = read_json(json_path)
    if not isinstance(data, dict):
        raise ValueError(f"{split}: COCO data must be a JSON object.")
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list) or not all(isinstance(item, dict) for item in data[key]):
            raise ValueError(f"{split}: '{key}' must be a list of objects.")
    categories, records, names = {}, {}, set()
    for category in data["categories"]:
        cid, name = category.get("id"), category.get("name")
        if (not is_int(cid) or cid < 0 or cid in categories or not isinstance(name, str)
                or not name.strip() or name != name.strip() or name in names):
            audit.issue(split, "categories", "Category IDs and nonempty names must be unique.")
            continue
        categories[cid] = name
        names.add(name)

    file_names = set()
    for image in data["images"]:
        iid, name = image.get("id"), image.get("file_name")
        if not is_int(iid) or iid < 0 or iid in records or not isinstance(name, str) or not name:
            audit.issue(split, "images", "Image ID must be a unique nonnegative integer with a file name.")
            continue
        if name in file_names:
            audit.issue(split, name, "Duplicate image file name.")
        file_names.add(name)
        size = audit.image(images_root, name, split)
        declared = image.get("width"), image.get("height")
        records[iid] = size
        if not all(is_int(value) and value > 0 for value in declared) or size != declared:
            audit.issue(split, name, f"Declared dimensions {declared} do not match decoded size {size}.")

    annotation_ids, detection_rows = set(), set()
    for annotation in data["annotations"]:
        aid = annotation.get("id")
        if not is_int(aid) or aid < 0 or aid in annotation_ids:
            audit.issue(split, "annotations", "Annotation IDs must be unique nonnegative integers.")
            continue
        annotation_ids.add(aid)
        iid, cid = annotation.get("image_id"), annotation.get("category_id")
        box = annotation.get("bbox")
        if not is_int(iid) or iid not in records or not is_int(cid) or cid not in categories:
            audit.issue(split, aid, "Unknown image or category ID.")
            continue
        size = records[iid]
        if size is None or not bbox_numbers(box) or not valid_box(box, *size):
            audit.issue(split, aid, "Bounding box must be finite, positive and inside the decoded image.")
            continue
        detection_signature = (iid, cid, *box)
        if detection_signature in detection_rows:
            audit.issue(split, aid, "Duplicate detection annotation.")
        detection_rows.add(detection_signature)
    return data, categories


def required_directory(value):
    candidate = Path(value)
    if not candidate.is_dir():
        raise argparse.ArgumentTypeError(f"Not a directory: {value}")
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    formats = parser.add_subparsers(dest="format", required=True)
    yolo = formats.add_parser("yolo", help="Check YOLO detection TXT; optionally produce COCO JSON.")
    yolo.add_argument("--classes", required=True, type=Path, help="UTF-8; one class name per line, in index order.")
    yolo.add_argument("--output-dir", type=Path, help="New output directory. Written only after a clean audit.")
    coco = formats.add_parser("coco", help="Check explicitly split COCO bbox annotation JSON.")
    for subparser, source_type in ((yolo, "labels"), (coco, "json")):
        subparser.add_argument("--report", required=True, type=Path, help="New JSON report path; never overwritten.")
        for split in SPLITS:
            subparser.add_argument(f"--{split}-images", required=True, type=required_directory)
            subparser.add_argument(f"--{split}-{source_type}", required=True,
                                   type=required_directory if source_type == "labels" else Path)
    args = parser.parse_args()

    audit = Audit()
    results = {}
    try:
        if args.report.exists():
            raise ValueError("The report path already exists.")
        if args.format == "yolo":
            if args.output_dir and (args.output_dir.exists() or not args.output_dir.parent.is_dir()):
                raise ValueError("The conversion directory must be new and its parent must exist.")
            classes = [name.strip() for name in args.classes.read_text(encoding="utf-8-sig").splitlines()]
            if not classes or not all(classes) or len(set(classes)) != len(classes):
                raise ValueError("Class list must contain unique, nonempty names, one per line.")
            for split in SPLITS:
                results[split] = yolo_split(audit, split, getattr(args, f"{split}_images"),
                                            getattr(args, f"{split}_labels"), classes)
        else:
            category_maps = {}
            for split in SPLITS:
                results[split], category_maps[split] = coco_split(
                    audit, split, getattr(args, f"{split}_images"), getattr(args, f"{split}_json"))
                if not category_maps[split]:
                    audit.issue(split, "categories", "No valid categories.")
            if category_maps["train"] != category_maps["val"]:
                audit.issue("train/val", "categories", "Train and validation category ID/name mappings differ.")
        for split, dataset in results.items():
            if not dataset["images"]:
                audit.issue(split, "images", "Split contains no images.")
        audit.duplicates()
        summary = {
            split: {"images": len(dataset["images"]), "annotations": len(dataset["annotations"]),
                    "annotations_by_category_id": dict(sorted(Counter(
                        row["category_id"] for row in dataset["annotations"] if is_int(row.get("category_id"))
                    ).items()))}
            for split, dataset in results.items()
        }
        write_json(args.report, {"format": args.format, "ok": not audit.issues, "splits": summary,
                                 "issues": audit.issues})
        if audit.issues:
            print(f"{len(audit.issues)} issue(s); see {args.report}")
            return 1
        if args.format == "yolo" and args.output_dir:
            args.output_dir.mkdir()
            for split, data in results.items():
                write_json(args.output_dir / f"instances_{split}.json", data)
        print(f"Audit passed; see {args.report}")
        return 0
    except (ValueError, OSError) as error:
        parser.exit(2, f"intake_dataset: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

