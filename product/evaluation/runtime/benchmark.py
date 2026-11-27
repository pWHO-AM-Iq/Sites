"""Sequential batch-1 runtime and input checks for existing custom detector assets."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import zipfile

import numpy as np
from PIL import Image
import psutil


def decode_image(blob):
    with Image.open(BytesIO(blob)) as image:
        return image.convert("RGB")


def encode_image(image, kind):
    buffer = BytesIO()
    image.save(buffer, format=kind)
    return buffer.getvalue()


def preprocess(image):
    width, height = image.size
    ratio = min(640 / width, 640 / height)
    resized = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    offset = ((640 - resized[0]) // 2, (640 - resized[1]) // 2)
    canvas = Image.new("RGB", (640, 640), (114, 114, 114))
    canvas.paste(image.resize(resized, Image.Resampling.BILINEAR), offset)
    tensor = np.ascontiguousarray(np.asarray(canvas).transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    return tensor, np.array((*offset, *offset)), np.array((resized[0] / width, resized[1] / height) * 2)


def detections(boxes, scores, image, offset, ratio, mode):
    if boxes.shape != (1, 8400, 4) or scores.shape != (1, 8400, 80):
        raise ValueError(f"Unexpected output shapes {boxes.shape}, {scores.shape}")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Model returned non-finite output")
    boxes, scores = boxes[0], scores[0]
    if mode == "nms_free":
        idx = np.argsort(-scores.max(axis=1), kind="stable")[:300]
        flat = np.argsort(-scores[idx].ravel(), kind="stable")[:300]
        row, label = divmod(flat, scores.shape[1])
        idx = idx[row]
        selected = scores[idx, label] > 0.35
        idx, label = idx[selected], label[selected]
    else:
        label = np.argmax(scores, axis=1)
        confidence = scores[np.arange(len(scores)), label]
        candidates = np.flatnonzero(confidence > 0.35)
        candidates = candidates[np.argsort(-confidence[candidates], kind="stable")]
        kept = []
        while len(candidates) and len(kept) < 300:
            chosen, candidates = candidates[0], candidates[1:]
            kept.append(chosen)
            size = np.maximum(0, np.minimum(boxes[chosen, 2:], boxes[candidates, 2:]) -
                              np.maximum(boxes[chosen, :2], boxes[candidates, :2]))
            intersection = size.prod(axis=1)
            chosen_area = np.maximum(0, boxes[chosen, 2:] - boxes[chosen, :2]).prod()
            areas = np.maximum(0, boxes[candidates, 2:] - boxes[candidates, :2]).prod(axis=1)
            overlap = intersection / np.maximum(chosen_area + areas - intersection, 1e-12)
            candidates = candidates[(label[candidates] != label[chosen]) | (overlap <= 0.7)]
        idx = np.asarray(kept, dtype=int)
        label = label[idx]
    width, height = image.size
    restored = np.clip((boxes[idx] - offset) / ratio, 0, (width, height, width, height))
    return np.column_stack((restored, scores[idx, label], label))


def build_backend(args, model):
    if args.backend.startswith("ort"):
        import onnxruntime as ort
        providers = {
            "ort-cpu": ["CPUExecutionProvider"],
            "ort-cuda": [("CUDAExecutionProvider", {"device_id": args.gpu})],
            "ort-trt": [("TensorrtExecutionProvider", {"device_id": args.gpu}),
                        ("CUDAExecutionProvider", {"device_id": args.gpu})],
        }[args.backend]
        expected = "CPUExecutionProvider" if args.backend == "ort-cpu" else providers[0][0]
        if expected not in ort.get_available_providers():
            raise RuntimeError(f"{expected} unavailable. Available: {ort.get_available_providers()}")
        if args.backend != "ort-cpu":
            ort.preload_dlls()
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model), options, providers=providers)
        session.disable_fallback()
        if session.get_providers()[0] != expected:
            raise RuntimeError(f"{expected} failed to activate: {session.get_providers()}")
        return (lambda tensor: session.run(["boxes", "scores"], {"images": tensor}),
                {"version": ort.__version__, "providers": session.get_providers(),
                 "provider_options": session.get_provider_options(), "transfer_in_forward": True},
                lambda: {})
    import torch
    if not args.core:
        raise RuntimeError("PyTorch requires the trusted distribution's core.py directory (--core)")
    sys.path.insert(0, str(args.core.resolve()))
    from core import load_torch_model
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu" if args.backend == "torch-cpu" else f"cuda:{args.gpu}")
    dtype = torch.float16 if args.backend == "torch-fp16" else torch.float32
    detector = load_torch_model(model, args.mode).eval().to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)

    @torch.inference_mode()
    def forward(tensor):
        input_tensor = torch.from_numpy(tensor).to(device=device, dtype=dtype)
        boxes, logits = detector(input_tensor)
        return boxes.float().cpu().numpy(), logits.float().sigmoid().cpu().numpy()

    def memory():
        if device.type == "cuda":
            return {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        return {}

    return forward, {"version": torch.__version__, "device": str(device),
                     "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "cuda_version": torch.version.cuda, "precision": str(dtype),
                     "transfer_in_forward": True}, memory


def inputs(path, count):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = sorted(name for name in archive.namelist() if name.lower().endswith(".jpg"))
            if not names:
                raise ValueError("Archive contains no JPEG files")
            return [(names[index], archive.read(names[index])) for index in
                    np.linspace(0, len(names) - 1, min(count, len(names)), dtype=int)]
    return [(path.name, path.read_bytes())]


def benchmark(samples, execute, warmup, iterations):
    for index in range(warmup):
        execute(lambda: decode_image(samples[index % len(samples)][1]))
    timings = []
    expected = {}
    mismatches = 0
    outputs = {}
    for index in range(iterations):
        name, blob = samples[index % len(samples)]
        timing, rows = execute(lambda: decode_image(blob))
        digest = hashlib.sha256(rows.tobytes()).hexdigest()
        mismatches += digest != expected.setdefault(name, digest)
        if name not in outputs:
            outputs[name] = rows.tolist()
        timings.append(timing)
    return summarize(timings) | {"repeat_output_mismatches": mismatches,
                                  "unique_inputs": len(expected), "sample_detections": outputs}


def summarize(timings):
    array = np.array(timings) * 1000
    return {
        "iterations": len(timings),
        "ms": {stage: {"mean": float(array[:, i].mean()), "p50": float(np.percentile(array[:, i], 50)),
                        "p95": float(np.percentile(array[:, i], 95))}
               for i, stage in enumerate(("decode", "preprocess", "forward_with_transfers",
                                          "postprocess", "e2e"))},
        "fps": float(1000 / array[:, -1].mean()),
    }


def compatibility(base, execute, video_path):
    base = base.resize((640, 480))
    random = np.random.default_rng(42)
    noisy = np.clip(np.asarray(base).astype(np.int16) + random.normal(0, 25, (480, 640, 3)),
                    0, 255).astype(np.uint8)
    small = Image.new("RGB", base.size, (114, 114, 114))
    small.paste(base.resize((80, 60)), (280, 210))
    multiple = Image.new("RGB", (1280, 960))
    for x in (0, 640):
        for y in (0, 480):
            multiple.paste(base, (x, y))
    cases = {"jpeg": (base, "JPEG"), "png": (base, "PNG"),
             "wide": (base.resize((1280, 180)), "PNG"),
             "tall": (base.resize((180, 1280)), "PNG"),
             "blank": (Image.new("RGB", base.size), "PNG"),
             "low_light": (base.point(lambda p: p // 8), "PNG"),
             "noise": (Image.fromarray(noisy), "PNG"),
             "small_content": (small, "PNG"), "tiled_content": (multiple, "PNG")}
    results = {}
    for name, (image, kind) in cases.items():
        blob = encode_image(image, kind)
        timing, rows = execute(lambda: decode_image(blob))
        results[name] = {"status": "completed", "detections": len(rows), "size": image.size,
                         "e2e_ms": timing[-1] * 1000}
    from PIL import UnidentifiedImageError
    try:
        execute(lambda: decode_image(b""))
    except (UnidentifiedImageError, OSError):
        results["empty_bytes"] = {"status": "rejected_at_decode"}
    else:
        raise AssertionError("Empty input was accepted")
    try:
        import cv2
    except ImportError:
        return results | {"video": {"status": "unavailable", "reason": "OpenCV not installed"}}
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, base.size)
    if not writer.isOpened():
        writer.release()
        return results | {"video": {"status": "unavailable", "reason": "MP4 writer failed to open"}}
    try:
        for i in range(30):
            frame = np.roll(np.asarray(base), 2 * i, axis=1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video_path))
    timings, counts = [], []

    def read_frame():
        ok, frame = capture.read()
        if not ok:
            raise EOFError
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    try:
        while True:
            try:
                timing, rows = execute(read_frame)
            except EOFError:
                break
            timings.append(timing)
            counts.append(len(rows))
    finally:
        capture.release()
    results["video"] = {"status": "completed" if len(counts) == 30 else "incomplete",
                        "expected_frames": 30, "processed_frames": len(counts),
                        "detections_per_frame": counts,
                        "performance": summarize(timings) if timings else None}
    return results


def run(args):
    result = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "mode": args.mode, "backend": args.backend, "batch": 1, "size": [640, 640],
              "threads": args.threads, "warmup": args.warmup,
              "hardware": {"os": platform.platform(), "cpu": platform.processor(),
                           "logical_cpus": os.cpu_count(), "python": platform.python_version()}}
    suffix = ".onnx" if args.backend.startswith("ort") else ".pt"
    model = args.models / f"custom_detector_{args.mode}{suffix}"
    result["model_file"] = model.name
    process = psutil.Process()
    result["rss_bytes_before_load"] = process.memory_info().rss
    try:
        with model.open("rb") as artifact:
            result["sha256"] = hashlib.file_digest(artifact, "sha256").hexdigest()
        start = perf_counter()
        forward, backend, backend_memory = build_backend(args, model)
        result["backend_details"] = backend
        result["session_init_ms"] = (perf_counter() - start) * 1000
        peak_rss = process.memory_info().rss

        def execute(decode):
            nonlocal peak_rss
            start = perf_counter()
            image = decode()
            decoded = perf_counter()
            tensor, offset, ratio = preprocess(image)
            prepared = perf_counter()
            boxes, scores = forward(tensor)
            inferred = perf_counter()
            rows = detections(boxes, scores, image, offset, ratio, args.mode)
            finished = perf_counter()
            if not np.isfinite(rows).all() or np.any(rows[:, 2:4] < rows[:, :2]):
                raise ValueError("Postprocessed detections failed geometry/finite checks")
            peak_rss = max(peak_rss, process.memory_info().rss)
            return (decoded - start, prepared - decoded, inferred - prepared,
                    finished - inferred, finished - start), rows

        samples = inputs(args.input, args.samples)
        result["sample_names"] = [name for name, _ in samples]
        result["steady_state"] = benchmark(samples, execute, args.warmup, args.iterations)
        if args.compatibility:
            result["compatibility"] = compatibility(decode_image(samples[0][1]), execute,
                                                    args.output.with_suffix(".mp4"))
        result["rss_bytes_peak_at_step_boundaries"] = peak_rss
        result["rss_bytes_at_end"] = process.memory_info().rss
        result["device_memory"] = backend_memory()
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "failed"
        result["reason"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result.get("reason", str(args.output)))
    return 0 if result["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ort-cpu", "ort-cuda", "ort-trt", "torch-cpu",
                                               "torch-fp32", "torch-fp16"), required=True)
    parser.add_argument("--mode", choices=("nms", "nms_free"), required=True)
    parser.add_argument("--models", type=Path, default=Path("models/custom_detector"))
    parser.add_argument("--core", type=Path)
    parser.add_argument("--input", type=Path, required=True, help="Image or ZIP containing JPEGs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--compatibility", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.samples, args.iterations) < 1 or args.warmup < 0 or args.gpu < 0:
        parser.error("Expected positive threads/samples/iterations and nonnegative warmup/GPU")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
