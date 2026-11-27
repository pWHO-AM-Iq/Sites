# Custom Detector ONNX inference example

The example reads an ONNX model supplied separately and returns person detections as a JSON array. Each detection contains `[left, top, right, bottom, confidence]` in original-image pixels. An image without detections returns an empty array.

## Setup and run

The release consists of this README, [detect_onnx.py](../../materials/detect_onnx.py), and [requirements-inference.txt](../../materials/requirements-inference.txt). Copy the example and requirements file into one directory. Use Python 3.12 and an image you have permission to process. Obtain the compatible Custom Detector ONNX artifact through its designated distributor and place it next to the script. The default file is `custom_detector_nms_free.onnx`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
.\.venv\Scripts\python.exe detect_onnx.py input.jpg --mode nms_free --score 0.35
```

Select `--mode nms` with the corresponding `custom_detector_nms.onnx` artifact. This example uses CPUExecutionProvider. See the [file interface](../../materials/SPEC.md) and [benchmark report](../../materials/BENCHMARK.md) for properties and evaluation conditions.

The example source was written against the documented input and output interface. Select and include the approved license for this source before release. Model artifacts, training images, and third-party model definitions are obtained and licensed separately. If publishing as a standalone repository, place the linked source, requirements, and referenced technical documents next to this README and adjust the relative links.
