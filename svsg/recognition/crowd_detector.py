"""Read full-body and head detections from the two-class CrowdHuman YOLOv5 ONNX export."""

from __future__ import annotations

from ast import literal_eval
from hashlib import file_digest
from pathlib import Path

from svsg.image_functions.raster import RasterFunction
from svsg.l1_compiler.detector_adapter import RawDetection


class CrowdHumanDetector:
    def __init__(self, weights: Path) -> None:
        import onnxruntime as ort

        with weights.open("rb") as stream:
            self.weights_sha256 = file_digest(stream, "sha256").hexdigest()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(weights), options, providers=["CPUExecutionProvider"]
        )
        names = self.session.get_modelmeta().custom_metadata_map.get("names", "{}")
        if literal_eval(names) != {0: "person", 1: "head"}:
            raise ValueError("expected a CrowdHuman ONNX export with classes 0=person and 1=head")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.model_name = weights.name

    def detect_raster(self, frame: RasterFunction) -> list[RawDetection]:
        import cv2
        import numpy as np

        size = 640
        scale = size / max(frame.width, frame.height)
        w, h = max(1, round(frame.width * scale)), max(1, round(frame.height * scale))
        dx, dy = (size - w) // 2, (size - h) // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        rgba = np.frombuffer(frame.samples, dtype=np.uint8).reshape(frame.height, frame.width, 4)
        canvas[dy : dy + h, dx : dx + w] = cv2.resize(rgba[:, :, :3], (w, h))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255)
        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 7:
            raise ValueError("expected CrowdHuman YOLOv5 output [1, anchors, 7]")
        rows = output[0]
        detections = []
        labels = rows[:, 5:].argmax(axis=1)
        for index, label in enumerate(("person", "head")):
            scores = rows[:, 4] * rows[:, 5 + index]
            keep = (labels == index) & (scores >= 0.35)
            scores = scores[keep]
            rects = rows[keep, :4].copy()
            rects[:, :2] -= rects[:, 2:] / 2
            for i in cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), 0.0, 0.58):
                x, y, width, height = map(float, rects[i])
                left = max(0, round((x - dx) * frame.width / w))
                top = max(0, round((y - dy) * frame.height / h))
                right = min(frame.width, round((x + width - dx) * frame.width / w))
                bottom = min(frame.height, round((y + height - dy) * frame.height / h))
                if left < right and top < bottom:
                    detections.append(
                        RawDetection(label, (left, top, right, bottom), float(scores[i]))
                    )
        return detections
