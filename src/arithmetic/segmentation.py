"""
Segmentation routines for arithmetic expressions.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, Iterable, List, Tuple

from .preprocess import ArithmeticPreprocessor


class ArithmeticSegmenter:
    """Split handwritten arithmetic expressions into individual symbol patches."""

    METHOD_ALIASES: Dict[str, str] = {
        "": "hybrid",
        "hybrid": "hybrid",
        "contours": "hybrid",
        "default": "hybrid",
        "connected": "connected_components",
        "connected_components": "connected_components",
        "components": "connected_components",
        "projection": "projection",
        "column_projection": "projection",
        "best": "best",
        "auto": "best",
        "auto_selection": "best",
    }

    def __init__(self, target_size: int = 28) -> None:
        self.preprocessor = ArithmeticPreprocessor(target_size=target_size)
        self.target_size = target_size

    def _normalise_method(self, method: str | None) -> str:
        if not method:
            return "hybrid"
        key = method.lower().strip().replace("-", "_")
        key = key.replace(" ", "_")
        return self.METHOD_ALIASES.get(key, key)

    def _prepare_binary(self, image: np.ndarray, variant: str = "standard") -> np.ndarray:
        gray = self.preprocessor.to_grayscale(image)
        normalized = self.preprocessor.normalise_background(gray)
        blurred = cv2.GaussianBlur(normalized, (3, 3), sigmaX=0.6)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        variant_key = variant.lower()
        if variant_key == "thin":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            binary = cv2.erode(binary, kernel, iterations=1)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        elif variant_key == "thick":
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            binary = cv2.dilate(binary, kernel_close, iterations=1)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    def _split_long_box(
        self, box: Tuple[int, int, int, int], binary: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Split overly wide bounding boxes using vertical projection."""
        x, y, w, h = box
        aspect = w / float(max(1, h))

        roi = binary[y : y + h, x : x + w]
        profile = np.sum(roi > 0, axis=0)
        if profile.max() == 0:
            return [box]

        smoothed = cv2.GaussianBlur(profile.astype(np.float32), (1, 5), 0).flatten()
        if smoothed.max() <= 0:
            return [box]

        threshold = 0.22 if aspect > 1.5 else 0.18
        valleys = np.where(smoothed < threshold * smoothed.max())[0]
        if valleys.size == 0:
            return [box]
        best_split = None
        longest = 0
        start = valleys[0]
        prev = valleys[0]
        for idx in valleys[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                run_len = prev - start + 1
                if run_len > longest:
                    longest = run_len
                    best_split = (start + prev) // 2
                start = idx
                prev = idx
        run_len = prev - start + 1
        if run_len > longest:
            longest = run_len
            best_split = (start + prev) // 2

        if aspect <= 1.35 and longest < max(4, int(0.18 * w)):
            return [box]
        if best_split is None or best_split <= 3 or best_split >= w - 3:
            return [box]

        left = (x, y, best_split, h)
        right = (x + best_split, y, w - best_split, h)
        return [left, right]

    def _extract_sorted_boxes(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 35:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h < 6:
                continue
            boxes.append((x, y, w, h))
        boxes.sort(key=lambda item: (item[0], item[1]))
        refined: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            refined.extend(self._split_long_box(box, binary))
        refined.sort(key=lambda item: (item[0], item[1]))
        return refined

    def _segment_hybrid(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        return self._extract_sorted_boxes(binary)

    def _segment_connected_components(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes: List[Tuple[int, int, int, int]] = []
        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            if area < 35 or h < 6:
                continue
            boxes.append((int(x), int(y), int(w), int(h)))
        boxes.sort(key=lambda item: (item[0], item[1]))
        refined: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            refined.extend(self._split_long_box(box, binary))
        refined.sort(key=lambda item: (item[0], item[1]))
        return refined

    def _segment_projection(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        projection = binary.sum(axis=0).astype(np.float32)
        if projection.max() <= 0:
            return []
        threshold = 0.12 * projection.max()
        segments: List[Tuple[int, int]] = []
        in_run = False
        start = 0
        width = binary.shape[1]
        for idx, value in enumerate(projection):
            if value > threshold:
                if not in_run:
                    start = idx
                    in_run = True
            else:
                if in_run:
                    segments.append((start, idx))
                    in_run = False
        if in_run:
            segments.append((start, width))

        boxes: List[Tuple[int, int, int, int]] = []
        height = binary.shape[0]
        for start_col, end_col in segments:
            if end_col - start_col <= 0:
                continue
            x0 = max(0, start_col - 1)
            x1 = min(width, end_col + 1)
            slice_bw = binary[:, x0:x1]
            ys = np.where(slice_bw > 0)[0]
            if ys.size == 0:
                continue
            y0 = int(ys.min())
            y1 = int(ys.max())
            w = int(max(1, x1 - x0))
            h = int(y1 - y0 + 1)
            boxes.append((x0, y0, w, h))

        refined: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            refined.extend(self._split_long_box(box, binary))
        refined.sort(key=lambda item: (item[0], item[1]))
        return refined

    def _dispatch_boxes(self, binary: np.ndarray, method: str) -> List[Tuple[int, int, int, int]]:
        if method == "connected_components":
            return self._segment_connected_components(binary)
        if method == "projection":
            return self._segment_projection(binary)
        return self._segment_hybrid(binary)

    def _boxes_to_patches(
        self, boxes: List[Tuple[int, int, int, int]], binary: np.ndarray
    ) -> List[np.ndarray]:
        patches: List[np.ndarray] = []
        for x, y, w, h in boxes:
            roi = binary[y : y + h, x : x + w]
            normalized = self.preprocessor.extract_square(roi)
            patches.append(normalized.astype(np.float32) / 255.0)
        return patches

    def _segment_with_method(
        self, image: np.ndarray, method: str
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        if method == "projection":
            binary = self._prepare_binary(image, variant="thick")
        else:
            binary = self._prepare_binary(image, variant="standard")
        boxes = self._dispatch_boxes(binary, method)
        if not boxes and method != "projection":
            binary = self._prepare_binary(image, variant="thick")
            boxes = self._dispatch_boxes(binary, method)
        patches = self._boxes_to_patches(boxes, binary) if boxes else []
        meta = {"method": method, "binary": binary, "boxes": boxes}
        return patches, meta

    def _score_candidate(self, patches: List[np.ndarray], meta: Dict[str, Any]) -> float:
        boxes: List[Tuple[int, int, int, int]] = meta.get("boxes") or []
        if not patches or not boxes:
            return -1.0
        fill_ratios = []
        for patch in patches:
            mask = patch > 0.2
            fill = float(mask.sum()) / float(mask.size)
            fill_ratios.append(fill)
        avg_fill = float(np.mean(fill_ratios))
        heights = [box[3] for box in boxes]
        avg_height = float(np.mean(heights)) / float(meta["binary"].shape[0] or 1)
        span = (boxes[-1][0] + boxes[-1][2]) - boxes[0][0]
        normalised_span = float(span) / float(meta["binary"].shape[1] or 1)
        count_penalty = 0.018 * max(0, len(boxes) - 12)
        skinny_penalty = 0.05 if avg_fill < 0.08 else 0.0
        return avg_fill + 0.12 * avg_height + 0.08 * normalised_span - count_penalty - skinny_penalty

    def _segment_best(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        best: Tuple[float, List[np.ndarray], Dict[str, Any]] | None = None
        for method in ("hybrid", "connected_components", "projection"):
            patches, meta = self._segment_with_method(image, method)
            if not patches:
                continue
            score = self._score_candidate(patches, meta)
            if best is None or score > best[0]:
                best = (score, patches, meta)
        if best is None:
            return self._segment_with_method(image, "hybrid")
        return best[1], best[2]

    def segment(self, image: np.ndarray, method: str = "hybrid") -> List[np.ndarray]:
        method_key = self._normalise_method(method)
        if method_key == "best":
            patches, _ = self._segment_best(image)
            return patches
        patches, _ = self._segment_with_method(image, method_key)
        return patches

    def segment_candidates(
        self, image: np.ndarray, methods: Iterable[str]
    ) -> Dict[str, Tuple[List[np.ndarray], Dict[str, Any]]]:
        results: Dict[str, Tuple[List[np.ndarray], Dict[str, Any]]] = {}
        for method in methods:
            method_key = self._normalise_method(method)
            if method_key == "best":
                continue
            patches, meta = self._segment_with_method(image, method_key)
            results[method_key] = (patches, meta)
        if "hybrid" not in results:
            results["hybrid"] = self._segment_with_method(image, "hybrid")
        return results

    def visualise(self, image: np.ndarray, method: str = "hybrid") -> np.ndarray:
        method_key = self._normalise_method(method)
        _, meta = self._segment_with_method(image, method_key)
        binary = meta["binary"]
        boxes = meta["boxes"]
        overlay = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for idx, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(
                overlay,
                str(idx + 1),
                (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 128, 255),
                1,
                cv2.LINE_AA,
            )
        return overlay
