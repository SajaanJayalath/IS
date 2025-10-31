"""
High-level pipeline orchestrating segmentation and recognition for arithmetic expressions.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .constants import MODEL_DISPLAY_NAMES, default_label_mapping
from .models import ArithmeticCNNModel, ArithmeticEnsemble
from .segmentation import ArithmeticSegmenter

try:
    from ..models import CNNModel  # type: ignore
except ImportError:  # pragma: no cover
    from models import CNNModel  # type: ignore

# Optional external MER adapter (Math-Expression-Recognizer-and-Solver)
try:  # pragma: no cover - optional integration
    from ..integrations.mer_adapter import MERSolverAdapter  # type: ignore
    _MER_AVAILABLE = True
except Exception:  # pragma: no cover
    MERSolverAdapter = None  # type: ignore
    _MER_AVAILABLE = False
SRC_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_PARENT not in sys.path:
    sys.path.append(SRC_PARENT)

from image_segmentation import MultiDigitProcessor  # type: ignore


class ArithmeticPipeline:
    """Segment and recognise handwritten arithmetic expressions."""

    def __init__(self, model_root: str | None = None) -> None:
        self.segmenter = ArithmeticSegmenter()
        self.models: Dict[str, object] = {}
        self.label_mapping: Dict[int, str] = default_label_mapping()
        self._label_to_index: Dict[str, int] = {symbol: idx for idx, symbol in self.label_mapping.items()}
        self.model_root = model_root or self._default_model_dir()
        self.digit_model: CNNModel | None = None
        self.digit_processor: MultiDigitProcessor | None = None
        self._mer: MERSolverAdapter | None = None  # external adapter, loaded lazily
        self._load_label_mapping()
        self.load_models()

    def _default_model_dir(self) -> str:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        return os.path.join(project_root, "models")

    def _load_label_mapping(self) -> None:
        candidate = os.path.join(self.model_root, "label_mapping_arithmetic.json")
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                self.label_mapping = {int(k): str(v) for k, v in data.items()}
                self._label_to_index = {symbol: idx for idx, symbol in self.label_mapping.items()}

    def load_models(self, force: bool = False) -> None:
        if self.models and not force:
            return
        cnn_path = os.path.join(self.model_root, "cnn_model_arithmetic.h5")
        if os.path.exists(cnn_path):
            cnn = ArithmeticCNNModel()
            cnn.load(cnn_path)
            self.models["cnn"] = cnn
            self.models["ensemble"] = ArithmeticEnsemble(cnn)
        else:
            # Initialise random-weight model as a fallback; caller must train before using.
            self.models["cnn"] = ArithmeticCNNModel()
            self.models["ensemble"] = ArithmeticEnsemble(self.models["cnn"])

        self._ensure_digit_model()

    def get_available_models(self) -> List[str]:
        names = [MODEL_DISPLAY_NAMES.get(key, key) for key in self.models.keys()]
        # Expose external MER option when adapter import is available; GUI also guards against dupes
        if _MER_AVAILABLE:
            disp = MODEL_DISPLAY_NAMES.get("mer", "MER (External)")
            if disp not in names:
                names.append(disp)
        return names

    def canonicalise_label(self, label: str) -> str:
        """Normalise predicted symbols for downstream evaluation."""
        if label in {"×", "x", "X", "A-", "times"}:
            return "*"
        if label in {"÷", "A�"}:
            return "/"
        return label

    def _resolve_model_key(self, model_display: str | None) -> str:
        if not model_display:
            return "cnn"
        for key, display in MODEL_DISPLAY_NAMES.items():
            if display == model_display:
                return key
        return model_display.lower()

    def _normalise_segmentation_method(self, method: str | None) -> str:
        if not method:
            return "hybrid"
        key = method.lower().strip().replace("-", "_")
        key = key.replace(" ", "_")
        if key in {"auto_selection", "auto", "best"}:
            return "best"
        if key in {"contours", "default"}:
            return "hybrid"
        return key

    def _predict_patch(
        self, patch: np.ndarray, predictor: Any
    ) -> Tuple[str, float, np.ndarray, np.ndarray]:
        patch_reshaped = patch.reshape(28, 28, 1)
        if hasattr(predictor, "predict_single"):
            label_idx, prob_vec = predictor.predict_single(patch_reshaped)
        else:
            batch = patch_reshaped.reshape(1, 28, 28, 1)
            result = predictor.predict(batch)
            label_idx = int(result.labels[0])
            prob_vec = result.probabilities[0]
        prob_vec = np.asarray(prob_vec, dtype=np.float32)
        if not (0 <= label_idx < len(prob_vec)):
            label_idx = int(np.argmax(prob_vec))
        label = self.label_mapping.get(label_idx, "?")
        canonical = self.canonicalise_label(label)
        confidence = float(prob_vec[label_idx]) if len(prob_vec) else 0.0
        canonical, confidence = self._refine_symbol(canonical, confidence, patch, prob_vec)
        glyph = (patch * 255.0).astype(np.uint8)
        return canonical, confidence, glyph, prob_vec

    def _classify_patches(
        self, patches: List[np.ndarray], predictor: Any
    ) -> Tuple[List[Tuple[str, float]], List[np.ndarray], List[np.ndarray]]:
        predictions: List[Tuple[str, float]] = []
        glyphs: List[np.ndarray] = []
        prob_vectors: List[np.ndarray] = []
        for patch in patches:
            canonical, confidence, glyph, prob_vec = self._predict_patch(patch, predictor)
            predictions.append((canonical, confidence))
            glyphs.append(glyph)
            prob_vectors.append(prob_vec)
        return predictions, glyphs, prob_vectors

    def _select_segmentation(
        self,
        image: np.ndarray,
        predictor: Any,
        candidate_methods: Sequence[str] | None = None,
    ) -> Tuple[str, List[np.ndarray], List[Tuple[str, float]], List[np.ndarray]]:
        methods = candidate_methods or ("hybrid", "connected_components", "projection")
        best: Tuple[float, str, List[np.ndarray], List[Tuple[str, float]], List[np.ndarray]] | None = None
        candidates = self.segmenter.segment_candidates(image, methods)
        for method, (patches, _meta) in candidates.items():
            if not patches:
                continue
            predictions, glyphs, _ = self._classify_patches(patches, predictor)
            confidences = [conf for _, conf in predictions]
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            fill_scores = [
                float((patch > 0.2).sum()) / float(patch.size) for patch in patches if patch.size
            ]
            avg_fill = float(np.mean(fill_scores)) if fill_scores else 0.0
            penalty = 0.02 * max(0, len(patches) - 12)
            score = avg_conf + 0.08 * avg_fill - penalty
            if best is None or score > best[0]:
                best = (score, method, patches, predictions, glyphs)
        if best is None:
            fallback_method = "hybrid"
            patches = self.segmenter.segment(image, fallback_method)
            predictions, glyphs, _ = self._classify_patches(patches, predictor) if patches else ([], [], [])
            return fallback_method, patches, predictions, glyphs
        return best[1], best[2], best[3], best[4]

    def _plus_structure_features(self, patch: np.ndarray) -> Tuple[float, float, float]:
        """Return structural score, corner occupancy, and vertical balance for identifying plus signs."""
        glyph = patch
        if glyph.ndim == 3:
            glyph = glyph[:, :, 0]
        binary = glyph > 0.35
        total = float(binary.sum())
        if total < 32:
            return 0.0, 1.0, 0.0
        height, width = binary.shape
        col_c = width // 2
        row_c = height // 2
        col_band = binary[:, max(0, col_c - 2) : min(width, col_c + 3)]
        row_band = binary[max(0, row_c - 2) : min(height, row_c + 3), :]
        col_ratio = float(col_band.sum()) / total
        row_ratio = float(row_band.sum()) / total
        if col_ratio < 0.12 or row_ratio < 0.12:
            return 0.0, 1.0, 0.0
        vert_presence = np.where(col_band.any(axis=1))[0]
        horiz_presence = np.where(row_band.any(axis=0))[0]
        v_extent = 0.0
        h_extent = 0.0
        if vert_presence.size >= 2:
            v_extent = (vert_presence[-1] - vert_presence[0] + 1) / float(height)
        if horiz_presence.size >= 2:
            h_extent = (horiz_presence[-1] - horiz_presence[0] + 1) / float(width)
        symmetry = 1.0 - abs(col_ratio - row_ratio)
        footprint = min(v_extent, h_extent)
        score = 0.45 * min(col_ratio, row_ratio) + 0.3 * symmetry + 0.25 * footprint

        third_h = max(1, height // 3)
        third_w = max(1, width // 3)
        corner_sum = (
            binary[:third_h, :third_w].sum()
            + binary[:third_h, -third_w:].sum()
            + binary[-third_h:, :third_w].sum()
            + binary[-third_h:, -third_w:].sum()
        )
        corner_ratio = float(corner_sum) / total if total else 1.0

        top_mass = float(binary[: height // 2, :].sum())
        bottom_mass = float(binary[height // 2 :, :].sum())
        v_balance = (
            min(top_mass, bottom_mass) / max(1.0, max(top_mass, bottom_mass))
            if (top_mass + bottom_mass) > 0
            else 0.0
        )

        return (
            float(max(0.0, min(1.0, score))),
            float(min(1.0, corner_ratio)),
            float(max(0.0, min(1.0, v_balance))),
        )

    def _asterisk_features(self, patch: np.ndarray) -> Tuple[float, float, float, int]:
        glyph = patch
        if glyph.ndim == 3:
            glyph = glyph[:, :, 0]
        binary = glyph > 0.35
        total = float(binary.sum())
        if total < 24:
            return 0.0, 0.0, 0.0, 0
        height, width = binary.shape
        col_c = width // 2
        row_c = height // 2
        col_slice = binary[:, max(0, col_c - 1) : min(width, col_c + 2)]
        row_slice = binary[max(0, row_c - 1) : min(height, row_c + 2), :]
        vertical = float(col_slice.sum()) / total
        horizontal = float(row_slice.sum()) / total
        diag_main = float(np.trace(binary.astype(np.float32))) / total
        diag_other = float(np.trace(np.fliplr(binary).astype(np.float32))) / total
        arms = np.asarray([vertical, horizontal, diag_main, diag_other], dtype=np.float32)
        mean_arm = float(arms.mean())
        min_arm = float(arms.min())
        diag_strength = float((diag_main + diag_other) * 0.5)
        comp_img = (binary.astype(np.uint8)) * 255
        num_components, _ = cv2.connectedComponents(comp_img, connectivity=8)
        return mean_arm, min_arm, diag_strength, int(max(0, num_components - 1))

    def _digit_shape_features(self, patch: np.ndarray) -> Dict[str, float]:
        glyph = patch
        if glyph.ndim == 3:
            glyph = glyph[:, :, 0]
        binary = glyph > 0.35
        total = float(binary.sum())
        height, width = binary.shape
        if total == 0:
            return {"aspect": 1.0, "left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0, "center": 0.0}
        third_h = max(1, height // 3)
        third_w = max(1, width // 3)
        left = float(binary[:, :third_w].sum()) / total
        right = float(binary[:, -third_w:].sum()) / total
        top = float(binary[:third_h, :].sum()) / total
        bottom = float(binary[-third_h:, :].sum()) / total
        center = float(
            binary[third_h : height - third_h, third_w : width - third_w].sum()
        ) / total
        aspect = width / float(max(1, height))
        return {
            "aspect": float(aspect),
            "left": float(left),
            "right": float(right),
            "top": float(top),
            "bottom": float(bottom),
            "center": float(center),
        }

    def _parenthesis_features(self, patch: np.ndarray) -> Tuple[float, str]:
        """
        Compute a score for a glyph resembling a parenthesis and the likely side.

        Heuristic characteristics:
        - Tall and narrow aspect ratio
        - Ink concentrated on one side band (left or right)
        - Sparse centre column occupancy
        """
        glyph = patch
        if glyph.ndim == 3:
            glyph = glyph[:, :, 0]
        binary = glyph > 0.35
        total = float(binary.sum())
        h, w = binary.shape
        if total < 18 or w == 0 or h == 0:
            return 0.0, "left"

        third_w = max(1, w // 3)
        left = float(binary[:, :third_w].sum()) / total
        center = float(binary[:, third_w : 2 * third_w].sum()) / total
        right = float(binary[:, -third_w:].sum()) / total

        # Expect tall, narrow glyph
        aspect = h / float(max(1, w))
        aspect_score = max(0.0, min(1.0, (aspect - 1.1) / 0.8))

        side = "left" if left >= right else "right"
        side_ratio = max(left, right)
        side_focus = max(0.0, side_ratio - center)

        # Low centre occupancy distinguishes from 0/8
        hollow_score = max(0.0, 1.0 - (center * 2.2))

        # Ensure vertical span is substantial
        row_presence = np.where(binary.any(axis=1))[0]
        v_extent = 0.0
        if row_presence.size >= 2:
            v_extent = (row_presence[-1] - row_presence[0] + 1) / float(h)
        span_score = max(0.0, min(1.0, (v_extent - 0.6) / 0.35))

        score = 0.38 * aspect_score + 0.34 * side_focus + 0.18 * hollow_score + 0.10 * span_score
        return float(max(0.0, min(1.0, score))), side

    def _ensure_digit_model(self) -> None:
        """Lazy-load digit classifier for low-confidence fallback."""
        if self.digit_model is not None:
            return

        candidates = [
            os.path.join(self.model_root, "cnn_model_mnist_csv.h5"),
            os.path.join(self.model_root, "cnn_model.h5"),
        ]
        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            try:
                model = CNNModel(num_classes=10, use_augmentation=False)
                model.load_model(candidate)
                self.digit_model = model
                return
            except Exception as exc:  # pragma: no cover - runtime environment specific
                print(f"Warning: failed to load digit fallback model '{candidate}': {exc}")
        self.digit_model = None

    def _ensure_digit_processor(self) -> None:
        """Initialise a digit-first multi-character processor for fallback flows."""
        if self.digit_processor is not None:
            return
        try:
            processor = MultiDigitProcessor()
            processor.set_processing_profile("digits")
            processor.set_allowed_characters([str(i) for i in range(10)])
            processor.load_models()
            self.digit_processor = processor
        except Exception as exc:  # pragma: no cover - dependent on environment assets
            print(f"Warning: failed to initialise digit fallback processor: {exc}")
            self.digit_processor = None

    def _refine_symbol(
        self,
        label: str,
        confidence: float,
        patch: np.ndarray,
        prob_vec: np.ndarray,
    ) -> Tuple[str, float]:
        """Apply lightweight heuristics to correct common confusions."""
        # Prefer structural detection for parentheses over high-prob digit misreads
        paren_score, paren_side = self._parenthesis_features(patch)
        if paren_score > 0.78 and label not in {"(", ")"}:
            inferred = "(" if paren_side == "left" else ")"
            # Blend heuristic strength with discounting of current belief
            zero_idx = self._label_to_index.get("0", -1)
            zero_conf = float(prob_vec[zero_idx]) if 0 <= zero_idx < len(prob_vec) else 0.0
            new_conf = max(confidence * 0.55, 0.62 + 0.25 * (paren_score - 0.78), 0.9 * (1.0 - zero_conf))
            return inferred, float(min(1.0, new_conf))
        plus_index = self._label_to_index.get("+", -1)
        four_index = self._label_to_index.get("4", -1)
        plus_prob = float(prob_vec[plus_index]) if plus_index != -1 and plus_index < len(prob_vec) else 0.0
        four_prob = float(prob_vec[four_index]) if four_index != -1 and four_index < len(prob_vec) else 0.0
        plus_score, corner_ratio, v_balance = self._plus_structure_features(patch)

        if label in {"/", "÷"} and confidence < 0.55:
            if plus_score > 0.28 and plus_score > confidence:
                new_conf = max(plus_prob, plus_score)
                return "+", new_conf

        if label == "+" and confidence < 0.25:
            boosted = max(confidence, plus_score, plus_prob)
            return label, boosted

        if label != "+":
            strong_cross = corner_ratio < 0.1 and plus_score > 0.48 and v_balance > 0.85
            if strong_cross or (plus_score > 0.48 and corner_ratio < 0.16 and v_balance > 0.9):
                gain = max(plus_prob, plus_score * (1.0 - corner_ratio))
                if strong_cross or confidence < 0.65 or gain >= confidence * 1.15:
                    return "+", max(confidence * 0.55, gain)

        asterisk_mean, asterisk_min, diag_strength, component_count = self._asterisk_features(patch)
        if (
            diag_strength > 0.14
            and asterisk_min > 0.1
            and asterisk_mean > 0.18
            and corner_ratio > 0.18
        ):
            if label != "*":
                boost = 0.5 + 0.45 * min(0.5, asterisk_mean)
                adjusted = max(confidence * 0.6, boost)
                if adjusted > confidence or component_count >= 3 or confidence < 0.85:
                    return "*", float(min(1.0, adjusted))

        if label == "3" and confidence < 0.4 and corner_ratio < 0.12 and v_balance < 0.9:
            inferred = max(four_prob, 0.5 * (1.0 - corner_ratio))
            if inferred > confidence or inferred > 0.18:
                return "4", max(confidence, inferred)

        if label.isdigit():
            features = self._digit_shape_features(patch)
            if label == "3" and confidence < 0.3:
                if features["aspect"] < 0.45 and features["left"] < 0.2 and features["right"] > 0.35:
                    return "1", max(confidence, 0.45 + 0.4 * (0.5 - features["left"]))
                if features["left"] > 0.32 and features["center"] < 0.35:
                    return "5", max(confidence, 0.45 + 0.4 * features["left"])
            if label == "5" and confidence < 0.3 and features["left"] < 0.18 and features["right"] > 0.4:
                return "3", max(confidence, 0.45 + 0.4 * features["right"])
            if label == "1" and confidence < 0.3 and features["aspect"] > 0.55:
                return "7", max(confidence, 0.4 + 0.3 * features["right"])

        if label.isdigit():
            self._ensure_digit_processor()
            if self.digit_processor is not None and confidence < 0.6:
                try:
                    alt_label, alt_conf = self.digit_processor.predict_single_digit(patch)
                    if alt_label is not None:
                        alt_label = str(alt_label)
                        if alt_conf > confidence + 0.1:
                            return alt_label, float(alt_conf)
                except Exception:
                    pass

        if label.isdigit() and self.digit_model is not None:
            try:
                digit_idx_arr, probs = self.digit_model.predict(patch)
                digit_idx = int(digit_idx_arr[0])
                digit_conf = float(probs[0][digit_idx])
                digit_label = str(digit_idx)
                improved = (
                    digit_label != label and digit_conf > confidence + 0.1
                ) or (digit_conf >= 0.55 and digit_conf > confidence)
                if improved:
                    return digit_label, digit_conf
            except Exception:
                pass

        return label, confidence

    def _run_digit_fallback(
        self, image: np.ndarray
    ) -> Tuple[str, List[Tuple[str, float]], List[np.ndarray]]:
        self._ensure_digit_processor()
        if self.digit_processor is None:
            return "", [], []
        try:
            sequence, preds, images = self.digit_processor.process_multi_digit_number(
                image, model_name="cnn", segmentation_method="contours"
            )
            return sequence, preds, images
        except Exception as exc:  # pragma: no cover
            print(f"Warning: digit fallback failed: {exc}")
            return "", [], []

    def recognise(
        self,
        image: np.ndarray,
        segmentation: str = "hybrid",
        model: str | None = None,
    ) -> Tuple[str, List[Tuple[str, float]], List[np.ndarray], str]:
        """Segment the image, classify each symbol, and return recognised string."""
        model_key = self._resolve_model_key(model)
        # Handle external MER integration as a special case (bypass our segmentation)
        if model_key == "mer":
            if not _MER_AVAILABLE:
                raise ValueError("MER integration not available in this build.")
            if self._mer is None:
                self._mer = MERSolverAdapter()  # type: ignore[arg-type]
            recognised, predictions, glyphs = self._mer.recognize(image)
            return recognised, predictions, glyphs, "mer"
        if model_key not in self.models:
            raise ValueError(f"Model '{model_key}' not available. Loaded: {list(self.models)}")

        predictor = self.models[model_key]
        method_key = self._normalise_segmentation_method(segmentation)
        if method_key == "best":
            chosen_method, patches, predictions, glyphs = self._select_segmentation(image, predictor)
            segmentation_used = f"auto->{chosen_method}"
        else:
            patches = self.segmenter.segment(image, method_key)
            if not patches:
                return "", [], [], method_key
            predictions, glyphs, _ = self._classify_patches(patches, predictor)
            segmentation_used = method_key

        if not patches:
            return "", [], [], segmentation_used

        recognised = "".join(symbol for symbol, _ in predictions)

        avg_conf = float(np.mean([conf for _, conf in predictions])) if predictions else 0.0
        digits_only = predictions and all(symbol.isdigit() for symbol, _ in predictions)
        if digits_only and avg_conf < 0.5:
            fallback_seq, fallback_preds, fallback_imgs = self._run_digit_fallback(image)
            if fallback_seq and len(fallback_preds) == len(predictions):
                recognised = fallback_seq
                predictions = [(lbl, float(conf)) for lbl, conf in fallback_preds]
                glyphs = [img.astype(np.uint8) if img.dtype != np.uint8 else img for img in fallback_imgs]
                segmentation_used = f"{segmentation_used}+digits"

        return recognised, predictions, glyphs, segmentation_used
