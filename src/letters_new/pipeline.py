from __future__ import annotations

from typing import List, Tuple
import os
import numpy as np
import cv2

from .preprocess import preprocess_single, normalize_background, binarize
from .segment import watershed_boxes
from .model_knn import load_model as load_knn
from models import CNNModel
import json
from .model_knn import train_knn_byclass, save_model


class LettersPipeline:
    def __init__(self) -> None:
        self.model = None            # k-NN or CNN
        self.label_map = None        # index -> char
        self._cnn = None             # Optional CNNModel
        self._preferred: str = 'auto'  # 'auto' | 'cnn' | 'knn'
        self._allowed: set[str] | None = None  # Optional restriction of output characters
        self._last_used_model: str | None = None
        self._cnn_input_size: int = 28
        self._knn_input_size: int = 28
        self._ensure_model()

    def _ensure_model(self):
        if self.model is not None:
            return
        # Load on demand. User can retrain with train_letters_new.py
        # Try CNN first (best accuracy)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proj_root = os.path.dirname(root)
        model_dirs = [os.path.join(proj_root, 'models'), os.path.join(root, 'models')]
        cnn_path = None
        map_path = None
        for d in model_dirs:
            if not d:
                continue
            p = os.path.join(d, 'cnn_model_nist_by_class.h5')
            m = os.path.join(d, 'label_mapping_nist_by_class.json')
            if os.path.exists(p) and os.path.exists(m):
                cnn_path, map_path = p, m
                break
        if cnn_path and map_path:
            try:
                self._cnn = CNNModel(num_classes=52, use_augmentation=False, architecture="letters")
                self._cnn.load_model(cnn_path)
                try:
                    input_shape = getattr(self._cnn.model, "input_shape", None)
                    if input_shape and len(input_shape) >= 3 and input_shape[1]:
                        self._cnn_input_size = int(input_shape[1])
                except Exception:
                    self._cnn_input_size = 28
                with open(map_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                self.label_map = {int(k): str(v) for k, v in mapping.items()}
                return
            except Exception:
                self._cnn = None

        # Fallback to k-NN
        model_path = os.path.join('models_new', 'letters_byclass_knn.pkl')
        if os.path.exists(model_path):
            self.model, self.label_map = load_knn(model_path)
            self._update_knn_input_size()
        else:
            self.model, self.label_map = train_knn_byclass(max_per_class=400, max_total=20000, n_neighbors=3)
            save_model(self.model, self.label_map)
            self._update_knn_input_size()

    def _update_knn_input_size(self) -> None:
        """Infer the expected patch side length for the k-NN model."""
        try:
            n_features = int(getattr(self.model, "n_features_in_", 0))
            if n_features > 0:
                side = int(round(np.sqrt(n_features)))
                if side * side == n_features:
                    self._knn_input_size = max(1, side)
        except Exception:
            self._knn_input_size = 28

    def _predict_patch(self, patch: np.ndarray) -> Tuple[str, float]:
        """Predict with optional test-time augmentation for CNN to improve robustness."""
        arr = patch.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        if arr.ndim == 1:
            side = int(round(np.sqrt(arr.size)))
            arr = arr.reshape(side, side)
        base = np.clip(arr, 0.0, 1.0)
        use_cnn = (self._preferred == 'cnn') or (self._preferred == 'auto' and self._cnn is not None)
        cnn_side = self._cnn_input_size if self._cnn is not None else base.shape[0]
        knn_side = max(1, self._knn_input_size)
        if use_cnn and self._cnn is not None and self.label_map:
            canvas = base
            if canvas.shape[0] != cnn_side:
                canvas = cv2.resize(canvas, (cnn_side, cnn_side), interpolation=cv2.INTER_CUBIC)
            variants = [canvas]
            try:
                h = w = cnn_side
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        M = np.float32([[1, 0, dx], [0, 1, dy]])
                        variants.append(cv2.warpAffine(canvas, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=1.0))
                for ang in (-5, 5):
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
                    variants.append(cv2.warpAffine(canvas, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=1.0))
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                v_uint8 = (np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8)
                variants.append(cv2.dilate(v_uint8, k, iterations=1).astype(np.float32) / 255.0)
                variants.append(cv2.erode(v_uint8, k, iterations=1).astype(np.float32) / 255.0)
            except Exception:
                pass
            X = np.stack([v.reshape(cnn_side, cnn_side, 1) for v in variants], axis=0).astype(np.float32)
            X = np.clip(X, 0.0, 1.0)
            _, probs = self._cnn.predict(X)
            prob_vec = probs.mean(axis=0)
            if self._allowed is not None and self.label_map:
                mask = np.ones_like(prob_vec, dtype=bool)
                for idx, ch in self.label_map.items():
                    if ch not in self._allowed:
                        mask[int(idx)] = False
                if np.any(mask):
                    clipped = np.where(mask, prob_vec, -1.0)
                    idx = int(np.argmax(clipped))
                    label = self.label_map.get(idx, str(idx))
                    self._last_used_model = 'CNN'
                    conf = float(prob_vec[idx])
                    label, conf = self._q_override_heuristic(canvas, label, conf)
                    return label, conf
            idx = int(np.argmax(prob_vec))
            label = self.label_map.get(idx, str(idx))
            self._last_used_model = 'CNN'
            conf = float(prob_vec[idx])
            label, conf = self._q_override_heuristic(canvas, label, conf)
            return label, conf
        # k-NN fallback
        if self.model is None:
            raise RuntimeError("k-NN model is not available for letter recognition.")
        base_for_knn = base
        if base_for_knn.shape[0] != knn_side:
            base_for_knn = cv2.resize(base_for_knn, (knn_side, knn_side), interpolation=cv2.INTER_AREA)
        X = base_for_knn.reshape(1, -1)
        pred = int(self.model.predict(X)[0])
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            conf = float(max(proba))
        else:
            conf = 1.0
        label = self.label_map.get(pred, '?') if self.label_map else '?'
        self._last_used_model = 'kNN'
        label, conf = self._q_override_heuristic(base_for_knn, label, conf)
        return label, conf

    def _segment(self, image: np.ndarray, method: str) -> List[np.ndarray]:
        gray = normalize_background(image)
        bin_img = binarize(gray)
        m = method.lower().replace(' ', '_')
        target_size = self._cnn_input_size if self._cnn is not None else self._knn_input_size
        if m in ('auto', 'auto_selection'):
            methods = ['watershed']
            best = None
            for mm in methods:
                patches = self._segment(image, mm)
                if not patches:
                    continue
                confs = [self._predict_patch(p)[1] for p in patches]
                avg = float(np.mean(confs)) if confs else 0.0
                if best is None or avg > best[0]:
                    best = (avg, patches)
            return best[1] if best else []

        if m in ('watershed',):
            boxes = watershed_boxes(gray)
        else:
            # Default to watershed for letters segmentation
            boxes = watershed_boxes(gray)
        if not boxes:
            return [preprocess_single(image, target=target_size)]
        patches: List[np.ndarray] = []
        for x, y, w, h in boxes:
            crop = gray[y:y+h, x:x+w]
            patches.append(preprocess_single(crop, target=target_size))
        return patches

    def recognize(self, image: np.ndarray, method: str = 'watershed', model: str = 'auto') -> Tuple[str, List[Tuple[str, float]], List[np.ndarray]]:
        self._ensure_model()
        self.set_preferred_model(model)
        # Initialize last-used to help GUI display an accurate backend name
        if (self._preferred == 'cnn' or (self._preferred == 'auto' and self._cnn is not None)):
            self._last_used_model = 'CNN'
        else:
            self._last_used_model = 'kNN'
        patches = self._segment(image, method)
        preds: List[Tuple[str, float]] = []
        for p in patches:
            label, conf = self._predict_patch(p)
            preds.append((label, conf))
        text = ''.join(l for l, _ in preds)
        # Return visualization-ready 8-bit patches
        viz = [(np.clip(p.squeeze(), 0.0, 1.0) * 255).astype('uint8') for p in patches]
        return text, preds, viz

    # -------- Model control / discovery ---------
    def get_available_models(self) -> List[str]:
        # Prefer kNN first for default selection (more robust on drawings)
        names: List[str] = []
        if self.model is not None:
            names.append('kNN')
        if self._cnn is not None:
            names.append('CNN')
        if not names:
            names.append('kNN')
        return names

    def set_preferred_model(self, name: str) -> None:
        key = (name or 'auto').lower()
        if key in ('cnn', 'knn', 'auto'):
            self._preferred = key
        elif key == 'random_forest' or key == 'rf' or key == 'svm':
            # Not supported in letters_new pipeline; default to auto
            self._preferred = 'auto'
        else:
            self._preferred = 'auto'

    def last_used_model(self) -> str:
        """Return a human-readable name of the backend used for the last call.

        Defaults to 'kNN' if not set yet.
        """
        return self._last_used_model or ('CNN' if self._cnn is not None and self._preferred == 'cnn' else 'kNN')

    # -------- Allowed character control ---------
    def set_allowed_characters(self, mode: str | None) -> None:
        """Restrict outputs to a subset: 'uppercase', 'lowercase', or None for all.

        This only affects inference-time selection; it does not change the model.
        """
        mode = (mode or '').strip().lower()
        if not mode:
            self._allowed = None
            return
        if mode in ('upper', 'uppercase'):
            self._allowed = {chr(c) for c in range(ord('A'), ord('Z') + 1)}
        elif mode in ('lower', 'lowercase'):
            self._allowed = {chr(c) for c in range(ord('a'), ord('z') + 1)}
        else:
            self._allowed = None

    # --------- Heuristics ---------
    def _uppercase_only(self) -> bool:
        if self._allowed is None:
            return False
        # All allowed are uppercase letters
        return all('A' <= ch <= 'Z' for ch in self._allowed)

    def _q_override_heuristic(self, patch01: np.ndarray, label: str, conf: float) -> Tuple[str, float]:
        """If the glyph looks like an uppercase 'Q', override low-confidence mistakes.

        Trigger only when:
        - UI restricted to uppercase; and
        - Current label is one of {'W','V','O'} or very low-confidence (<0.45);
        - Patch exhibits high roundness and a diagonal tail in lower-right quadrant.
        """
        try:
            if not self._uppercase_only():
                return label, conf
            cand = label
            if cand not in {'W', 'V', 'O'} and conf >= 0.45:
                return label, conf
            # Prepare uint8 image
            g = patch01.astype(np.float32)
            if g.ndim == 3 and g.shape[-1] == 1:
                g = g.squeeze(-1)
            if g.ndim == 1:
                side = int(round(np.sqrt(g.size)))
                g = g.reshape(side, side)
            side = int(g.shape[0])
            img = (np.clip(g, 0.0, 1.0) * 255).astype(np.uint8)
            scale = max(1.0, side / 28.0)
            # Binary, black ink on white
            _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Roundness via contour circularity
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return label, conf
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            per = cv2.arcLength(c, True) + 1e-6
            circularity = 4.0 * np.pi * area / (per * per)
            if circularity < 0.45:
                return label, conf
            # Check for diagonal tail in lower-right quadrant using Hough
            edges = cv2.Canny(bw, 50, 150)
            h, w = edges.shape
            sub = edges[h//2:, w//2:]
            min_len = max(6, int(round(6 * scale)))
            max_gap = max(2, int(round(2 * scale)))
            lines = cv2.HoughLinesP(sub, 1, np.pi/180, threshold=10, minLineLength=min_len, maxLineGap=max_gap)
            tail_found = False
            if lines is not None:
                for x1, y1, x2, y2 in lines[:,0,:]:
                    dx, dy = x2 - x1, y2 - y1
                    if abs(dx) < 2 and abs(dy) < 2:
                        continue
                    angle = np.degrees(np.arctan2(dy, dx))
                    # Look for roughly diagonal line slanting down-right
                    if 15 <= angle <= 75 or -75 <= angle <= -15:
                        if np.hypot(dx, dy) >= max(8, int(round(8 * scale))):
                            tail_found = True
                            break
            if tail_found:
                # Promote to Q when we find a tail on a rounded glyph
                return 'Q', max(conf, 0.55)
            return label, conf
        except Exception:
            return label, conf
