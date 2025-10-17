from __future__ import annotations

from typing import List, Tuple
import os
import numpy as np
import cv2

from .preprocess import preprocess_single, normalize_background, binarize
from .segment import cc_boxes, watershed_boxes
from .model_knn import load_model as load_knn


class LettersPipeline:
    def __init__(self) -> None:
        self.model = None
        self.label_map = None
        self._ensure_model()

    def _ensure_model(self):
        if self.model is not None:
            return
        # Load on demand. User can retrain with train_letters_new.py
        model_path = os.path.join('models_new', 'letters_knn.pkl')
        if os.path.exists(model_path):
            self.model, self.label_map = load_knn(model_path)
        else:
            # Lazy fallback: small on-the-fly training (few samples) to start
            from .model_knn import train_knn, save_model
            self.model, self.label_map = train_knn(max_samples=8000, n_neighbors=3)
            save_model(self.model, self.label_map)

    def _predict_patch(self, patch: np.ndarray) -> Tuple[str, float]:
        # kNN returns label; we approximate confidence via inverse rank distance
        arr = patch.astype(np.float32)
        X = arr.reshape(1, -1)
        pred = int(self.model.predict(X)[0])
        # Approximate confidence using neighbor votes
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            conf = float(proba[pred - 1]) if pred - 1 < len(proba) else 1.0
        else:
            conf = 1.0
        label = self.label_map.get(pred, '?')
        return label, conf

    def _segment(self, image: np.ndarray, method: str) -> List[np.ndarray]:
        gray = normalize_background(image)
        bin_img = binarize(gray)
        if method == 'watershed':
            boxes = watershed_boxes(gray)
        elif method == 'cc':
            boxes = cc_boxes(255 - bin_img)
        else:
            boxes = cc_boxes(255 - bin_img)
        if not boxes:
            return [preprocess_single(image)]
        patches: List[np.ndarray] = []
        for x, y, w, h in boxes:
            crop = gray[y:y+h, x:x+w]
            patches.append(preprocess_single(crop))
        return patches

    def recognize(self, image: np.ndarray, method: str = 'cc') -> Tuple[str, List[Tuple[str, float]], List[np.ndarray]]:
        self._ensure_model()
        patches = self._segment(image, method)
        preds: List[Tuple[str, float]] = []
        for p in patches:
            label, conf = self._predict_patch(p.reshape(28, 28))
            preds.append((label, conf))
        text = ''.join(l for l, _ in preds)
        # Return visualization-ready 8-bit patches
        viz = [(p.squeeze() * 255).astype('uint8') for p in patches]
        return text, preds, viz

