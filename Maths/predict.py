import json
import os
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np

# Resolve paths relative to this file so running from the project root
# or from within the Maths folder both work.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL = os.path.join(_HERE, "models", "math_symbol_model.h5")
_DEFAULT_LABELS = os.path.join(_HERE, "models", "labels.json")


@lru_cache(maxsize=1)
def _load(model_path: str, labels_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    try:
        from tensorflow.keras.models import load_model  # Lazy import to speed GUI startup
    except Exception as e:
        raise RuntimeError("TensorFlow/Keras is required for prediction. Install with: pip install tensorflow") from e
    model = load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)["idx_to_label"]
    return model, labels


def _shape_features(img: np.ndarray):
    """Compute simple geometric features from a 28x28 grayscale image [0,1]."""
    arr = (img < 0.8).astype(np.uint8)  # black strokes as 1s
    if arr.max() == 0:
        return {
            "ar": 1.0,
            "h_over_w": 1.0,
            "mid_h_ratio": 0.0,
            "mid_v_ratio": 0.0,
            "density": 0.0,
        }
    ys, xs = np.where(arr > 0)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    h = max(1, y2 - y1 + 1)
    w = max(1, x2 - x1 + 1)
    ar = w / float(h)
    density = arr.sum() / float(arr.shape[0] * arr.shape[1])
    # central bands
    mid_h_band = arr[arr.shape[0]//2 - 2: arr.shape[0]//2 + 3, :]
    mid_v_band = arr[:, arr.shape[1]//2 - 2: arr.shape[1]//2 + 3]
    mid_h_ratio = mid_h_band.mean()
    mid_v_ratio = mid_v_band.mean()
    return {
        "ar": ar,
        "h_over_w": h / float(w),
        "mid_h_ratio": float(mid_h_ratio),
        "mid_v_ratio": float(mid_v_ratio),
        "density": float(density),
    }


def _heuristic_fix(label: str, top2: List[Tuple[str, float]], img: np.ndarray) -> str:
    f = _shape_features(img)
    ar = f["ar"]
    mid_h = f["mid_h_ratio"]
    mid_v = f["mid_v_ratio"]
    density = f["density"]

    def choose(target: str, fallback: str = None):
        for l, _p in top2:
            if l == target:
                return target
        return fallback or label

    # Very thin horizontally -> minus
    if ar > 2.0 and mid_v < 0.12 and mid_h > 0.10:
        return choose('-', '-')

    # Very thin vertically with solid center line -> likely '1'
    # (Parentheses are thin but have low vertical-center ink.)
    if ar < 0.45 and mid_v >= 0.10:
        return choose('1', '1')

    # Avoid aggressively relabeling as '+' — this was
    # causing many digits (e.g., 9/5) to flip to plus.
    # Only keep '+' if the model already predicted it.
    if label == '+':
        return '+'

    # Detect two-lobed "8" using the row density profile: an "8" usually
    # has strong strokes in the top and bottom halves with a thinner waist.
    try:
        arr = (img < 0.8).astype(np.uint8)

        # Row profile to check "waist"
        prof = arr.sum(axis=1)
        n = prof.shape[0]
        top_max = float(prof[: n // 2].max()) if n > 0 else 0.0
        bot_max = float(prof[n // 2 :].max()) if n > 0 else 0.0
        mid_band = prof[max(0, n // 2 - 2) : min(n, n // 2 + 3)]
        mid_val = float(mid_band.mean()) if mid_band.size else 0.0
        has_two_lobes = (
            top_max >= 3 and bot_max >= 3 and mid_val < 0.6 * min(top_max, bot_max)
        )

        # Count enclosed background components (holes). 0 has 1; 8 has 2.
        from collections import deque

        bg = (1 - arr).astype(np.uint8)
        h, w = bg.shape
        visited = np.zeros_like(bg, dtype=bool)
        q = deque()
        # seed with border background
        for x in range(w):
            if bg[0, x] and not visited[0, x]:
                q.append((0, x)); visited[0, x] = True
            if bg[h - 1, x] and not visited[h - 1, x]:
                q.append((h - 1, x)); visited[h - 1, x] = True
        for y in range(h):
            if bg[y, 0] and not visited[y, 0]:
                q.append((y, 0)); visited[y, 0] = True
            if bg[y, w - 1] and not visited[y, w - 1]:
                q.append((y, w - 1)); visited[y, w - 1] = True
        # BFS to mark external background
        while q:
            y, x = q.popleft()
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

        internal = (bg.astype(bool) & (~visited))
        seen = np.zeros_like(internal, dtype=bool)
        holes = 0
        for yy in range(h):
            for xx in range(w):
                if internal[yy, xx] and not seen[yy, xx]:
                    holes += 1
                    q.append((yy, xx))
                    seen[yy, xx] = True
                    while q:
                        y, x = q.popleft()
                        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and internal[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
    except Exception:
        has_two_lobes = False
        holes = 0

    # Pull 8/0 confidences when available (top2 only)
    p = {l: pr for (l, pr) in top2}
    p8 = p.get('8')
    p0 = p.get('0')

    is_eight_shape = has_two_lobes and holes >= 2
    is_zero_shape = (holes <= 1) and (not has_two_lobes)

    # If model leans to 0/9/6 but the shape clearly matches an '8',
    # prefer 8. Use probabilities (if present) to avoid overriding a
    # much stronger '0'.
    if label in {"0", "9", "6"} and is_eight_shape:
        # Be conservative: only flip to '8' when the model already
        # assigns non-trivial probability to '8'. This avoids turning
        # clean zeros into eights.
        if (p8 is not None) and (p0 is None or p8 >= 0.9 * p0):
            return choose("8", label)

    # If predicted 8 but the shape doesn't look like an '8', only flip
    # to 0 when '0' is reasonably competitive.
    if label == "8" and not is_eight_shape:
        # Flip to '0' when the geometry doesn't look like an eight
        # and '0' is at least somewhat competitive, or the hole count
        # clearly indicates a single interior.
        if (p0 is not None and p8 is not None and p0 >= 0.7 * p8) or is_zero_shape:
            return choose("0", label)

    # Parenthesis vs '1'/'4': tall+thin with low center vertical ink,
    # and stroke biased to one side suggests '(' or ')'.
    if label in {"1", "4"}:
        try:
            arr2 = (img < 0.8).astype(np.uint8)
            h2, w2 = arr2.shape
            left_band = arr2[:, : max(1, w2 // 3)]
            right_band = arr2[:, w2 - max(1, w2 // 3) :]
            left_d = float(left_band.mean())
            right_d = float(right_band.mean())
            if ar < 0.85 and density < 0.30 and mid_v < 0.16:
                if left_d > right_d * 1.25:
                    return choose("(", "(")
                if right_d > left_d * 1.25:
                    return choose(")", ")")
        except Exception:
            pass

    # Keep confident loop-like predictions as-is
    if label in {"0", "8", "9", "6"} and 0.7 <= ar <= 1.4 and density > 0.08:
        return label

    return label

def _paren_side(img: np.ndarray) -> Optional[str]:
    """Return 'left' if stroke density is heavier on the left third,
    'right' if heavier on the right third, else None. Requires a tall,
    thin, low-center-ink shape typical of parentheses.
    """
    f = _shape_features(img)
    ar = f["ar"]
    mid_v = f["mid_v_ratio"]
    density = f["density"]
    if not (ar < 0.85 and density < 0.30 and mid_v < 0.16):
        return None
    arr = (img < 0.8).astype(np.uint8)
    h, w = arr.shape
    left_d = float(arr[:, : max(1, w // 3)].mean())
    right_d = float(arr[:, w - max(1, w // 3) :].mean())
    if left_d > right_d * 1.2:
        return "left"
    if right_d > left_d * 1.2:
        return "right"
    return None

def _rebalance_parentheses(labels: List[str], images: np.ndarray,
                            boxes: Optional[List[Tuple[int,int,int,int]]] = None) -> None:
    """In-place: if parentheses are unbalanced, flip a likely '1'/'4'
    into the missing '(' or ')', based on shape and left/right position.
    """
    if not labels:
        return
    open_n = labels.count('(')
    close_n = labels.count(')')
    if open_n == close_n:
        return

    candidates = [i for i, l in enumerate(labels) if l in {"1", "4"}]
    if not candidates:
        return

    def score(i: int, want: str) -> float:
        side = _paren_side(images[i])
        side_ok = (side == ("left" if want == "(" else "right")) if side is not None else False
        pos_bonus = 0.0
        if boxes is not None and len(boxes) == len(labels):
            xs = [b[0] for b in boxes]
            max_x = max(xs) if xs else 1
            x = boxes[i][0]
            if want == "(":
                pos_bonus = max(0.0, 1.0 - (x / float(max_x + 1)))
            else:
                pos_bonus = float(x) / float(max_x + 1)
        base = 1.0 if side_ok else (0.3 if side is not None else 0.0)
        return base + 0.2 * pos_bonus

    if open_n < close_n:
        # Need an opening '('
        best = max(candidates, key=lambda i: score(i, "("))
        if score(best, "(") > 0.0:
            labels[best] = '('
    else:
        # Need a closing ')'
        best = max(candidates, key=lambda i: score(i, ")"))
        if score(best, ")") > 0.0:
            labels[best] = ')'


def predict_symbols(
    symbol_images: List[np.ndarray],
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    model_path: str = _DEFAULT_MODEL,
    labels_path: str = _DEFAULT_LABELS,
) -> Tuple[str, List[str]]:
    """
    Predict labels for segmented 28x28 grayscale symbol images (values in [0,1]).
    Returns the reconstructed expression string and per-symbol raw labels.
    """
    if not symbol_images:
        return "", []

    model, labels = _load(model_path, labels_path)

    X = np.stack([img for img in symbol_images], axis=0)
    if X.ndim == 3:
        X = X[..., np.newaxis]

    probs = model.predict(X, verbose=0)
    idxs = np.argmax(probs, axis=1)
    raw_labels: List[str] = []
    for k, i in enumerate(idxs):
        top = i
        # collect top2 labels for heuristics
        top2_idx = np.argsort(probs[k])[-2:][::-1]
        top2 = [(labels[j], float(probs[k][j])) for j in top2_idx]
        lab = labels[top]
        # Apply shape-aware heuristic correction
        lab = _heuristic_fix(lab, top2, X[k, ..., 0])
        raw_labels.append(lab)

    # Merge split components that together form an '8' (two zeros stacked)
    if boxes is not None and len(boxes) == len(raw_labels):
        i = 0
        while i < len(raw_labels) - 1:
            if raw_labels[i] == '0' and raw_labels[i+1] == '0':
                x1, y1, w1, h1 = boxes[i]
                x2, y2, w2, h2 = boxes[i+1]
                # Geometry cues for stacked loops:
                # - Horizontal centers close OR decent horizontal overlap
                # - Combined height much taller than either component
                # These are more tolerant than the previous strict overlap rule.
                cx1 = x1 + w1 / 2.0
                cx2 = x2 + w2 / 2.0
                center_dx = abs(cx1 - cx2)
                x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                min_w = max(1, min(w1, w2))
                overlap_ratio = x_overlap / float(min_w)
                union_h = max(y1+h1, y2+h2) - min(y1, y2)
                max_h = max(h1, h2)
                centers_close = center_dx <= 0.4 * max(w1, w2)
                tall_enough = union_h >= 1.5 * max_h
                if (tall_enough and (overlap_ratio >= 0.25 or centers_close)):
                    raw_labels[i] = '8'
                    del raw_labels[i+1]
                    # merge the boxes for downstream use
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1+w1, x2+w2) - nx
                    nh = union_h
                    boxes[i] = (nx, ny, nw, nh)
                    del boxes[i+1]
                    continue
            i += 1

    # Try to rebalance parentheses by flipping likely '1'/'4' glyphs
    try:
        imgs_flat = X[..., 0]
        _rebalance_parentheses(raw_labels, imgs_flat, boxes)
    except Exception:
        pass

    # Map dataset labels to expression characters
    mapped = []
    for lab in raw_labels:
        if lab == "times":
            ch = "*"
        elif lab == "div" or lab == "forward_slash":
            ch = "/"
        else:
            ch = lab

        # Collapse consecutive operators that can arise from over-segmentation
        if mapped and ch in "+-*/" and mapped[-1] in "+-*/":
            # keep the latest operator; replace previous
            mapped[-1] = ch
        else:
            mapped.append(ch)

    # Heuristic: insert implicit multiplication between digit and '(' or ')' and digit/ '('
    expr_parts: List[str] = []
    for i, ch in enumerate(mapped):
        if i > 0:
            prev = mapped[i - 1]
            if ((prev.isdigit() and ch == "(") or
                (prev == ")" and (ch.isdigit() or ch == "("))):
                expr_parts.append("*")
        expr_parts.append(ch)

    expression = "".join(expr_parts)
    return expression, raw_labels


def predict_symbols_detailed(
    symbol_images: List[np.ndarray],
    boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    model_path: str = _DEFAULT_MODEL,
    labels_path: str = _DEFAULT_LABELS,
) -> Tuple[str, List[str], List[float], np.ndarray]:
    """Like predict_symbols, but also returns per-symbol top confidence and raw probability matrix.

    Returns:
      expression, raw_labels, confidences, probs
    """
    if not symbol_images:
        return "", [], [], np.zeros((0, 0))

    model, labels = _load(model_path, labels_path)

    X = np.stack([img for img in symbol_images], axis=0)
    if X.ndim == 3:
        X = X[..., np.newaxis]

    probs = model.predict(X, verbose=0)
    idxs = np.argmax(probs, axis=1)
    raw_labels: List[str] = []
    confs: List[float] = []
    for k, i in enumerate(idxs):
        top = i
        top2_idx = np.argsort(probs[k])[-2:][::-1]
        top2 = [(labels[j], float(probs[k][j])) for j in top2_idx]
        lab = labels[top]
        lab = _heuristic_fix(lab, top2, X[k, ..., 0])
        raw_labels.append(lab)
        confs.append(float(probs[k][i]))

    # Merge split components that together form an '8' using geometry when boxes given
    if boxes is not None and len(boxes) == len(raw_labels):
        i = 0
        while i < len(raw_labels) - 1:
            if raw_labels[i] == '0' and raw_labels[i+1] == '0':
                x1, y1, w1, h1 = boxes[i]
                x2, y2, w2, h2 = boxes[i+1]
                cx1 = x1 + w1 / 2.0
                cx2 = x2 + w2 / 2.0
                center_dx = abs(cx1 - cx2)
                x_overlap = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                min_w = max(1, min(w1, w2))
                overlap_ratio = x_overlap / float(min_w)
                union_h = max(y1+h1, y2+h2) - min(y1, y2)
                max_h = max(h1, h2)
                centers_close = center_dx <= 0.4 * max(w1, w2)
                tall_enough = union_h >= 1.5 * max_h
                if (tall_enough and (overlap_ratio >= 0.25 or centers_close)):
                    raw_labels[i] = '8'
                    confs[i] = float((confs[i] + confs[i+1]) / 2.0) if i+1 < len(confs) else confs[i]
                    del raw_labels[i+1]
                    if i+1 < len(confs):
                        del confs[i+1]
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1+w1, x2+w2) - nx
                    nh = union_h
                    boxes[i] = (nx, ny, nw, nh)
                    del boxes[i+1]
                    continue
            i += 1

    # Try to rebalance parentheses by flipping likely '1'/'4' glyphs
    try:
        imgs_flat = X[..., 0]
        _rebalance_parentheses(raw_labels, imgs_flat, boxes)
    except Exception:
        pass

    mapped = []
    for lab in raw_labels:
        if lab == "times":
            ch = "*"
        elif lab == "div" or lab == "forward_slash":
            ch = "/"
        else:
            ch = lab

        if mapped and ch in "+-*/" and mapped[-1] in "+-*/":
            mapped[-1] = ch
        else:
            mapped.append(ch)

    expr_parts: List[str] = []
    for i, ch in enumerate(mapped):
        if i > 0:
            prev = mapped[i - 1]
            if ((prev.isdigit() and ch == "(") or
                (prev == ")" and (ch.isdigit() or ch == "("))):
                expr_parts.append("*")
        expr_parts.append(ch)

    expression = "".join(expr_parts)
    return expression, raw_labels, confs, probs
