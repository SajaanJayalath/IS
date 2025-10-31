import cv2
import numpy as np
from typing import List, Tuple


def cc_boxes(binary_white_bg: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Connected components with denoise and gentle closing to connect strokes.

    Expects black ink on white background.
    """
    inv = 255 - binary_white_bg  # foreground white
    # Denoise very small specks
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = np.zeros_like(bw)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 10:
            keep[labels == i] = 255
    # Gentle closing to bridge small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 20 or w < 3 or h < 3:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    boxes.sort(key=lambda b: b[0])
    return boxes


def watershed_boxes(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Watershed segmentation tuned for handwritten letters.

    Steps:
    - Pre-blur and compute morphological gradient to guide watershed ridges
    - Marker extraction via distance transform with adaptive threshold
    - Post-process boxes: filter tiny blobs, merge close fragments, split overly wide
    """
    # Normalize to uint8 grayscale
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    if g.max() <= 1.0:
        g = g * 255.0
    g = np.clip(g, 0, 255).astype(np.uint8)

    # Light denoise
    g_blur = cv2.GaussianBlur(g, (3, 3), 0)

    # Morphological gradient emphasizes edges for watershed input
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(g_blur, cv2.MORPH_GRADIENT, k3)

    # Binary mask (white foreground)
    _, thr = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k3, iterations=1)
    sure_bg = cv2.dilate(opening, k3, iterations=2)

    # Distance transform and adaptive foreground threshold
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    if dist.max() > 0:
        nz = dist[dist > 0]
        # Adaptive threshold between 0.3*max and 60th percentile
        thr_val = max(0.30 * dist.max(), float(np.percentile(nz, 60)))
    else:
        thr_val = 0
    sure_fg = (dist >= thr_val).astype(np.uint8) * 255
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers from connected components on sure foreground
    num, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed on gradient image for cleaner boundaries
    markers = cv2.watershed(cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR), markers)

    # Extract boxes, filter tiny components
    boxes: List[Tuple[int, int, int, int]] = []
    min_area = 25
    min_size = 6
    for m in range(2, markers.max() + 1):
        ys, xs = np.where(markers == m)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        w, h = x1 - x0, y1 - y0
        if w < min_size or h < min_size or (w * h) < min_area:
            continue
        # small margin expand within image bounds
        x0e = max(0, x0 - 1)
        y0e = max(0, y0 - 1)
        x1e = min(g.shape[1], x1 + 1)
        y1e = min(g.shape[0], y1 + 1)
        boxes.append((x0e, y0e, x1e - x0e, y1e - y0e))

    boxes.sort(key=lambda b: b[0])

    # Post-process: merge close fragments and split overly wide boxes
    from .segment import merge_close_boxes, split_wide_boxes as _split
    boxes = merge_close_boxes(boxes, gap=3)
    boxes = _split(g, boxes, aspect=1.35)
    boxes.sort(key=lambda b: b[0])
    return boxes


def projection_boxes(gray_or_binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Segment using a vertical projection profile.

    Accepts grayscale or binary image. Returns left-to-right boxes.
    """
    if gray_or_binary.ndim == 3:
        gray = cv2.cvtColor(gray_or_binary, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_binary.copy()

    # Produce binary with white foreground for easy counting
    if gray.max() <= 1:
        g = (gray * 255).astype(np.uint8)
    else:
        g = gray
    _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col = (thr > 0).astype(np.uint8)
    profile = col.sum(axis=0).astype(np.float32)
    if profile.size == 0:
        return []

    # Smooth profile
    win = max(3, int(0.03 * len(profile)))
    if win % 2 == 0:
        win += 1
    kernel = np.ones((win,), dtype=np.float32) / float(win)
    sm = np.convolve(profile, kernel, mode='same')

    # Find valleys
    w = len(sm)
    margin = max(2, int(0.03 * w))
    thr_val = 0.25 * sm.max()
    cuts: List[int] = []
    i = margin
    while i < w - margin:
        if sm[i] < thr_val:
            s = i
            while i < w - margin and sm[i] < thr_val:
                i += 1
            e = i - 1
            cuts.append(int((s + e) // 2))
        else:
            i += 1

    # Build segments with minimum width
    segments: List[Tuple[int, int]] = []
    last = 0
    minw = max(10, int(0.12 * w))
    for c in cuts:
        if c - last >= minw:
            segments.append((last, c))
            last = c
    if w - last >= minw:
        segments.append((last, w))

    if not segments:
        # Single box spanning foreground rows
        ys = np.where(np.any(col > 0, axis=1))[0]
        if ys.size == 0:
            return []
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return [(0, y0, w, y1 - y0)]

    # Convert to bounding boxes constrained to foreground rows
    boxes: List[Tuple[int, int, int, int]] = []
    ys = np.where(np.any(col > 0, axis=1))[0]
    if ys.size == 0:
        return []
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    for x0, x1 in segments:
        boxes.append((int(x0), y0, int(x1 - x0), y1 - y0))
    boxes.sort(key=lambda b: b[0])
    return boxes


def split_wide_boxes(gray: np.ndarray, boxes: List[Tuple[int, int, int, int]], aspect: float = 1.4) -> List[Tuple[int, int, int, int]]:
    """Split overly wide boxes using local projection inside each box."""
    out: List[Tuple[int, int, int, int]] = []
    for (x, y, w, h) in boxes:
        if h == 0:
            continue
        if w / float(h) <= aspect or w < 20:
            out.append((x, y, w, h))
            continue
        roi = gray[y:y + h, x:x + w]
        sub_boxes = projection_boxes(roi)
        if len(sub_boxes) <= 1:
            out.append((x, y, w, h))
        else:
            for sx, sy, sw, sh in sub_boxes:
                out.append((x + sx, y + sy, sw, sh))
    out.sort(key=lambda b: b[0])
    return out


def merge_close_boxes(boxes: List[Tuple[int, int, int, int]], gap: int = 4) -> List[Tuple[int, int, int, int]]:
    """Merge boxes that are very close horizontally (likely fragmented strokes)."""
    if not boxes:
        return []
    merged: List[Tuple[int, int, int, int]] = []
    cur = list(boxes[0])
    for (x, y, w, h) in boxes[1:]:
        cx, cy, cw, ch = cur
        if x <= cx + cw + gap and abs((y + h/2) - (cy + ch/2)) < max(h, ch):
            nx = min(cx, x)
            ny = min(cy, y)
            nx2 = max(cx + cw, x + w)
            ny2 = max(cy + ch, y + h)
            cur = [nx, ny, nx2 - nx, ny2 - ny]
        else:
            merged.append((int(cur[0]), int(cur[1]), int(cur[2]), int(cur[3])))
            cur = [x, y, w, h]
    merged.append((int(cur[0]), int(cur[1]), int(cur[2]), int(cur[3])))
    merged.sort(key=lambda b: b[0])
    return merged
