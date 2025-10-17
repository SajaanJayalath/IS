import cv2
import numpy as np
from typing import List, Tuple


def cc_boxes(binary_white_bg: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # Expect black ink on white
    inv = 255 - binary_white_bg
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 20 or w < 3 or h < 3:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes


def watershed_boxes(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # Quick-and-dirty watershed for touching letters
    if gray.max() <= 1:
        g = (gray * 255).astype(np.uint8)
    else:
        g = gray
    _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    num, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), markers)
    boxes: List[Tuple[int, int, int, int]] = []
    for m in range(2, markers.max() + 1):
        ys, xs = np.where(markers == m)
        if ys.size == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    boxes.sort(key=lambda b: b[0])
    return boxes

