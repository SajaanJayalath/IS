"""
Image Segmentation Module for Handwritten Number Recognition System
Implements digit separation and multi-digit number processing
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from image_preprocessing import ImagePreprocessor
class ImageSegmenter:
    """
    Image segmentation for separating individual digits from multi-digit numbers
    """
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        
    def find_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """Find contours in the image"""
        # Ensure binary image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold - use inverted threshold for black digits on white background
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            binary = ((1 - image) * 255).astype(np.uint8)
        
        # Light morphological erosion to break thin joints between close digits
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=0)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def filter_contours(self, contours: List[np.ndarray], 
                       min_area: int = 50, max_area: int = 50000,
                       min_aspect_ratio: float = 0.1, max_aspect_ratio: float = 5.0) -> List[np.ndarray]:
        """Filter contours based on size and aspect ratio"""
        filtered_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Filter by aspect ratio
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
                
            filtered_contours.append(contour)
        
        return filtered_contours
    
    def get_bounding_boxes(self, contours: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """Get bounding boxes for contours"""
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
        return bounding_boxes
    
    def sort_bounding_boxes_left_to_right(self, bounding_boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Sort bounding boxes from left to right"""
        return sorted(bounding_boxes, key=lambda box: box[0])  # Sort by x coordinate
    
    def extract_digits(self, image: np.ndarray, bounding_boxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Extract individual digit images from bounding boxes"""
        digits = []
        
        for x, y, w, h in bounding_boxes:
            # Extract digit region
            digit_roi = image[y:y+h, x:x+w]
            
            # Add padding to make it square
            max_dim = max(w, h)
            padded_size = int(max_dim * 1.2)  # Add 20% padding
            
            # Create padded image
            padded_digit = np.zeros((padded_size, padded_size), dtype=image.dtype)
            
            # Center the digit in the padded image
            y_offset = (padded_size - h) // 2
            x_offset = (padded_size - w) // 2
            padded_digit[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi
            
            # Resize to 28x28 for MNIST compatibility
            resized_digit = cv2.resize(padded_digit, (28, 28), interpolation=cv2.INTER_AREA)
            
            digits.append(resized_digit)
        
        return digits
    def _to_binary_inv(self, image: np.ndarray) -> np.ndarray:
        """Convert to inverted binary (digits as white)."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            binary = ((1 - image) * 255).astype(np.uint8)
        return binary
    def _find_best_valley(self, profile: np.ndarray, start: int, end: int, window: int = 6, drop_ratio: float = 0.95) -> Optional[int]:
        if profile.size == 0 or end - start <= 0:
            return None
        start = max(1, start)
        end = min(len(profile) - 1, end)
        best_idx = None
        best_ratio = drop_ratio
        for idx in range(start, end):
            left = profile[max(0, idx - window):idx]
            right = profile[idx + 1:min(len(profile), idx + window + 1)]
            if left.size == 0 or right.size == 0:
                continue
            local_peak = max(left.max(), right.max(), 1e-6)
            ratio = profile[idx] / local_peak
            if ratio < best_ratio:
                best_ratio = ratio
                best_idx = idx
        return best_idx

    def split_box_by_vertical_projection(self, image: np.ndarray, box: Tuple[int, int, int, int],
                                         smooth_window: int = 5, valley_ratio: float = 0.25,
                                         min_width: int = 18) -> List[Tuple[int, int, int, int]]:
        """
        If a bounding box likely contains multiple touching digits, split it using
        vertical projection (column histogram) to find a valley between digits.
        Returns one or more boxes.
        """
        x, y, w, h = box
        if w < min_width:
            return [box]
        binary = self._to_binary_inv(image)
        roi = binary[y:y+h, x:x+w]
        if roi.size == 0:
            return [box]
        col_sums = np.sum(roi > 0, axis=0).astype(np.float32)
        if col_sums.size == 0 or np.max(col_sums) == 0:
            return [box]
        # Smooth the profile to reduce noise
        kernel = np.ones((smooth_window,), dtype=np.float32) / float(smooth_window)
        smoothed = np.convolve(col_sums, kernel, mode='same')
        maxv = smoothed.max()
        threshold = valley_ratio * maxv
        # Candidate valley indices away from edges
        margin = max(2, int(0.1 * w))
        candidates = [i for i, v in enumerate(smoothed) if v < threshold and margin < i < (w - margin)]
        if not candidates:
            if maxv > 0:
                min_idx = int(np.argmin(smoothed))
                min_ratio = smoothed[min_idx] / maxv if maxv else 1.0
                if margin < min_idx < (w - margin) and min_ratio < 0.92:
                    left_w = min_idx
                    right_w = w - min_idx
                    min_part = max(10, int(0.25 * w))
                    if left_w >= min_part and right_w >= min_part:
                        tentative_boxes = [(x, y, left_w, h), (x + min_idx, y, right_w, h)]
                        split_results: List[Tuple[int, int, int, int]] = []
                        for b in tentative_boxes:
                            aspect = b[2] / max(1, b[3])
                            if b[2] > 2 * min_width and aspect > 1.4:
                                split_results.extend(self.split_box_by_vertical_projection(image, b, smooth_window, valley_ratio, min_width))
                            else:
                                split_results.append(b)
                        return split_results
            return [box]
        # Choose the longest contiguous valley and split at its center
        best_start = candidates[0]
        best_end = candidates[0]
        run_start = candidates[0]
        prev = candidates[0]
        best_len = 1
        for idx in candidates[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                run_len = prev - run_start + 1
                if run_len > best_len:
                    best_len = run_len
                    best_start, best_end = run_start, prev
                run_start = idx
                prev = idx
        # finalize last run
        run_len = prev - run_start + 1
        if run_len > best_len:
            best_len = run_len
            best_start, best_end = run_start, prev
        split_at = int((best_start + best_end) / 2)
        left_w = split_at
        right_w = w - split_at
        # Ensure both parts are reasonably wide
        min_part = max(10, int(0.25 * w))
        if left_w < min_part or right_w < min_part:
            return [box]
        left = (x, y, left_w, h)
        right = (x + split_at, y, right_w, h)
        result: List[Tuple[int, int, int, int]] = []
        for b in (left, right):
            # Recursively split if still very wide
            aspect = b[2] / max(1, b[3])
            if b[2] > 2 * min_width and aspect > 1.4:
                result.extend(self.split_box_by_vertical_projection(image, b, smooth_window, valley_ratio, min_width))
            else:
                result.append(b)
        return result

    def split_wide_boxes(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                         aspect_threshold: float = 1.35) -> List[Tuple[int, int, int, int]]:
        """Split boxes with large width:height aspect ratio using projection."""
        final_boxes: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            x, y, w, h = box
            if h == 0:
                continue
            aspect = w / float(h)
            if aspect > aspect_threshold and w > 20:
                final_boxes.extend(self.split_box_by_vertical_projection(image, box))
            else:
                final_boxes.append(box)
        # Sort left-to-right after splitting
        final_boxes = self.sort_bounding_boxes_left_to_right(final_boxes)
        return final_boxes
    
    def merge_fragment_boxes(self, boxes: List[Tuple[int, int, int, int]],
                             area_ratio: float = 0.45,
                             overlap_ratio: float = 0.5,
                             gap_ratio: float = 0.35) -> List[Tuple[int, int, int, int]]:
        """
        Merge small fragmented components with neighbouring boxes when they
        likely belong to the same digit. Helps avoid treating detached strokes
        (e.g., a loose top bar of a '2') as separate digits.
        """
        if len(boxes) <= 1:
            return boxes

        boxes = [tuple(map(int, box)) for box in boxes]
        merged = True
        while merged:
            merged = False
            boxes.sort(key=lambda b: b[0])
            for i in range(len(boxes)):
                xi, yi, wi, hi = boxes[i]
                area_i = wi * hi
                if area_i == 0:
                    continue
                for j in range(i + 1, len(boxes)):
                    xj, yj, wj, hj = boxes[j]
                    area_j = wj * hj
                    if area_j == 0:
                        continue
                    small = min(area_i, area_j)
                    large = max(area_i, area_j)
                    if large == 0 or (small / large) > area_ratio:
                        continue

                    overlap_w = max(0, min(xi + wi, xj + wj) - max(xi, xj))
                    if overlap_w == 0:
                        continue
                    if overlap_w / float(min(wi, wj)) < overlap_ratio:
                        continue

                    vertical_gap = max(0, max(yi, yj) - min(yi + hi, yj + hj))
                    if vertical_gap > gap_ratio * max(hi, hj):
                        continue

                    new_x0 = min(xi, xj)
                    new_y0 = min(yi, yj)
                    new_x1 = max(xi + wi, xj + wj)
                    new_y1 = max(yi + hi, yj + hj)
                    boxes[i] = (new_x0, new_y0, new_x1 - new_x0, new_y1 - new_y0)
                    del boxes[j]
                    merged = True
                    break
                if merged:
                    break
        return boxes


    def connected_components_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """Segment digits using connected components analysis"""
        # Ensure binary image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            binary = ((1 - image) * 255).astype(np.uint8)
        # Erosion disabled to preserve thin gaps in digits like '3'
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=0)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        digits = []
        candidate_boxes = []
        
        # Process each component (skip background label 0)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Filter by size (more lenient for hand-drawn digits)
            if area < 20 or w < 3 or h < 3:
                continue
            
            # Collect box for possible splitting later
            candidate_boxes.append((x, y, w, h))
        candidate_boxes = self.merge_fragment_boxes(candidate_boxes)
        # Split wide boxes and then extract digits
        final_boxes = self.split_wide_boxes(image, candidate_boxes)
        for (x, y, w, h) in final_boxes:
            component_mask = np.zeros_like(binary)
            component_mask[y:y+h, x:x+w] = binary[y:y+h, x:x+w]
            # Add padding and resize
            max_dim = max(w, h)
            padded_size = int(max_dim * 1.2)
            padded_digit = np.zeros((padded_size, padded_size), dtype=np.uint8)
            y_offset = (padded_size - h) // 2
            x_offset = (padded_size - w) // 2
            padded_digit[y_offset:y_offset+h, x_offset:x_offset+w] = component_mask[y:y+h, x:x+w]
            # Resize to 28x28
            resized_digit = cv2.resize(padded_digit, (28, 28), interpolation=cv2.INTER_AREA)
            digits.append(resized_digit)
        
        # final_boxes is already left-to-right from split_wide_boxes
        return digits
    
    def contour_based_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """Segment digits using contour detection"""
        # Find contours
        contours = self.find_contours(image)
        
        # Filter contours
        filtered_contours = self.filter_contours(contours)
        
        # Get bounding boxes
        bounding_boxes = self.get_bounding_boxes(filtered_contours)
        # Split wide boxes to handle touching digits
        bounding_boxes = self.split_wide_boxes(image, bounding_boxes)
        
        # Sort left to right
        sorted_boxes = self.sort_bounding_boxes_left_to_right(bounding_boxes)
        
        # Extract digits
        digits = self.extract_digits(image, sorted_boxes)
        
        return digits
    def projection_based_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """Segment digits using global vertical projection profile.
        Works by computing the column-wise sum of foreground pixels,
        smoothing it, and cutting at low-ink valleys to separate digits.
        """
        # Convert to inverted binary (digits white)
        binary = self._to_binary_inv(image)
        # Erosion disabled by default to avoid closing gaps
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=0)
        # Compute vertical projection
        col_sums = np.sum(binary > 0, axis=0).astype(np.float32)
        if col_sums.size == 0:
            return []
        # Smooth profile
        smooth_window = max(3, int(0.03 * len(col_sums)))
        if smooth_window % 2 == 0:
            smooth_window += 1
        kernel = np.ones((smooth_window,), dtype=np.float32) / float(smooth_window)
        smoothed = np.convolve(col_sums, kernel, mode='same')
        # Determine valley threshold
        maxv = smoothed.max() if smoothed.size else 0
        threshold = 0.25 * maxv
        # Find cut indices (runs where profile is below threshold)
        w = binary.shape[1]
        margin = max(3, int(0.05 * w))
        min_part = max(10, int(0.25 * w))
        cuts: List[int] = []
        i = margin
        while i < w - margin:
            if smoothed[i] < threshold:
                start = i
                while i < w - margin and smoothed[i] < threshold:
                    i += 1
                end = i - 1
                cut = int((start + end) / 2)
                cuts.append(cut)
            else:
                i += 1
        # Build segments from cuts
        segments: List[Tuple[int, int]] = []
        last = 0
        min_width = max(12, int(0.12 * w))
        for c in cuts:
            if c - last >= min_width:
                segments.append((last, c))
                last = c
        if w - last >= min_width:
            segments.append((last, w))
        # If no valid segments, return as one digit
        if not segments:
            segments = [(0, w)]
        if len(segments) == 1 and maxv > 0:
            rows = np.where(np.any(binary > 0, axis=1))[0]
            total_height = int(rows.max() - rows.min() + 1) if rows.size else binary.shape[0]
            total_width = segments[0][1] - segments[0][0]
            aspect_ratio = total_width / max(total_height, 1) if total_height else 0
            valley_idx = self._find_best_valley(smoothed, margin, w - margin, window=8, drop_ratio=0.98)
            if valley_idx is not None and aspect_ratio > 1.08:
                left_w = valley_idx
                right_w = w - valley_idx
                if left_w >= min_width and right_w >= min_width:
                    segments = [(0, valley_idx), (valley_idx, w)]
        # Extract digit images per segment
        digits: List[np.ndarray] = []
        for x0, x1 in segments:
            # Bounding box covering the full height where foreground exists
            col = binary[:, x0:x1]
            ys = np.where(np.any(col > 0, axis=1))[0]
            if ys.size == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            w_seg, h_seg = max(1, x1 - x0), max(1, y1 - y0)
            # Create padded square image then resize to 28x28
            max_dim = max(w_seg, h_seg)
            padded_size = int(max_dim * 1.2)
            padded_digit = np.zeros((padded_size, padded_size), dtype=np.uint8)
            y_offset = (padded_size - h_seg) // 2
            x_offset = (padded_size - w_seg) // 2
            padded_digit[y_offset:y_offset+h_seg, x_offset:x_offset+w_seg] = binary[y0:y1, x0:x1]
            resized_digit = cv2.resize(padded_digit, (28, 28), interpolation=cv2.INTER_AREA)
            digits.append(resized_digit)
        return digits
    
    def segment_multi_digit_number(self, image: np.ndarray, method: str = 'contours') -> List[np.ndarray]:
        """
        Segment multi-digit number into individual digits
        
        Args:
            image: Input image containing handwritten number
            method: Segmentation method ('contours' or 'connected_components')
        
        Returns:
            List of individual digit images (28x28, normalized)
        """
        # Preprocess image for segmentation
        preprocessed = self.preprocessor.preprocess_pipeline(
            image,
            # Use closing instead of opening to preserve loops in 6/9
            steps=['grayscale', 'clahe', 'median', 'threshold', 'morphology_close']
        )
        
        if method == 'contours':
            digits = self.contour_based_segmentation(preprocessed)
        elif method == 'connected_components':
            digits = self.connected_components_segmentation(preprocessed)
        elif method in ('projection', 'projection_profile'):
            digits = self.projection_based_segmentation(preprocessed)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
        # Normalize digits for model input
        normalized_digits = []
        profile = getattr(self, 'processing_profile', 'digits')
        for digit in digits:
            # Ensure proper normalization to [0,1]
            if digit.max() > 1:
                normalized = digit.astype(np.float32) / 255.0
            else:
                normalized = digit.astype(np.float32)

            # For segmentation, we work with white-ink on black background to
            # compute center-of-mass reliably. Ensure that convention first.
            if normalized.mean() > 0.5:
                normalized = 1.0 - normalized

            # Centering: apply MNIST-style center-of-mass shifting only for digits.
            if profile != 'letters':
                centered_u8 = self.preprocessor.mnist_center_of_mass((normalized * 255).astype(np.uint8))
                normalized = centered_u8.astype(np.float32) / 255.0

            # Match model training conventions:
            # - digits: MNIST-style white glyph on black background
            # - letters: NIST by_class training uses black glyph on white background
            if profile == 'letters':
                # Convert to black-on-white if currently white-on-black
                if normalized.mean() < 0.5:
                    normalized = 1.0 - normalized

            normalized_digits.append(normalized)
        
        return normalized_digits
    
    def visualize_segmentation(self, image: np.ndarray, method: str = 'contours', return_fig: bool = False):
        """Visualize the segmentation process.
        If return_fig is True, returns the Matplotlib Figure instead of showing it.
        """
        # Get preprocessed image
        preprocessed = self.preprocessor.preprocess_pipeline(
            image, 
            steps=['grayscale', 'clahe', 'median', 'threshold', 'morphology']
        )
        
        # Get segmented digits
        digits = self.segment_multi_digit_number(image, method)
        
        # Create visualization
        num_digits = len(digits)
        if num_digits == 0:
            print("No digits found in image")
            return None if return_fig else None
        
        fig, axes = plt.subplots(2, max(3, num_digits), figsize=(15, 8))
        
        # Show original and preprocessed
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(preprocessed, cmap='gray')
        axes[0, 1].set_title('Preprocessed')
        axes[0, 1].axis('off')
        
        # Show bounding boxes on original
        if method == 'contours':
            contours = self.find_contours(preprocessed)
            filtered_contours = self.filter_contours(contours)
            bounding_boxes = self.get_bounding_boxes(filtered_contours)
            
            image_with_boxes = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
            for x, y, w, h in bounding_boxes:
                cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            axes[0, 2].imshow(image_with_boxes)
            axes[0, 2].set_title('Detected Regions')
            axes[0, 2].axis('off')
        
        # Show individual digits
        for i, digit in enumerate(digits):
            if i < axes.shape[1]:
                axes[1, i].imshow(digit, cmap='gray')
                axes[1, i].set_title(f'Digit {i+1}')
                axes[1, i].axis('off')
        
        # Hide unused subplots
        for i in range(num_digits, axes.shape[1]):
            if i < axes.shape[1]:
                axes[1, i].axis('off')
        
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()
    
    def merge_overlapping_boxes(self, bounding_boxes: List[Tuple[int, int, int, int]], 
                               overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes"""
        if not bounding_boxes:
            return []
        
        # Sort by x coordinate
        boxes = sorted(bounding_boxes, key=lambda box: box[0])
        merged = [boxes[0]]
        
        for current in boxes[1:]:
            last_merged = merged[-1]
            
            # Check for overlap
            x1, y1, w1, h1 = last_merged
            x2, y2, w2, h2 = current
            
            # Calculate overlap
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_area = overlap_x * min(h1, h2)
            
            min_area = min(w1 * h1, w2 * h2)
            overlap_ratio = overlap_area / min_area if min_area > 0 else 0
            
            if overlap_ratio > overlap_threshold:
                # Merge boxes
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                merged[-1] = (new_x, new_y, new_w, new_h)
            else:
                merged.append(current)
        
        return merged
class MultiDigitProcessor:
    """Process multi-digit strings using the available recognition models."""
    def __init__(self, model_loader_func=None):
        self.segmenter = ImageSegmenter()
        self.model_loader = model_loader_func
        self.models: Dict[str, Any] = {}
        self.index_to_char: Dict[int, str] = {}
        self.allowed_labels: Optional[Set[str]] = None
        self.allowed_indices: Optional[List[int]] = None
        self.processing_profile: str = 'digits'
    def _recompute_allowed_indices(self) -> None:
        if not self.index_to_char:
            self.allowed_indices = None
            return
        if not self.allowed_labels:
            self.allowed_indices = sorted(self.index_to_char.keys())
            return
        indices = [idx for idx, char in self.index_to_char.items() if char in self.allowed_labels]
        self.allowed_indices = indices if indices else None
    def set_processing_profile(self, profile: str) -> None:
        key = (profile or 'digits').lower()
        self.processing_profile = 'letters' if key in {'letters', 'alphanumeric'} else 'digits'
        if hasattr(self.segmenter, 'processing_profile'):
            self.segmenter.processing_profile = self.processing_profile
    def set_allowed_characters(self, labels: Optional[Iterable[str]]) -> None:
        self.allowed_labels = set(str(label) for label in labels) if labels else None
        self._recompute_allowed_indices()
    def _load_label_mapping(self, candidate_dirs: Iterable[str]) -> None:
        if self.index_to_char:
            return
        for base_dir in candidate_dirs:
            if not base_dir or not os.path.isdir(base_dir):
                continue
            for name in sorted(os.listdir(base_dir)):
                if not (name.startswith('label_mapping') and name.endswith('.json')):
                    continue
                mapping_path = os.path.join(base_dir, name)
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as handle:
                        data = json.load(handle)
                    if isinstance(data, dict) and data:
                        mapping = {int(k): str(v) for k, v in data.items()}
                        if mapping:
                            self.index_to_char = mapping
                            self._recompute_allowed_indices()
                            print(f"Loaded label mapping from {mapping_path}")
                            return
                except Exception as exc:
                    print(f"Warning: unable to load label mapping {mapping_path}: {exc}")
        if not self.index_to_char:
            self.index_to_char = {i: str(i) for i in range(10)}
            self._recompute_allowed_indices()
    def load_models(self) -> None:
        if self.model_loader:
            self.models = self.model_loader()
            self._recompute_allowed_indices()
            return
        try:
            from models import CNNModel, SVMModel, RandomForestModel
            src_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(src_dir)
            primary_models_dir = os.path.join(project_root, 'models')
            fallback_models_dir = os.path.join(src_dir, 'models')
            self._load_label_mapping([primary_models_dir, fallback_models_dir])
            def find_model_path(filename: str) -> Optional[str]:
                for base in (primary_models_dir, fallback_models_dir):
                    if not base:
                        continue
                    candidate = os.path.join(base, filename)
                    if os.path.exists(candidate):
                        return candidate
                return None
            def candidate_names(basename: str) -> Iterable[str]:
                yield basename
                yield basename.replace('.', f'_{self.processing_profile}.')
                yield basename.replace('.', '_combined.')
            loaded: Dict[str, Any] = {}
            for display_name, cls, basename in (
                ('cnn', CNNModel, 'cnn_model.h5'),
                ('svm', SVMModel, 'svm_model.pkl'),
                ('rf', RandomForestModel, 'rf_model.pkl'),
            ):
                model_path = None
                for candidate in candidate_names(basename):
                    model_path = find_model_path(candidate)
                    if model_path:
                        break
                if not model_path:
                    continue
                try:
                    model = cls()
                    model.load_model(model_path)
                    loaded[display_name] = model
                    print(f"Loaded {display_name.upper()} model from {model_path}")
                except Exception as exc:
                    print(f"Warning: failed to load {display_name.upper()} model from {model_path}: {exc}")
            self.models = loaded
            self._recompute_allowed_indices()
        except Exception as exc:
            print(f"Error loading models: {exc}")
    def _predict_with_model(self, model, digit_image: np.ndarray) -> np.ndarray:
        base = np.asarray(digit_image, dtype=np.float32)
        if base.ndim == 2:
            base = base.reshape(28, 28, 1)
        variants = [base]
        try:
            for angle in (-10, -5, 5, 10):
                matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
                rotated = cv2.warpAffine((base[:, :, 0] * 255.0).astype(np.uint8), matrix, (28, 28),
                                         flags=cv2.INTER_LINEAR, borderValue=0)
                variants.append(rotated.reshape(28, 28, 1).astype(np.float32) / 255.0)
        except Exception:
            pass
        prob_sum = None
        last_probs = None
        for variant in variants:
            _, probs = model.predict(variant)
            probs_arr = np.asarray(probs, dtype=np.float32)
            if probs_arr.ndim == 2:
                probs_arr = probs_arr[0]
            last_probs = probs_arr
            if prob_sum is None:
                prob_sum = probs_arr
            else:
                prob_sum += probs_arr
        if prob_sum is None:
            if last_probs is None:
                raise ValueError('Model returned no probabilities')
            prob_vec = last_probs
        else:
            prob_vec = prob_sum / max(len(variants), 1)
        if prob_vec.sum() <= 0:
            prob_vec = np.ones_like(prob_vec) / float(len(prob_vec))
        else:
            prob_vec = prob_vec / prob_vec.sum()
        return prob_vec.reshape(1, -1)
    def predict_single_digit(self, digit_image: np.ndarray, model_name: str = 'cnn') -> Tuple[str, float]:
        """Predict a single character using the specified model."""
        if model_name not in self.models or self.models.get(model_name) is None:
            raise ValueError(f"Model {model_name} not loaded")
        model = self.models[model_name]
        digit_array = np.asarray(digit_image)
        if digit_array.ndim == 2:
            digit_array = digit_array.reshape(28, 28, 1)
        digit_array = digit_array.astype(np.float32)
        if digit_array.max() > 1.0:
            digit_array /= 255.0
        probabilities = self._predict_with_model(model, digit_array)
        prob_vec = probabilities[0]
        if self.allowed_indices:
            candidates = [(idx, prob_vec[idx]) for idx in self.allowed_indices if 0 <= idx < len(prob_vec)]
            if candidates:
                idx, confidence = max(candidates, key=lambda item: item[1])
            else:
                idx = int(np.argmax(prob_vec))
                confidence = float(prob_vec[idx])
        else:
            idx = int(np.argmax(prob_vec))
            confidence = float(prob_vec[idx])
        label = self.index_to_char.get(idx, str(idx))
        return label, confidence
    def auto_select_segmentation(self, image: np.ndarray, model_name: str = 'cnn',
                                 candidate_methods: Optional[list[str]] = None):
        """Try multiple segmentation methods and pick the one with the highest confidence."""
        if candidate_methods is None:
            candidate_methods = ["connected_components", "projection"]
        best = None
        for method in candidate_methods:
            digit_images = self.segmenter.segment_multi_digit_number(image, method)
            if not digit_images:
                continue
            preds: List[Tuple[str, float]] = []
            try:
                for dimg in digit_images:
                    digit, conf = self.predict_single_digit(dimg, model_name)
                    preds.append((digit, conf))
            except Exception:
                continue
            avg_conf = float(np.mean([c for _, c in preds])) if preds else 0.0
            number_string = ''.join(p[0] for p in preds)
            if best is None or avg_conf > best[0]:
                best = (avg_conf, method, number_string, preds, digit_images)
        if best is None:
            return "", [], [], "", 0.0
        avg, method, number_string, preds, imgs = best
        return number_string, preds, imgs, method, avg
    def process_multi_digit_number(self, image: np.ndarray,
                                  model_name: str = 'cnn',
                                  segmentation_method: str = 'contours') -> Tuple[str, List[Tuple[str, float]], List[np.ndarray]]:
        """Process image into characters, classify each one, and return predictions."""
        if not self.models:
            self.load_models()
        digit_images = self.segmenter.segment_multi_digit_number(image, segmentation_method)
        if not digit_images:
            return "", [], []
        predictions: List[Tuple[str, float]] = []
        for digit_img in digit_images:
            digit, confidence = self.predict_single_digit(digit_img, model_name)
            predictions.append((digit, confidence))
        number_string = ''.join(pred for pred, _ in predictions)
        return number_string, predictions, digit_images
    def visualize_multi_digit_processing(self, image: np.ndarray,
                                         model_name: str = 'cnn',
                                         segmentation_method: str = 'contours') -> None:
        """Visualize the complete multi-digit processing pipeline."""
        number_string, predictions, digit_images = self.process_multi_digit_number(
            image, model_name, segmentation_method
        )
        if not digit_images:
            print("No digits detected in image")
            return
        num_digits = len(digit_images)
        fig, axes = plt.subplots(3, max(3, num_digits), figsize=(15, 10))
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        self.segmenter.visualize_segmentation(image, segmentation_method)
        for i, (digit_img, (pred, conf)) in enumerate(zip(digit_images, predictions)):
            if i < axes.shape[1]:
                axes[1, i].imshow(digit_img, cmap='gray')
                axes[1, i].set_title(f'Char {i+1}')
                axes[1, i].axis('off')
                axes[2, i].text(0.5, 0.5, f"Predicted: {pred}\nConfidence: {conf:.3f}", ha='center', va='center', transform=axes[2, i].transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[2, i].axis('off')
        for i in range(num_digits, axes.shape[1]):
            axes[1, i].axis('off')
            axes[2, i].axis('off')
        fig.suptitle(f'Recognized Text: {number_string} (Model: {model_name.upper()})',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
def test_segmentation():
    """Test segmentation with sample images"""
    processor = MultiDigitProcessor()
    
    # Create test multi-digit image
    test_image = np.zeros((60, 150), dtype=np.uint8)
    
    # Draw "123"
    cv2.putText(test_image, '123', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    
    print("Testing multi-digit segmentation...")
    
    # Test segmentation
    segmenter = ImageSegmenter()
    digits = segmenter.segment_multi_digit_number(test_image)
    
    print(f"Found {len(digits)} digits")
    
    # Visualize results
    segmenter.visualize_segmentation(test_image)
if __name__ == "__main__":
    test_segmentation()
