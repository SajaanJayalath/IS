"""
Image Preprocessing Module for Handwritten Number Recognition System
Implements advanced preprocessing techniques for real-world handwritten images
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

class ImagePreprocessor:
    """
    Advanced image preprocessing for handwritten number recognition
    """
    
    def __init__(self):
        self.preprocessing_steps = []
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization to improve contrast"""
        return cv2.equalizeHist(image)
    
    def adaptive_histogram_equalization(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur for noise reduction"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def median_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply median filter for noise reduction"""
        return cv2.medianBlur(image, kernel_size)
    
    def adaptive_threshold(self, image: np.ndarray, max_value: int = 255, 
                          adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          threshold_type: int = cv2.THRESH_BINARY,
                          block_size: int = 11, C: int = 2) -> np.ndarray:
        """Apply adaptive thresholding for binarization"""
        return cv2.adaptiveThreshold(image, max_value, adaptive_method, 
                                   threshold_type, block_size, C)
    
    def morphological_operations(self, image: np.ndarray, operation: str = 'opening', 
                                kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
        """Apply morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'opening':
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'closing':
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == 'erosion':
            return cv2.erode(image, kernel, iterations=iterations)
        elif operation == 'dilation':
            return cv2.dilate(image, kernel, iterations=iterations)
        else:
            return image

    def mnist_center_of_mass(self, image: np.ndarray) -> np.ndarray:
        """
        Center a 28x28 digit image by shifting its center of mass to the center.

        Expects a single-channel image of shape (28, 28) with the digit as
        bright foreground on dark background (MNIST style). Returns a 28x28 image
        of the same dtype.
        """
        if image is None or image.size == 0:
            return image

        # Ensure 28x28 single-channel
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape != (28, 28):
            image = self.resize_image(image, (28, 28))

        # Work in uint8 for cv2 moments; foreground should be white
        img = image.copy()
        if img.dtype != np.uint8:
            # If it's normalized float, scale up
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        # Threshold to binary (white digit on black)
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # If background appears white (drawn case), invert to match MNIST
        if bw.mean() > 127:
            bw = 255 - bw

        M = cv2.moments(bw)
        if abs(M["m00"]) < 1e-3:
            return image  # empty; nothing to center

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shift_x = int(round(14 - cx))
        shift_y = int(round(14 - cy))

        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(img, M_shift, (28, 28), flags=cv2.INTER_NEAREST, borderValue=0)
        return shifted
    
    def edge_detection(self, image: np.ndarray, low_threshold: int = 50, 
                      high_threshold: int = 150) -> np.ndarray:
        """Apply Canny edge detection"""
        return cv2.Canny(image, low_threshold, high_threshold)
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct image rotation/skew"""
        # Find contours to determine skew angle
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
            
        # Find the largest contour (assuming it's the main content)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Correct angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        # Rotate image
        if abs(angle) > 0.5:  # Only rotate if angle is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = (28, 28)) -> np.ndarray:
        """Resize image to target size while maintaining aspect ratio"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create target size image with padding
        result = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return result
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values to 0-1 range"""
        return image.astype(np.float32) / 255.0
    
    def invert_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Invert image if background is darker than foreground"""
        # Calculate mean of border pixels to determine background
        border_pixels = np.concatenate([
            image[0, :],  # top row
            image[-1, :],  # bottom row
            image[:, 0],  # left column
            image[:, -1]  # right column
        ])
        
        border_mean = np.mean(border_pixels)
        center_mean = np.mean(image[image.shape[0]//4:3*image.shape[0]//4, 
                                   image.shape[1]//4:3*image.shape[1]//4])
        
        # If border is darker than center, invert
        if border_mean < center_mean:
            return 255 - image
        return image
    
    def preprocess_pipeline(self, image: np.ndarray, steps: List[str] = None) -> np.ndarray:
        """
        Apply complete preprocessing pipeline
        
        Args:
            image: Input image
            steps: List of preprocessing steps to apply
                  Options: ['grayscale', 'hist_eq', 'clahe', 'blur', 'median', 
                           'threshold', 'morphology', 'deskew', 'invert', 'resize', 'normalize']
        """
        if steps is None:
            steps = ['grayscale', 'clahe', 'median', 'threshold', 'morphology', 
                    'deskew', 'invert', 'resize', 'normalize']
        
        processed = image.copy()
        
        for step in steps:
            if step == 'grayscale':
                processed = self.convert_to_grayscale(processed)
            elif step == 'hist_eq':
                processed = self.histogram_equalization(processed)
            elif step == 'clahe':
                processed = self.adaptive_histogram_equalization(processed)
            elif step == 'blur':
                processed = self.gaussian_blur(processed)
            elif step == 'median':
                processed = self.median_filter(processed)
            elif step == 'threshold':
                processed = self.adaptive_threshold(processed)
            elif step == 'morphology':
                # Default remains opening for noise removal
                processed = self.morphological_operations(processed, 'opening')
            elif step == 'morphology_close':
                # Alternative that preserves loops (useful for 6/9)
                processed = self.morphological_operations(processed, 'closing')
            elif step == 'center_mass':
                processed = self.mnist_center_of_mass(processed)
            elif step == 'deskew':
                processed = self.deskew_image(processed)
            elif step == 'invert':
                processed = self.invert_if_needed(processed)
            elif step == 'resize':
                processed = self.resize_image(processed)
            elif step == 'normalize':
                processed = self.normalize_image(processed)
        
        return processed
    
    def visualize_preprocessing_steps(self, image: np.ndarray, steps: List[str] = None) -> None:
        """Visualize the effect of each preprocessing step"""
        if steps is None:
            steps = ['grayscale', 'clahe', 'threshold', 'morphology', 'deskew']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        processed = image.copy()
        
        for i, step in enumerate(steps[:5], 1):
            if step == 'grayscale':
                processed = self.convert_to_grayscale(processed)
            elif step == 'clahe':
                processed = self.adaptive_histogram_equalization(processed)
            elif step == 'threshold':
                processed = self.adaptive_threshold(processed)
            elif step == 'morphology':
                processed = self.morphological_operations(processed, 'opening')
            elif step == 'deskew':
                processed = self.deskew_image(processed)
            
            axes[i].imshow(processed, cmap='gray')
            axes[i].set_title(f'After {step}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()


def preprocess_for_mnist_model(image: np.ndarray) -> np.ndarray:
    """Preprocess a user-drawn digit so it matches MNIST training data."""
    if image is None:
        raise ValueError('Input image is None')

    gray = np.asarray(image)
    if gray.ndim == 3 and gray.shape[2] == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    if gray.dtype != np.uint8:
        gray = gray.astype(np.float32)
        if gray.max() <= 1.0:
            gray *= 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Binarise and invert so foreground strokes are white like MNIST digits
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        digit_crop = thresh
    else:
        x, y, w, h = cv2.boundingRect(coords)
        digit_crop = thresh[y:y + h, x:x + w]
        if w == 0 or h == 0:
            digit_crop = thresh

    target_size = 28
    pad_margin = 4
    max_dim = max(digit_crop.shape[:2])
    if max_dim == 0:
        resized = np.zeros((target_size, target_size), dtype=np.uint8)
    else:
        scale = (target_size - pad_margin) / float(max_dim)
        new_w = max(1, int(round(digit_crop.shape[1] * scale)))
        new_h = max(1, int(round(digit_crop.shape[0] * scale)))
        resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y0 = (target_size - resized.shape[0]) // 2
    x0 = (target_size - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized

    canvas = canvas.astype(np.float32) / 255.0
    return canvas.reshape(target_size, target_size, 1)
def prepare_for_character_model(image: np.ndarray, target_size: int = 28) -> np.ndarray:
    """
    Prepare an image drawn by the user for the character models trained on the NIST by_class dataset.

    This mirrors the preprocessing performed during training: convert to grayscale, optionally invert so
    foreground is bright, resize with aspect-ratio preservation, and normalise to [0, 1].
    """
    if image is None:
        raise ValueError('Input image is None')

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError('Input image has invalid dimensions')

    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    center = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    if center.size > 0 and border.mean() < center.mean():
        gray = 255 - gray

    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    canvas = canvas.astype(np.float32) / 255.0
    return canvas.reshape(target_size, target_size, 1)
