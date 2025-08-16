"""
Image Segmentation Module for Handwritten Number Recognition System
Implements digit separation and multi-digit number processing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
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
    
    def connected_components_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """Segment digits using connected components analysis"""
        # Ensure binary image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if image.max() > 1:
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            binary = ((1 - image) * 255).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        digits = []
        bounding_boxes = []
        
        # Process each component (skip background label 0)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # Filter by size (more lenient for hand-drawn digits)
            if area < 20 or w < 3 or h < 3:
                continue
            
            # Extract component
            component_mask = (labels == i).astype(np.uint8) * 255
            digit_roi = component_mask[y:y+h, x:x+w]
            
            # Add padding and resize
            max_dim = max(w, h)
            padded_size = int(max_dim * 1.2)
            
            padded_digit = np.zeros((padded_size, padded_size), dtype=np.uint8)
            y_offset = (padded_size - h) // 2
            x_offset = (padded_size - w) // 2
            padded_digit[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi
            
            # Resize to 28x28
            resized_digit = cv2.resize(padded_digit, (28, 28), interpolation=cv2.INTER_AREA)
            
            digits.append(resized_digit)
            bounding_boxes.append((x, y, w, h))
        
        # Sort digits left to right
        if bounding_boxes:
            sorted_indices = sorted(range(len(bounding_boxes)), 
                                  key=lambda i: bounding_boxes[i][0])
            digits = [digits[i] for i in sorted_indices]
        
        return digits
    
    def contour_based_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """Segment digits using contour detection"""
        # Find contours
        contours = self.find_contours(image)
        
        # Filter contours
        filtered_contours = self.filter_contours(contours)
        
        # Get bounding boxes
        bounding_boxes = self.get_bounding_boxes(filtered_contours)
        
        # Sort left to right
        sorted_boxes = self.sort_bounding_boxes_left_to_right(bounding_boxes)
        
        # Extract digits
        digits = self.extract_digits(image, sorted_boxes)
        
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
            steps=['grayscale', 'clahe', 'median', 'threshold', 'morphology']
        )
        
        if method == 'contours':
            digits = self.contour_based_segmentation(preprocessed)
        elif method == 'connected_components':
            digits = self.connected_components_segmentation(preprocessed)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
        # Normalize digits for model input
        normalized_digits = []
        for digit in digits:
            # Ensure proper normalization
            if digit.max() > 1:
                normalized = digit.astype(np.float32) / 255.0
            else:
                normalized = digit.astype(np.float32)
            
            # Invert if needed (MNIST has white digits on black background)
            if normalized.mean() > 0.5:
                normalized = 1.0 - normalized
            
            normalized_digits.append(normalized)
        
        return normalized_digits
    
    def visualize_segmentation(self, image: np.ndarray, method: str = 'contours') -> None:
        """Visualize the segmentation process"""
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
            return
        
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
    """
    Process multi-digit numbers using segmentation and recognition
    """
    
    def __init__(self, model_loader_func=None):
        self.segmenter = ImageSegmenter()
        self.model_loader = model_loader_func
        self.models = {}
        
    def load_models(self):
        """Load trained models for digit recognition"""
        if self.model_loader:
            self.models = self.model_loader()
        else:
            # Default model loading
            try:
                from models import CNNModel, SVMModel, RandomForestModel
                
                # Load CNN model
                cnn_model = CNNModel()
                cnn_model.load_model('src/models/cnn_model.h5')
                self.models['cnn'] = cnn_model
                
                # Load SVM model
                svm_model = SVMModel()
                svm_model.load_model('src/models/svm_model.pkl')
                self.models['svm'] = svm_model
                
                # Load Random Forest model
                rf_model = RandomForestModel()
                rf_model.load_model('src/models/rf_model.pkl')
                self.models['rf'] = rf_model
                
                print("Models loaded successfully")
                
            except Exception as e:
                print(f"Error loading models: {e}")
    
    def predict_single_digit(self, digit_image: np.ndarray, model_name: str = 'cnn') -> Tuple[int, float]:
        """
        Predict single digit using specified model
        
        Args:
            digit_image: 28x28 normalized digit image
            model_name: Model to use ('cnn', 'svm', 'rf')
        
        Returns:
            Tuple of (predicted_digit, confidence)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        # Use the predict method from our model classes
        predictions, probabilities = model.predict(digit_image)
        
        # Extract prediction and confidence
        if hasattr(predictions, '__len__') and len(predictions) > 0:
            predicted_digit = predictions[0]
        else:
            predicted_digit = predictions
        
        # Extract confidence from probabilities
        if hasattr(probabilities, '__len__') and len(probabilities) > 0:
            if len(probabilities.shape) > 1:
                confidence = probabilities[0][predicted_digit]
            else:
                confidence = probabilities[predicted_digit]
        else:
            confidence = 1.0
        
        return int(predicted_digit), float(confidence)
    
    def process_multi_digit_number(self, image: np.ndarray, 
                                  model_name: str = 'cnn',
                                  segmentation_method: str = 'contours') -> Tuple[str, List[Tuple[int, float]], List[np.ndarray]]:
        """
        Process complete multi-digit number
        
        Args:
            image: Input image containing handwritten number
            model_name: Model to use for digit recognition
            segmentation_method: Segmentation method to use
        
        Returns:
            Tuple of (complete_number_string, individual_predictions, digit_images)
        """
        # Load models if not already loaded
        if not self.models:
            self.load_models()
        
        # Segment image into individual digits
        digit_images = self.segmenter.segment_multi_digit_number(image, segmentation_method)
        
        if not digit_images:
            return "", [], []
        
        # Predict each digit
        predictions = []
        for digit_img in digit_images:
            digit, confidence = self.predict_single_digit(digit_img, model_name)
            predictions.append((digit, confidence))
        
        # Construct complete number
        number_string = ''.join([str(pred[0]) for pred in predictions])
        
        return number_string, predictions, digit_images
    
    def visualize_multi_digit_processing(self, image: np.ndarray, 
                                       model_name: str = 'cnn',
                                       segmentation_method: str = 'contours') -> None:
        """Visualize the complete multi-digit processing pipeline"""
        # Process the number
        number_string, predictions, digit_images = self.process_multi_digit_number(
            image, model_name, segmentation_method
        )
        
        if not digit_images:
            print("No digits detected in image")
            return
        
        # Create visualization
        num_digits = len(digit_images)
        fig, axes = plt.subplots(3, max(3, num_digits), figsize=(15, 10))
        
        # Show original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Show segmentation visualization
        self.segmenter.visualize_segmentation(image, segmentation_method)
        
        # Show individual digit predictions
        for i, (digit_img, (pred, conf)) in enumerate(zip(digit_images, predictions)):
            if i < axes.shape[1]:
                axes[1, i].imshow(digit_img, cmap='gray')
                axes[1, i].set_title(f'Digit {i+1}')
                axes[1, i].axis('off')
                
                axes[2, i].text(0.5, 0.5, f'Predicted: {pred}\nConfidence: {conf:.3f}', 
                               ha='center', va='center', transform=axes[2, i].transAxes,
                               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                axes[2, i].axis('off')
        
        # Hide unused subplots
        for i in range(num_digits, axes.shape[1]):
            if i < axes.shape[1]:
                axes[1, i].axis('off')
                axes[2, i].axis('off')
        
        # Show final result
        fig.suptitle(f'Recognized Number: {number_string} (Model: {model_name.upper()})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return number_string, predictions

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
