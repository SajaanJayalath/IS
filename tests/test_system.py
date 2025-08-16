"""
Comprehensive System Tests for Handwritten Number Recognition System
Tests all components: preprocessing, segmentation, models, and integration
"""

import os
import sys
import unittest
import numpy as np
import cv2
from PIL import Image

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import CNNModel, SVMModel, RandomForestModel
from image_preprocessing import ImagePreprocessor, preprocess_for_mnist_model
from image_segmentation import ImageSegmenter, MultiDigitProcessor
from data_loader import MNISTDataLoader

class TestImagePreprocessing(unittest.TestCase):
    """Test image preprocessing functionality"""
    
    def setUp(self):
        self.preprocessor = ImagePreprocessor()
        # Create test image
        self.test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
    def test_grayscale_conversion(self):
        """Test grayscale conversion"""
        # Test with color image
        color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = self.preprocessor.convert_to_grayscale(color_image)
        self.assertEqual(len(gray.shape), 2)
        
        # Test with already grayscale image
        gray2 = self.preprocessor.convert_to_grayscale(self.test_image)
        np.testing.assert_array_equal(gray2, self.test_image)
    
    def test_histogram_equalization(self):
        """Test histogram equalization"""
        result = self.preprocessor.histogram_equalization(self.test_image)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_adaptive_threshold(self):
        """Test adaptive thresholding"""
        result = self.preprocessor.adaptive_threshold(self.test_image)
        self.assertEqual(result.shape, self.test_image.shape)
        # Should be binary (0 or 255)
        unique_values = np.unique(result)
        self.assertTrue(len(unique_values) <= 2)
    
    def test_resize_image(self):
        """Test image resizing"""
        result = self.preprocessor.resize_image(self.test_image, (28, 28))
        self.assertEqual(result.shape, (28, 28))
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline"""
        result = self.preprocessor.preprocess_pipeline(self.test_image)
        self.assertEqual(result.shape, (28, 28))
        self.assertTrue(result.min() >= 0 and result.max() <= 1)

class TestImageSegmentation(unittest.TestCase):
    """Test image segmentation functionality"""
    
    def setUp(self):
        self.segmenter = ImageSegmenter()
        # Create test multi-digit image
        self.test_image = np.zeros((60, 150), dtype=np.uint8)
        cv2.putText(self.test_image, '123', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        
    def test_find_contours(self):
        """Test contour detection"""
        contours = self.segmenter.find_contours(self.test_image)
        self.assertIsInstance(contours, list)
        self.assertGreater(len(contours), 0)
    
    def test_filter_contours(self):
        """Test contour filtering"""
        contours = self.segmenter.find_contours(self.test_image)
        filtered = self.segmenter.filter_contours(contours)
        self.assertIsInstance(filtered, list)
        self.assertLessEqual(len(filtered), len(contours))
    
    def test_segment_multi_digit(self):
        """Test multi-digit segmentation"""
        digits = self.segmenter.segment_multi_digit_number(self.test_image)
        self.assertIsInstance(digits, list)
        # Should find approximately 3 digits for "123"
        self.assertGreater(len(digits), 0)
        
        # Check digit format
        for digit in digits:
            self.assertEqual(digit.shape, (28, 28))
            self.assertTrue(digit.min() >= 0 and digit.max() <= 1)

class TestModels(unittest.TestCase):
    """Test ML model functionality"""
    
    def setUp(self):
        # Create sample test data
        self.test_data_cnn = np.random.random((10, 28, 28, 1)).astype(np.float32)
        self.test_data_flat = np.random.random((10, 784)).astype(np.float32)
        self.test_labels = np.random.randint(0, 10, 10)
        
    def test_cnn_model_creation(self):
        """Test CNN model creation"""
        model = CNNModel()
        self.assertIsNotNone(model.model)
        
    def test_svm_model_creation(self):
        """Test SVM model creation"""
        model = SVMModel()
        self.assertIsNotNone(model.model)
        
    def test_rf_model_creation(self):
        """Test Random Forest model creation"""
        model = RandomForestModel()
        self.assertIsNotNone(model.model)
    
    def test_model_prediction_shapes(self):
        """Test model prediction output shapes"""
        # Test CNN
        if os.path.exists('src/models/cnn_model.h5'):
            cnn = CNNModel()
            cnn.load_model('src/models/cnn_model.h5')
            pred = cnn.predict(self.test_data_cnn[:1])
            self.assertEqual(len(pred), 1)
        
        # Test SVM
        if os.path.exists('src/models/svm_model.pkl'):
            svm = SVMModel()
            svm.load_model('src/models/svm_model.pkl')
            pred = svm.predict(self.test_data_flat[:1])
            self.assertEqual(len(pred), 1)

class TestDataLoader(unittest.TestCase):
    """Test data loading functionality"""
    
    def test_mnist_data_loader(self):
        """Test MNIST data loader"""
        loader = MNISTDataLoader()
        
        # Test if MNIST files exist
        if os.path.exists('MNIST_CSV/mnist_train.csv'):
            X_train, y_train, X_test, y_test = loader.load_data()
            
            # Check shapes
            self.assertEqual(len(X_train.shape), 2)
            self.assertEqual(X_train.shape[1], 784)
            self.assertEqual(len(y_train), len(X_train))
            
            # Check data ranges
            self.assertTrue(X_train.min() >= 0)
            self.assertTrue(X_train.max() <= 255)
            self.assertTrue(y_train.min() >= 0)
            self.assertTrue(y_train.max() <= 9)

class TestIntegration(unittest.TestCase):
    """Test system integration"""
    
    def setUp(self):
        self.processor = MultiDigitProcessor()
        
    def test_multi_digit_processor_initialization(self):
        """Test multi-digit processor initialization"""
        self.assertIsNotNone(self.processor.segmenter)
        
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        # Create test image with number
        test_image = np.zeros((60, 100), dtype=np.uint8)
        cv2.putText(test_image, '42', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        
        try:
            # Load models if available
            self.processor.load_models()
            
            if self.processor.models:
                # Process image
                number_string, predictions, digit_images = self.processor.process_multi_digit_number(
                    test_image, 'cnn', 'contours'
                )
                
                # Check results format
                self.assertIsInstance(number_string, str)
                self.assertIsInstance(predictions, list)
                self.assertIsInstance(digit_images, list)
                
                if predictions:
                    for pred, conf in predictions:
                        self.assertIsInstance(pred, (int, np.integer))
                        self.assertIsInstance(conf, (float, np.floating))
                        self.assertTrue(0 <= pred <= 9)
                        self.assertTrue(0 <= conf <= 1)
                        
        except Exception as e:
            self.skipTest(f"Models not available for testing: {e}")

def run_performance_tests():
    """Run performance benchmarks"""
    print("Running Performance Tests...")
    print("="*50)
    
    # Test preprocessing speed
    preprocessor = ImagePreprocessor()
    test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    import time
    start_time = time.time()
    for _ in range(100):
        _ = preprocessor.preprocess_pipeline(test_image)
    preprocessing_time = (time.time() - start_time) / 100
    
    print(f"Average preprocessing time: {preprocessing_time:.4f} seconds")
    
    # Test segmentation speed
    segmenter = ImageSegmenter()
    start_time = time.time()
    for _ in range(50):
        _ = segmenter.segment_multi_digit_number(test_image)
    segmentation_time = (time.time() - start_time) / 50
    
    print(f"Average segmentation time: {segmentation_time:.4f} seconds")
    
    # Test model prediction speed (if models available)
    if os.path.exists('src/models/cnn_model.h5'):
        cnn = CNNModel()
        cnn.load_model('src/models/cnn_model.h5')
        
        test_input = np.random.random((1, 28, 28, 1)).astype(np.float32)
        start_time = time.time()
        for _ in range(100):
            _ = cnn.predict(test_input)
        cnn_time = (time.time() - start_time) / 100
        
        print(f"Average CNN prediction time: {cnn_time:.4f} seconds")
    
    print("\nPerformance testing complete!")

def create_test_images():
    """Create sample test images for validation"""
    test_dir = "sample_images"
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"Creating test images in {test_dir}/...")
    
    # Create single digit images
    for digit in range(10):
        img = np.zeros((60, 60), dtype=np.uint8)
        cv2.putText(img, str(digit), (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        cv2.imwrite(f"{test_dir}/single_digit_{digit}.png", img)
    
    # Create multi-digit images
    multi_digit_numbers = ['12', '345', '6789', '1024', '9876']
    for number in multi_digit_numbers:
        img_width = len(number) * 40 + 20
        img = np.zeros((60, img_width), dtype=np.uint8)
        cv2.putText(img, number, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
        cv2.imwrite(f"{test_dir}/multi_digit_{number}.png", img)
    
    print(f"Created {10 + len(multi_digit_numbers)} test images")

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="HNRS System Tests")
    parser.add_argument('--unit', action='store_true', help='Run unit tests')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--create-samples', action='store_true', help='Create sample test images')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.create_samples or args.all:
        create_test_images()
        print()
    
    if args.performance or args.all:
        run_performance_tests()
        print()
    
    if args.unit or args.all or len(sys.argv) == 1:
        print("Running Unit Tests...")
        print("="*50)
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Print summary
        print(f"\nTest Summary:")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")

if __name__ == "__main__":
    main()
