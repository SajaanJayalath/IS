"""
Main Entry Point for Handwritten Number Recognition System (HNRS)
Provides command-line interface and GUI launcher
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gui import main as gui_main
from models import CNNModel, SVMModel, RandomForestModel
from image_preprocessing import preprocess_for_mnist_model
from image_segmentation import MultiDigitProcessor

def test_single_image(image_path, model_name='cnn'):
    """Test recognition on a single image via command line"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        print(f"Processing image: {image_path}")
        print(f"Using model: {model_name.upper()}")
        
        # Initialize processor
        processor = MultiDigitProcessor()
        processor.load_models()
        
        # Process image
        number_string, predictions, digit_images = processor.process_multi_digit_number(
            image, model_name, 'contours'
        )
        
        if number_string:
            print(f"Recognized number: {number_string}")
            print("Individual predictions:")
            for i, (digit, confidence) in enumerate(predictions):
                print(f"  Digit {i+1}: {digit} (confidence: {confidence:.3f})")
        else:
            print("No digits detected in image")
            
    except Exception as e:
        print(f"Error processing image: {e}")

def test_models():
    """Test all models with sample data"""
    try:
        print("Testing model loading and basic functionality...")
        
        # Load models
        models = {}
        
        # CNN
        if os.path.exists('src/models/cnn_model.h5'):
            cnn = CNNModel()
            cnn.load_model('src/models/cnn_model.h5')
            models['CNN'] = cnn
            print("✓ CNN model loaded")
        
        # SVM
        if os.path.exists('src/models/svm_model.pkl'):
            svm = SVMModel()
            svm.load_model('src/models/svm_model.pkl')
            models['SVM'] = svm
            print("✓ SVM model loaded")
        
        # Random Forest
        if os.path.exists('src/models/rf_model.pkl'):
            rf = RandomForestModel()
            rf.load_model('src/models/rf_model.pkl')
            models['RF'] = rf
            print("✓ Random Forest model loaded")
        
        if not models:
            print("No trained models found. Please run train_models.py first.")
            return
        
        # Create test image (digit "5")
        test_image = np.zeros((28, 28), dtype=np.float32)
        # Simple pattern for digit 5
        test_image[5:10, 5:20] = 1.0  # Top horizontal line
        test_image[5:15, 5:8] = 1.0   # Left vertical line
        test_image[12:15, 5:20] = 1.0 # Middle horizontal line
        test_image[15:25, 17:20] = 1.0 # Right vertical line
        test_image[22:25, 5:20] = 1.0  # Bottom horizontal line
        
        print("\nTesting models with sample digit...")
        
        # Test each model
        for name, model in models.items():
            try:
                if name == 'CNN':
                    input_data = test_image.reshape(1, 28, 28, 1)
                else:
                    input_data = test_image.reshape(1, -1)
                
                prediction = model.predict(input_data)
                if hasattr(prediction, '__len__'):
                    pred_digit = prediction[0]
                else:
                    pred_digit = prediction
                    
                print(f"{name}: Predicted digit = {pred_digit}")
                
            except Exception as e:
                print(f"{name}: Error during prediction - {e}")
        
        print("\nModel testing complete!")
        
    except Exception as e:
        print(f"Error testing models: {e}")

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'cv2', 'numpy', 'PIL', 'matplotlib', 'tkinter',
        'tensorflow', 'sklearn', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'tkinter':
                import tkinter
            elif package == 'tensorflow':
                import tensorflow
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install missing packages using: pip install <package_name>")
        return False
    else:
        print("\nAll dependencies are installed!")
        return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Handwritten Number Recognition System")
    parser.add_argument('--gui', action='store_true', help='Launch GUI application')
    parser.add_argument('--test-image', type=str, help='Test recognition on single image')
    parser.add_argument('--model', type=str, choices=['cnn', 'svm', 'rf'], 
                       default='cnn', help='Model to use for recognition')
    parser.add_argument('--test-models', action='store_true', help='Test model loading and basic functionality')
    parser.add_argument('--check-deps', action='store_true', help='Check if all dependencies are installed')
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        check_dependencies()
        return
    
    # Test models
    if args.test_models:
        test_models()
        return
    
    # Test single image
    if args.test_image:
        if not os.path.exists(args.test_image):
            print(f"Error: Image file {args.test_image} not found")
            return
        test_single_image(args.test_image, args.model)
        return
    
    # Launch GUI (default behavior)
    if args.gui or len(sys.argv) == 1:
        print("Launching GUI application...")
        try:
            gui_main()
        except Exception as e:
            print(f"Error launching GUI: {e}")
            print("Make sure all dependencies are installed (run with --check-deps)")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
