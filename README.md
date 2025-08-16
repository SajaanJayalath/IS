# Handwritten Number Recognition System (HNRS)

A comprehensive machine learning system for recognizing handwritten numbers using multiple ML techniques including CNN, SVM, and Random Forest models.

## 🎯 Project Overview

This project implements a complete Handwritten Number Recognition System (HNRS) as part of the COS30018 Intelligent Systems assignment. The system can recognize both single digits and multi-digit numbers from images using advanced image processing and machine learning techniques.

### Key Features

- **Multiple ML Models**: CNN (99.23% accuracy), SVM (96.34% accuracy), Random Forest (95.11% accuracy)
- **Advanced Image Processing**: Histogram equalization, noise reduction, morphological operations
- **Image Segmentation**: Automatic digit separation for multi-digit numbers
- **Interactive GUI**: Drawing canvas and image upload functionality
- **Real-time Recognition**: Instant prediction with confidence scores
- **Model Comparison**: Side-by-side performance analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shavi-dil/Intelligent_Systems_COS30018.git
cd Intelligent_Systems_COS30018
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models (if not already trained):
```bash
cd src
python train_models.py
```

4. Launch the GUI application:
```bash
python src/main.py --gui
```

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── models.py                 # ML model implementations
│   ├── data_loader.py            # MNIST data loading utilities
│   ├── train_models.py           # Model training script
│   ├── image_preprocessing.py    # Advanced image preprocessing
│   ├── image_segmentation.py     # Multi-digit segmentation
│   ├── gui.py                    # GUI application
│   ├── main.py                   # Main entry point
│   └── models/                   # Trained model files
│       ├── cnn_model.h5
│       ├── svm_model.pkl
│       └── rf_model.pkl
├── tests/                        # Test suite
│   └── test_system.py           # Comprehensive system tests
├── MNIST_CSV/                    # MNIST dataset in CSV format
├── sample_images/                # Test images (generated)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Usage

### GUI Application

Launch the main GUI application:
```bash
python src/main.py --gui
```

**Features:**
- **Drawing Canvas**: Draw numbers directly with your mouse
- **Image Upload**: Load handwritten number images from files
- **Model Selection**: Choose between CNN, SVM, or Random Forest
- **Segmentation Options**: Contour-based or connected components
- **Visualization**: View preprocessing and segmentation steps
- **Model Comparison**: Compare all models on the same image

### Command Line Interface

Test single images:
```bash
python src/main.py --test-image path/to/image.png --model cnn
```

Check dependencies:
```bash
python src/main.py --check-deps
```

Test model functionality:
```bash
python src/main.py --test-models
```

### Running Tests

Run comprehensive system tests:
```bash
python tests/test_system.py --all
```

Run specific test types:
```bash
python tests/test_system.py --unit          # Unit tests only
python tests/test_system.py --performance   # Performance benchmarks
python tests/test_system.py --create-samples # Create test images
```

## 🧠 Machine Learning Models

### CNN Model (Primary - 99.23% Accuracy)
- **Architecture**: 3 Convolutional blocks + Dense layers
- **Features**: Batch normalization, dropout, max pooling
- **Training Time**: ~3.6 minutes
- **Best For**: Highest accuracy, complex pattern recognition

### SVM Model (96.34% Accuracy)
- **Kernel**: RBF (Radial Basis Function)
- **Features**: Probability estimates, efficient training
- **Training Time**: ~43 seconds
- **Best For**: Fast inference, good generalization

### Random Forest Model (95.11% Accuracy)
- **Estimators**: 100 trees
- **Features**: Fast training, robust to overfitting
- **Training Time**: <1 second
- **Best For**: Fastest processing, ensemble learning

## 🖼️ Image Processing Pipeline

### Preprocessing Steps
1. **Grayscale Conversion**: Convert color images to grayscale
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
3. **Noise Reduction**: Median filtering and Gaussian blur
4. **Adaptive Thresholding**: Binary image conversion
5. **Morphological Operations**: Opening/closing for noise removal
6. **Deskewing**: Automatic rotation correction
7. **Normalization**: Pixel value scaling (0-1 range)

### Segmentation Methods
- **Contour Detection**: Find digit boundaries using contour analysis
- **Connected Components**: Separate digits using connected component analysis
- **Bounding Box Extraction**: Individual digit isolation
- **Left-to-Right Ordering**: Proper digit sequence reconstruction

## 📊 Performance Metrics

| Model | Accuracy | Training Time | Inference Speed | Memory Usage |
|-------|----------|---------------|-----------------|--------------|
| CNN | 99.23% | 218.51s | ~0.01s | High |
| SVM | 96.34% | 43.49s | ~0.001s | Medium |
| Random Forest | 95.11% | 0.78s | ~0.0001s | Low |

## 🎮 GUI Features

### Input Methods
- **Drawing Canvas**: 280x280 pixel canvas for direct drawing
- **Image Upload**: Support for PNG, JPG, JPEG, BMP, TIFF formats
- **Clear Function**: Reset canvas for new drawings

### Processing Options
- **Model Selection**: Choose between CNN, SVM, Random Forest
- **Segmentation Method**: Contours or connected components
- **Visualization Toggles**: Show preprocessing and segmentation steps
- **Real-time Processing**: Instant recognition results

### Results Display
- **Main Recognition**: Complete number result
- **Individual Digits**: Per-digit predictions with confidence
- **Model Information**: Selected model and parameters
- **Performance Metrics**: Processing time and accuracy stats

## 🧪 Testing

The system includes comprehensive testing:

### Unit Tests
- Image preprocessing functionality
- Segmentation algorithms
- Model loading and prediction
- Data pipeline integrity

### Integration Tests
- End-to-end pipeline testing
- GUI component testing
- Multi-model comparison
- Error handling validation

### Performance Tests
- Processing speed benchmarks
- Memory usage analysis
- Scalability testing
- Real-time performance validation

## 📈 Technical Specifications

### System Requirements
- **OS**: Windows 10/11, macOS, Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for models and dependencies

### Dependencies
- **Core ML**: TensorFlow, scikit-learn, PyTorch
- **Image Processing**: OpenCV, PIL/Pillow
- **GUI**: tkinter, matplotlib
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## 🔍 Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Retrain models if files are missing
cd src
python train_models.py
```

**GUI not launching:**
```bash
# Check dependencies
python src/main.py --check-deps
```

**Poor recognition accuracy:**
- Ensure image has good contrast
- Try different preprocessing options
- Use CNN model for best accuracy
- Check image quality and resolution

### Error Messages

- **"No digits detected"**: Image may need better preprocessing or different segmentation method
- **"Model not loaded"**: Run training script or check model file paths
- **"Import error"**: Install missing dependencies using pip

## 📚 Documentation

### Code Documentation
- All classes and functions include comprehensive docstrings
- Type hints for better code clarity
- Inline comments for complex algorithms

### User Manual
- Step-by-step GUI usage instructions
- Command-line interface examples
- Troubleshooting guide

## 🎯 Assignment Requirements Compliance

### Task Completion
- ✅ **Task 1**: Image Preprocessing (8 marks) - Advanced preprocessing pipeline
- ✅ **Task 2**: Image Segmentation (8 marks) - Multi-digit separation
- ✅ **Task 3**: ML Implementation (24 marks) - Three model comparison
- ✅ **Task 4**: Integration & Testing (10 marks) - Complete system
- ✅ **GUI**: User interface (10 marks) - Full-featured application

### Technical Requirements
- ✅ MNIST dataset training and evaluation
- ✅ Multiple ML technique comparison
- ✅ GUI for user interaction
- ✅ Image acquisition from files
- ✅ Single and multi-digit recognition
- ✅ Comprehensive testing and evaluation

## 👥 Team Contribution

This project demonstrates collaborative development with clear module separation:
- **Data Pipeline**: MNIST loading and preprocessing
- **ML Models**: CNN, SVM, Random Forest implementation
- **Image Processing**: Advanced preprocessing and segmentation
- **GUI Development**: User interface and integration
- **Testing**: Comprehensive test suite and validation

## 📄 License

This project is developed for educational purposes as part of COS30018 Intelligent Systems coursework.

## 🤝 Contributing

For course-related contributions:
1. Follow the established code structure
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure all models maintain >95% accuracy

---

**Developed for COS30018 - Intelligent Systems**  
**Due: 02/11/2025**  
**Team Project - Handwritten Number Recognition**
