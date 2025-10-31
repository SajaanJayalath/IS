# Handwritten Character Recognition System (HNRS)

## Overview

HNRS is a Python application for recognizing handwritten digits, letters, and arithmetic expressions. It provides a Tkinter GUI to draw or upload images, plus a small CLI for testing. Under the hood it combines robust preprocessing, multiple segmentation strategies, and several classifier backends (CNN, SVM, Random Forest, k-NN). The project was developed for COS30018 Intelligent Systems.

## Features

- Multi-mode recognition: digits, letters, arithmetic expressions
- Segmentation methods: contours, connected components, projection, watershed (letters)
- Models: CNN (TensorFlow), SVM, Random Forest, k-NN (letters), and an arithmetic CNN/ensemble
- Tkinter GUI: draw/erase, upload, visualize preprocessing/segmentation, per-character confidences
- Optional external MER integration for arithmetic recognition

## Repository Structure

- `src/main.py`: CLI/GUI entry point
- `src/gui.py`: Tkinter interface, mode switching, visualization
- `src/models.py`: CNN, SVM, Random Forest implementations for digits/characters
- `src/image_preprocessing.py`: preprocessing and normalization utilities
- `src/image_segmentation.py`: segmentation + multi-character processing (digits)
- `src/arithmetic/`: arithmetic expression pipeline (segmentation, CNN/ensemble, heuristics)
- `src/letters_new/`: letters pipeline (watershed segmentation; k-NN and optional CNN)
- `src/integrations/mer_adapter.py`: adapter for external Math-Expression-Recognizer-and-Solver
- `src/data_loader.py`: dataset loaders (MNIST CSV, NIST by_class, EMNIST, SVHN, combined)
- `models/`: pretrained weights and label mappings
  - `cnn_model.h5`, `cnn_model_mnist_csv.h5`, `cnn_model_nist_by_class.h5`
  - `svm_model.pkl`, `rf_model.pkl` (+ `*_mnist_csv.pkl` variants)
  - `cnn_model_arithmetic.h5`, `arithmetic_confusion_matrix.npy`
  - `label_mapping.json`, `label_mapping_mnist_csv.json`, `label_mapping_nist_by_class.json`, `label_mapping_arithmetic.json`
- `models_new/`: additional models (e.g., `letters_byclass_knn.pkl`)
- `data/by_class/by_class/`: NIST SD19 by_class dataset (if available)
- `Maths/`: legacy sandbox for arithmetic experiments (not used by the GUI)

## Requirements

- Python 3.10 or 3.11 recommended
- Install dependencies:
  - `pip install -r requirements.txt`
  - TensorFlow is required for CNN features; SVM/RandomForest/k-NN work without it
  - Torch/torchvision are listed for optional dataset loaders; core GUI does not require them

## Run the App

- Launch GUI (default when no args provided):
  - `python src/main.py --gui`

- CLI options (from `src/main.py`):
  - `python src/main.py --check-deps`  Check/install status of key packages
  - `python src/main.py --test-models`  Load available models and run a basic smoke test
  - `python src/main.py --test-image path/to/image.png --model cnn|svm|rf|ensemble`  Recognize a single image via CLI

### GUI Tips

- Modes: choose Digits, Letters, or Arithmetic in the GUI
- Segmentation: digits support auto/contours/connected components/projection; letters use watershed; arithmetic offers Best/Hybrid/Connected Components/Projection
- Models: the selector reflects what’s available for the active mode
  - Digits: CNN/SVM/Random Forest (from `models/`)
  - Letters: k-NN (from `models_new/letters_byclass_knn.pkl`) or CNN (`cnn_model_nist_by_class.h5`)
  - Arithmetic: CNN/Ensemble if `cnn_model_arithmetic.h5` exists; optional external “MER (External)” if configured

## Models and Weights

- Place weights and label maps in `models/` (root) or `src/models/`. The app searches both.
- Common files used by the GUI:
  - Digits: `cnn_model.h5` or `cnn_model_mnist_csv.h5`, `svm_model.pkl`, `rf_model.pkl`, plus `label_mapping_mnist_csv.json` or `label_mapping.json`
  - Letters: `cnn_model_nist_by_class.h5` with `label_mapping_nist_by_class.json`, or `models_new/letters_byclass_knn.pkl`
  - Arithmetic: `cnn_model_arithmetic.h5` with `label_mapping_arithmetic.json`

## Datasets

- Letters: the k-NN pipeline expects NIST SD19 by_class images under `data/by_class/by_class/` (hex-named folders per class). If `models_new/letters_byclass_knn.pkl` is missing and this dataset is present, the letters pipeline can train a k-NN model.
- MNIST CSV: used by some loaders in `src/data_loader.py` if you run training scripts or custom experiments.
- EMNIST/SVHN: supported via torchvision loaders in `src/data_loader.py` for experimentation (optional).

### Train a quick k-NN letters model

If you have NIST by_class data and want a fast letters baseline for the GUI:

```bash
python - <<"PY"
import sys, os
sys.path.append('src')
from letters_new.model_knn import train_knn_byclass, save_model
m, mapping = train_knn_byclass(max_per_class=400, max_total=20000, n_neighbors=3)
save_model(m, mapping)
print('Saved models_new/letters_byclass_knn.pkl')
PY
```

## Optional: External MER Integration (Arithmetic)

To use the external Math-Expression-Recognizer-and-Solver with the GUI:

1) Copy the repo into one of:
   - `external/Math-Expression-Recognizer-and-Solver`
   - `src/external/Math-Expression-Recognizer-and-Solver`
   - `vendor/Math-Expression-Recognizer-and-Solver`

2) Ensure its `math-recog-model.h5` is present as per that project’s instructions.

The GUI will show a model choice “MER (External)” under Arithmetic when detected. The adapter is implemented in `src/integrations/mer_adapter.py`.

## Troubleshooting

- TensorFlow unavailable: CNN features require a compatible TensorFlow (e.g., TF 2.15 on Python 3.10/3.11). Without TF, use SVM/Random Forest/k-NN backends.
- Empty or low-confidence results: try a different segmentation method (GUI selector) or ensure the correct label mapping JSON exists in `models/`.
- Letters mode shows only k-NN: provide `cnn_model_nist_by_class.h5` and `label_mapping_nist_by_class.json` to enable CNN.
- Arithmetic mode missing: provide `cnn_model_arithmetic.h5` and `label_mapping_arithmetic.json`, or configure the external MER repo.

## Maintainer

- Sajaan Jayalath

Feel free to open issues or PRs for bugs and enhancements.

