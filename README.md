# Handwritten Character Recognition System (HNRS)

## Overview

This repository contains the Handwritten Character Recognition System (HNRS)
developed by **Sajaan Jayalath** for the COS30018 Intelligent Systems unit.
HNRS recognises hand‑written digits (0‑9) and letters (A‑Z, a‑z) by combining
advanced image preprocessing with multiple machine learning models. The system
packages everything needed to draw characters, upload images, run recognition,
compare models, and retrain networks from raw datasets.

## Feature Highlights

- **Multi-target support** – switch instantly between digit and letter modes.
- **Three model families** – Convolutional Neural Network (CNN), Support Vector
  Machine (SVM), and Random Forest classifiers with ready-to-use weights.
- **Robust preprocessing** – deskewing, contrast enhancement, morphology, and
  adaptive thresholding tailored for noisy handwriting.
- **Intelligent segmentation** – contour, projection, and connected-component
  strategies with automatic method selection for multi-character inputs.
- **Tkinter GUI** – draw characters, visualise preprocessing/segmentation, and
  inspect per-character confidences in real time.
- **Training pipelines** – scripts for MNIST digit models and NIST by_class
  letter models, including metadata and label-mapping export.

## Repository Layout

```
.
├── MNIST_CSV/                  # Datasets (tracked with Git LFS)
├── models/                     # Saved model weights + metadata
├── sample_images/              # Example handwritten character images
├── src/
│   ├── data_loader.py          # Data loading helpers (MNIST, EMNIST, NIST)
│   ├── generate_synthetic_digits.py
│   ├── gui.py                  # Tkinter front-end
│   ├── image_preprocessing.py  # Preprocessing + augmentation utilities
│   ├── image_segmentation.py   # Segmentation + multi-character logic
│   ├── main.py                 # CLI / GUI entry point
│   ├── models.py               # CNN, SVM, Random Forest implementations
│   ├── train_models.py         # Digit (MNIST-style) training pipeline
│   └── train_letters.py        # Letter (NIST by_class) training pipeline
├── tests/                      # Optional automated checks
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Getting Started

### 1. Install prerequisites

- Python 3.10 or later is recommended (Python 3.8+ supported).
- Ensure [Git](https://git-scm.com/) and [Git LFS](https://git-lfs.github.com/)
  are installed on your machine.

### 2. Clone the repository

```bash
git clone https://github.com/SajaanJayalath/IS.git
cd IS
```

If a different folder name is preferred (e.g. `Intelligent_Systems_COS30018`),
rename the directory after cloning.

### 3. Pull large assets with Git LFS

The EMNIST/NIST datasets and related `.mat`/`.ubyte` files are tracked via Git
LFS. Run the following once inside the repository:

```bash
git lfs install
git lfs pull
```

This downloads the full datasets referenced by the project.

### 4. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Application

Launch the Tkinter GUI to draw or upload characters:

```bash
python src/main.py --gui
```

Workflow inside the GUI:

1. Choose **Digits** or **Letters** as the recognition target.
2. Select a model (CNN, SVM, Random Forest) and segmentation strategy.
3. Draw on the canvas or upload an image file.
4. Click **Recognize Drawing** or **Process Current Image** to view results.
5. Toggle the *Show preprocessing* / *Show segmentation* checkboxes to inspect
   intermediate steps.

## Command-line Utilities

The entry script also provides handy CLI options:

```bash
python src/main.py --help
python src/main.py --test-image path/to/sample.png --model cnn
python src/main.py --test-models
python src/main.py --check-deps
```

## Training the Models

Pre-trained weights are included, but you can retrain or fine-tune models:

### MNIST / Digit models

```bash
cd src
python train_models.py
```

Outputs are stored under `models/` (e.g. `cnn_model_mnist_csv.h5`) along with a
`label_mapping_mnist_csv.json` file for inference.

### NIST by_class / Letter models

```bash
cd src
python train_letters.py --data-dir data/by_class/by_class
```

Adjust CLI flags to include/exclude uppercase, lowercase, or digit classes, and
to tweak Random Forest/CNN hyperparameters. Trained weights and metadata are
written to `models/` with a corresponding `label_mapping_nist_by_class.json`.

## Machine Learning Models

| Model          | Accuracy* | Highlights                                   |
|----------------|-----------|----------------------------------------------|
| CNN            | 99.2%     | Data augmentation, batch norm, dropout       |
| SVM (RBF)      | 96.3%     | Fast inference, balanced accuracy/simplicity |
| Random Forest  | 95.1%     | Lightweight baseline, quick to train         |

\*Reported on MNIST validation splits during development. Letter accuracy will
depend on chosen NIST class subsets.

## Recognition Pipeline

1. **Preprocessing** – grayscale conversion, CLAHE, denoising, adaptive
   thresholding, morphology, optional deskew.
2. **Segmentation** – contour detection, connected components, or projection
   profiles with heuristics for touching characters.
3. **Normalization** – size normalisation, optional center-of-mass alignment,
   background polarity adjustment (digits vs letters), and scaling to [0, 1].
4. **Classification** – selected model predicts each character; ensemble mode
   (if multiple models loaded) averages probabilities.
5. **Result aggregation** – characters are ordered left-to-right and displayed
   with per-character confidences and overall text output.

## Dataset Notes

- `MNIST_CSV/` holds MNIST CSV exports plus EMNIST `.mat`/`.ubyte` files.
- `data/by_class/` (optional) should contain the extracted NIST Special
  Database 19 *by_class* dataset for letter training.
- Large files are stored using Git LFS. Collaborators must install Git LFS and
  run `git lfs pull` after cloning to obtain the datasets.

## Testing

Basic test scaffolding lives under `tests/`. Example usage:

```bash
python -m pytest -q          # if pytest is available
python tests/test_system.py  # custom integration script
```

## Troubleshooting

- **TensorFlow unavailable** – The CNN model needs TensorFlow 2.15 (or similar)
  for Python 3.10/3.11. Install using `pip install tensorflow==2.15.*` or stick
  with SVM/Random Forest models.
- **Large file checkout errors** – Ensure `git lfs install` and `git lfs pull`
  run successfully; without them you will only have pointer files.
- **Blank predictions** – Verify label mapping files in `models/` correspond to
  the active recognition mode and that the segmentation method detects all
  characters.

## Maintainer

- **Sajaan Jayalath** – sajjayalath@example.com *(replace with preferred contact)*

Contributions and bug reports are welcome via GitHub issues or pull requests on
[`SajaanJayalath/IS`](https://github.com/SajaanJayalath/IS).

