# CNN-Based Handwritten Letter and Digit Classification

**Course:** Image Analysis and Object Recognition  
**University:** Bauhaus-Universität Weimar  
**Semester:** Summer Semester 2026

## Project Overview

This project implements handwritten letter and digit classification using convolutional neural networks (CNNs) in PyTorch.

The main tasks are:

- EMNIST Letters classification
- CNN architecture experiments
- Hyperparameter experiments
- Selection and evaluation of the best letter model
- Transfer learning from letters to MNIST digits
- Training a digit model from scratch for comparison
- Testing trained models on self-created handwritten images

The best letter model achieved approximately **93.51% test accuracy**.

The transfer-learning digit model achieved approximately **97.86% test accuracy**.

The from-scratch digit model achieved approximately **99.17% test accuracy**.

On the self-created handwritten test images:

- Letters: **25/26 = 96.15%**
- Digits: **9/10 = 90.00%**
- Overall: **34/36 = 94.44%**

## Project Structure

```text
handwritten-classification/
├── requirements.txt
├── models/
│   ├── best_letter_classifier.pth
│   └── best_digit_classifier.pth
│
├── results/
│   ├── figures/
│   ├── *.json
│   ├── *.csv
│   └── final_project_results_summary.txt
│
├── src/
│   ├── check_setup.py
│   ├── data_loader.py
│   ├── dataset_analysis.py
│   ├── digit_utils.py
│   ├── finalize_letter_model.py
│   ├── generate_result_summary.py
│   ├── inference_custom_images.py
│   ├── letter_experiment_utils.py
│   ├── models.py
│   ├── path_utils.py
│   ├── training_utils.py
│   ├── train_baseline.py
│   ├── train_complex.py
│   ├── train_feature_maps.py
│   ├── train_kernel_size.py
│   ├── train_learning_rate_low.py
│   ├── train_learning_rate_high.py
│   ├── train_batch_size_small.py
│   ├── train_batch_size_large.py
│   ├── train_epochs_long.py
│   ├── transfer_learning_digits.py
│   └── train_digits_from_scratch.py
│
└── test_images/
    ├── letters/
    └── digits/
```

## Dataset

Two torchvision datasets are used.

### EMNIST Letters

- 26 classes: A-Z
- 124,800 original training images
- 20,800 test images
- Training set split into:
  - 112,320 training images
  - 12,480 validation images
- Image size: 28 × 28
- Grayscale
- Stratified train/validation split
- Random seed: 42

EMNIST images are rotated and mirrored to restore their correct visual orientation.

### MNIST Digits

MNIST is used for the digit-classification and transfer-learning experiments.

## Letter Experiments

The project progressively modifies the CNN configuration.

| Experiment | Main Change |
|---|---|
| Baseline | One convolution layer |
| E1 | Deeper CNN |
| E2 | Reduced feature maps |
| E3 | 5 × 5 convolution kernels |
| E4 | Lower learning rate |
| E5 | Higher learning rate |
| E6 | Batch size 32 |
| E7 | Batch size 128 |
| E8 | 10 training epochs |

The selected E8 model uses:

```text
Input            : 1 × 28 × 28
Conv channels    : 16 → 32
Kernel size      : 5 × 5
Hidden units     : 128
Learning rate    : 0.001
Batch size       : 64
Epochs           : 10
Random seed      : 42
```

Best E8 validation accuracy:

```text
93.68%
```

Final letter test accuracy:

```text
93.51%
```

## Digit Transfer Learning

The selected letter model is adapted from 26 letter classes to 10 digit classes.

The final classification layer is replaced:

```text
26 outputs → 10 outputs
```

Earlier layers are frozen and only the new output layer is trained.

Results:

```text
Validation accuracy : 97.62%
Test accuracy       : 97.86%
Trainable parameters: 1,290
Total parameters    : 215,370
Training time       : 43.98 seconds
```

A control model with the same architecture was also trained from scratch.

```text
Validation accuracy : 98.95%
Test accuracy       : 99.17%
Training time       : 69.86 seconds
```

Transfer learning reduced training time by approximately **37.05%**, although the from-scratch model achieved **1.31 percentage points higher test accuracy**.

## Self-Created Image Testing

The final models were evaluated on 36 manually created handwritten images:

```text
26 letters
10 digits
```

Results:

```text
Letters : 25 / 26 = 96.15%
Digits  :  9 / 10 = 90.00%
Overall : 34 / 36 = 94.44%
```

The two incorrect predictions were:

```text
Letter Q → A
Digit 6  → 5
```

## Running the Project

Use Python 3.10 or newer. Run the following commands from the project root.
Install the dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

The scripts download the public EMNIST and MNIST datasets to `data/` when
needed. Downloaded datasets are excluded from version control. The two final
model checkpoints and 36 handwritten sample images are already included, so
you can try inference without retraining.

### Check the environment

```bash
python src/check_setup.py
```

### Inspect the dataset

```bash
python src/data_loader.py
python src/dataset_analysis.py
```

### View command-line options

Every major script supports `--help`.

Example:

```bash
python src/train_epochs_long.py --help
```

### Run the baseline

```bash
python src/train_baseline.py
```

### Run the selected E8 experiment

```bash
python src/train_epochs_long.py
```

### Finalize the selected letter model

```bash
python src/finalize_letter_model.py
```

### Train the transfer-learning digit model

```bash
python src/transfer_learning_digits.py
```

### Train the digit control model from scratch

```bash
python src/train_digits_from_scratch.py
```

### Test custom handwritten images

```bash
python src/inference_custom_images.py
```

### Regenerate the result summary

```bash
python src/generate_result_summary.py
```

## Reusable Command-Line Options

The refactored scripts allow important settings to be changed without modifying source code.

For example:

```bash
python src/train_epochs_long.py --batch-size 64 --learning-rate 0.001 --epochs 10 --seed 42
```

Custom image inference also accepts configurable model, image, CSV, and figure paths.

Run:

```bash
python src/inference_custom_images.py --help
```

to see all available options.

## Important Note

Run `train_epochs_long.py` before `finalize_letter_model.py` when reproducing
the training workflow: the intermediate E8 checkpoint is not bundled. Training
scripts using their default paths may overwrite the included models or results.

The `models/` directory contains the final selected letter and digit
classifiers, so retraining is not required to inspect the saved results or run
custom-image inference. The recorded metrics come from the original run.
