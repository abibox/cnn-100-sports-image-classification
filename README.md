# CNN Image Classification — 100 Sports Categories

A convolutional neural network for classifying images of **100 different sports** using **VGG16 transfer learning** with a two-phase fine-tuning strategy. The model is trained on the [100 Sports Image Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification) dataset from Kaggle and applies seven distinct optimisation techniques to maximise accuracy.

---

## Table of Contents

- [Task Overview](#task-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Optimisation Techniques](#optimisation-techniques)
- [Training Strategy](#training-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Prediction & Inference](#prediction--inference)
- [Getting Started](#getting-started)
- [Notebook Structure](#notebook-structure)
- [Outputs](#outputs)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Task Overview

```
┌──────────────┐      ┌──────────────────┐      ┌──────────────────────┐
│  Input Image │─────▶│  VGG16 + Custom  │─────▶│  Predicted Sport     │
│  (224×224)   │      │  Dense Layers    │      │  + Confidence Score  │
└──────────────┘      └──────────────────┘      └──────────────────────┘
```

- **Input:** An RGB image of a sport being played (224 × 224 pixels)
- **Output:** One of 100 sport categories (e.g., "basketball", "swimming", "football") with a confidence score

**Real-world applications:**
- Auto-tagging sports images on social media
- Organising sports photo libraries
- Content moderation for sports platforms
- Sports media analysis and cataloguing

---

## Dataset

**[100 Sports Image Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)** from Kaggle.

| Split | Images | Purpose |
|-------|-------:|---------|
| Train | 13,493 | Model training with data augmentation |
| Validation | 500 | Hyperparameter tuning and early stopping |
| Test | 500 | Final evaluation (no augmentation) |
| **Total** | **~14,500** | |

- **Classes:** 100 sport categories
- **Image format:** RGB, resized to 224 × 224
- **Categories include:** Ball sports (basketball, soccer, tennis), water sports (swimming, diving, surfing), combat sports (boxing, wrestling), winter sports (skiing, ice hockey), track & field events (running, javelin), and many more

The dataset is downloaded automatically via the `opendatasets` library using Kaggle API credentials.

---

## Model Architecture

The model uses **VGG16** pre-trained on ImageNet as a feature extractor, with custom classification layers on top:

```
VGG16 (pre-trained on ImageNet, top removed)
    │
    ▼
Global Average Pooling 2D
    │
    ▼
Batch Normalisation
    │
    ▼
Dense (512 units, ReLU)
    │
    ▼
Dropout (50%)
    │
    ▼
Dense (256 units, ReLU)
    │
    ▼
Dropout (30%)
    │
    ▼
Dense (100 units, Softmax) → Output
```

| Component | Details |
|-----------|---------|
| Base model | VGG16 (pre-trained on ImageNet — 1.2M images, 1000 classes) |
| Pooling | Global Average Pooling 2D |
| Hidden layers | 512 → 256 (both ReLU activated) |
| Regularisation | Batch Normalisation + Dropout (50% / 30%) |
| Output | 100-class softmax |
| Loss function | Categorical cross-entropy |
| Optimiser | Adam |

---

## Optimisation Techniques

Seven techniques are applied to improve model performance and reduce overfitting:

| # | Technique | Configuration | Purpose |
|---|-----------|---------------|---------|
| 1 | **Data Augmentation** | Rotation ±20°, width/height shift ±20%, horizontal flip, zoom ±20%, shear ±10% | Increase training data variety, reduce overfitting |
| 2 | **Transfer Learning** | VGG16 pre-trained on ImageNet with frozen base layers | Leverage learned feature representations |
| 3 | **Batch Normalisation** | Applied after global average pooling | Faster convergence, acts as mild regularisation |
| 4 | **Dropout** | 50% after first dense layer, 30% after second | Prevent overfitting by randomly deactivating neurons |
| 5 | **Learning Rate Scheduling** | `ReduceLROnPlateau`: factor 0.5, patience 3, min LR 1e-7 | Fine-grained optimisation near convergence |
| 6 | **Early Stopping** | Monitor `val_loss`, patience 5, restore best weights | Prevent overfitting, save training time |
| 7 | **Fine-tuning** | Unfreeze last 4 VGG16 layers in Phase 2 with lower LR | Adapt pre-trained features to the sports domain |

---

## Training Strategy

Training uses a **two-phase approach** to progressively adapt the model:

### Phase 1 — Feature Extraction (Base Model Frozen)

| Parameter | Value |
|-----------|-------|
| Epochs | 10 |
| Learning rate | 1e-3 |
| Trainable layers | Custom dense layers only |
| VGG16 base | Completely frozen |

The top classification layers learn to map VGG16's pre-trained features to the 100 sport categories.

### Phase 2 — Fine-Tuning (Last 4 Layers Unfrozen)

| Parameter | Value |
|-----------|-------|
| Epochs | 15 |
| Learning rate | 1e-5 (100× lower) |
| Trainable layers | Custom layers + last 4 VGG16 layers |
| VGG16 base | Partially unfrozen |

The final VGG16 convolutional blocks are unfrozen with a much lower learning rate to fine-tune the learned features for sport-specific patterns without destroying the pre-trained representations.

**Both phases** use early stopping, learning rate scheduling, and model checkpointing — the best model (by validation accuracy) is saved automatically.

---

## Evaluation Metrics

The model is evaluated on the held-out test set (500 images) using five metrics:

| Metric | Averaging | Description |
|--------|-----------|-------------|
| **Accuracy** | — | Overall fraction of correct predictions |
| **Precision** | Weighted | Proportion of positive predictions that are correct, per class |
| **Recall** | Weighted | Proportion of actual positives correctly identified, per class |
| **F1-Score** | Weighted | Harmonic mean of precision and recall |
| **AUC-ROC** | Weighted, one-vs-rest | Area under the ROC curve for multiclass (binarised) |

### Visualisations

The notebook produces:

- **Training curves** — accuracy and loss across both phases (combined plot)
- **Confusion matrix** — top 15 most frequent classes for readability
- **Sample predictions** — test images with predicted vs true labels and correct/incorrect markers
- **Per-class classification report** — precision, recall, and F1 for every sport category

---

## Prediction & Inference

The notebook includes a reusable function to classify any new image (local file or URL):

```python
predicted_sport, confidence = predict_sport_from_image(
    "path/to/image.jpg",   # or a URL
    model=model_cnn,
    class_names=class_names
)
# Output: "basketball", 94.3%
```

The function:
1. Loads the image from a local path or URL
2. Resizes to 224 × 224 and normalises pixel values
3. Handles both RGB and grayscale inputs
4. Returns the predicted sport name and confidence percentage
5. Displays the image with the prediction and a top-5 bar chart

---

## Getting Started

### Requirements

- **Hardware:** GPU recommended (Google Colab with T4/A100 or local CUDA GPU)
- **Python:** 3.8+

### Installation

```bash
pip install tensorflow keras opendatasets matplotlib seaborn scikit-learn pillow requests
```

### Kaggle API Setup

The notebook downloads the dataset via `opendatasets`. You'll need your Kaggle credentials:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **Create New Token**
2. When prompted by the notebook, enter your Kaggle username and API key

### Running

1. Open `TASK_2a_CNN_Image_Classification.ipynb` in Jupyter or Google Colab.
2. Run all cells sequentially — the notebook handles dataset download, training, evaluation, and inference.

---

## Notebook Structure

| Cell(s) | Section | Description |
|---------|---------|-------------|
| 0 | Introduction | Dataset description, optimisation techniques overview |
| 1–2 | Setup | Install dependencies, import libraries, set seeds |
| 3–4 | Data Loading | Download dataset from Kaggle, count images per split |
| 5–6 | Data Preparation | Configure `ImageDataGenerator` with augmentation, create train/val/test generators, visualise samples |
| 7 | Model Building | VGG16 transfer learning + custom dense head with BatchNorm and Dropout |
| 8 | Callbacks | Early stopping, LR scheduling, model checkpointing |
| 9 | Phase 1 Training | Train top layers with base model frozen (10 epochs, LR 1e-3) |
| 10 | Phase 2 Training | Unfreeze last 4 VGG16 layers, fine-tune (15 epochs, LR 1e-5) |
| 11 | Training Curves | Plot combined accuracy and loss across both phases |
| 12–15 | Evaluation | Test set predictions, metrics (accuracy/precision/recall/F1/AUC), confusion matrix, sample predictions |
| 16–17 | Summary | Optimisation techniques recap and final results |
| 18–20 | Interpretation | Prediction interpretation, sample predictions with images |
| 21–22 | Inference Function | `predict_sport_from_image()` for classifying new images from file or URL |
| 23 | Final Summary | Complete results and use-case summary |

---

## Outputs

| Artifact | Description |
|----------|-------------|
| `best_sports_cnn_model.keras` | Best model weights (saved by `ModelCheckpoint` on highest `val_accuracy`) |
| Training curves plot | Accuracy and loss across both training phases |
| Confusion matrix | Top-15 class heatmap |
| Classification report | Per-class precision, recall, F1 for all 100 categories |
| Sample prediction grid | Test images with predicted and true labels |

---

## Tech Stack

| Library | Role |
|---------|------|
| [TensorFlow / Keras](https://www.tensorflow.org/) | Model building, training, VGG16 transfer learning |
| [scikit-learn](https://scikit-learn.org/) | Evaluation metrics (accuracy, precision, recall, F1, AUC-ROC) |
| [opendatasets](https://github.com/JovianHQ/opendatasets) | Kaggle dataset download |
| [Matplotlib / Seaborn](https://matplotlib.org/) | Training curves, confusion matrix, prediction visualisations |
| [Pillow](https://pillow.readthedocs.io/) | Image loading for inference function |
| [NumPy / Pandas](https://numpy.org/) | Data manipulation |

---

## License

This project is for educational and research purposes.
