# Waste Classification — Complete Code Documentation
## CNN & DCNN Notebooks: Line-by-Line Explanation

---

> **Purpose of this document:**  
> Every cell, every line, every parameter — explained in plain language.  
> Why it was written that way, what it does, and the logic behind it.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [CNN Notebook — Cell-by-Cell](#2-cnn-notebook--cell-by-cell)
   - [Cell 1 — Dependencies](#cell-1--dependencies)
   - [Cell 2 — Imports & Configuration](#cell-2--imports--global-configuration)
   - [Cell 3 — Dataset Verification](#cell-3--dataset-verification)
   - [Cell 4 — Augmentation Pipeline](#cell-4--data-augmentation--preprocessing-pipeline)
   - [Cell 5 — Visualize Augmented Samples](#cell-5--visualize-augmented-samples)
   - [Cell 6 — CNN Architecture](#cell-6--cnn-architecture)
   - [Cell 7 — Compile & Callbacks](#cell-7--model-compilation--callbacks)
   - [Cell 8 — Training](#cell-8--model-training)
   - [Cell 9 — Training Curves](#cell-9--training-curves-visualization)
   - [Cell 10 — Evaluation](#cell-10--full-model-evaluation)
   - [Cell 11 — Confusion Matrix](#cell-11--confusion-matrix)
   - [Cell 12 — ROC Curve & Metrics](#cell-12--classification-report--roc-curve)
   - [Cell 13 — Sample Predictions](#cell-13--sample-predictions)
   - [Cell 14 — Feature Maps](#cell-14--feature-map-visualization)
   - [Cell 15 — Report Card](#cell-15--final-report-card)
   - [Cell 16 — Save Model](#cell-16--save-model)
   - [Cell 17 — Inference Helper](#cell-17--inference-helper)
3. [DCNN Notebook — Cell-by-Cell](#3-dcnn-notebook--cell-by-cell)
   - [Cell 1 — Imports & Configuration](#dcnn-cell-1--imports--configuration)
   - [Cell 2 — Dataset Verification](#dcnn-cell-2--dataset-verification)
   - [Cell 3 — Augmentation & Generators](#dcnn-cell-3--augmentation-pipeline--generators)
   - [Cell 4 — Class Sample Grid](#dcnn-cell-4--class-sample-grid)
   - [Cell 5 — DCNN Architecture](#dcnn-cell-5--dcnn-architecture)
   - [Cell 6 — Compile & Callbacks](#dcnn-cell-6--compile--callbacks)
   - [Cell 7 — Training](#dcnn-cell-7--training)
   - [Cell 8 — Training Curves](#dcnn-cell-8--training-curves)
   - [Cell 9 — Evaluation](#dcnn-cell-9--test-evaluation)
   - [Cell 10 — Confusion Matrix](#dcnn-cell-10--confusion-matrix-12x12)
   - [Cell 11 — Classification Report](#dcnn-cell-11--classification-report)
   - [Cell 12 — ROC Curves (OvR)](#dcnn-cell-12--roc-auc-one-vs-rest)
   - [Cell 13 — Sample Predictions](#dcnn-cell-13--sample-predictions)
   - [Cell 14 — Grad-CAM](#dcnn-cell-14--grad-cam-heatmap)
   - [Cell 15 — CNN vs DCNN Comparison](#dcnn-cell-15--cnn-vs-dcnn-comparison)
   - [Cell 16 — Report Card](#dcnn-cell-16--final-report-card)
   - [Cell 17 — Save Model](#dcnn-cell-17--save-model)
   - [Cell 18 — Inference Helper](#dcnn-cell-18--inference-helper)
4. [Key Concepts Reference](#4-key-concepts-reference)
5. [CNN vs DCNN Decision Logic](#5-cnn-vs-dcnn-decision-logic)

---

## 1. Project Overview

### What we are building

Two deep learning models that take a photo of waste as input and output a classification label:

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| CNN | Any waste image | Organic OR Recyclable | First-pass sorting gate |
| DCNN | Any waste image | 1 of 12 waste categories | Facility routing |

### Why two separate models?

The CNN solves the binary problem — a simpler, faster model for the easiest decision.  
The DCNN solves the detailed problem — a deeper, more powerful model for fine-grained classification.  
Together they form a two-stage pipeline: CNN gates the stream, DCNN routes each item.

### Technology Stack

```
TensorFlow 2.x / Keras    ← deep learning framework
NumPy                     ← numerical computation
Matplotlib + Seaborn      ← visualization
scikit-learn              ← evaluation metrics
Pillow (PIL)              ← image loading and manipulation
OpenCV (cv2)              ← Grad-CAM image processing (DCNN only)
```

---

## 2. CNN Notebook — Cell-by-Cell

---

### Cell 1 — Dependencies

```python
# Run this cell only if dependencies are not installed
# !pip install tensorflow matplotlib scikit-learn seaborn pillow -q
print('Dependencies ready.')
```

**Why this cell exists:**  
When running on Kaggle, all major packages are pre-installed, so the install line is commented out.  
The `print` confirms execution reached this point — useful for debugging in notebook environments  
where silent cells can be skipped without obvious feedback.

**The `-q` flag** on pip suppresses verbose install output, keeping the notebook clean.

**Kaggle path note in the comment** is critical — new users often cannot find their dataset  
because Kaggle mounts it at a path like `/kaggle/input/dataset-name/`, not the current directory.

---

### Cell 2 — Imports & Global Configuration

#### Section A: Library Imports

```python
import os
import random
import warnings
import numpy as np
```

- `os` — used for file path construction (`os.path.join`), directory listing (`os.listdir`),  
  and folder creation (`os.makedirs`). Keeps code OS-agnostic (works on Windows and Linux).
- `random` — used for sampling random images for visualization. `random.sample()` picks  
  without replacement, unlike `random.choice()` which can repeat.
- `warnings` — `warnings.filterwarnings('ignore')` suppresses TensorFlow's verbose deprecation  
  messages that clutter notebook output during training.
- `numpy as np` — the fundamental numerical library. Used for array operations, reshaping  
  predictions, computing metrics, and passing data to matplotlib.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
```

- `matplotlib.pyplot` — primary plotting library. `plt.subplots()` creates multi-panel figures.
- `matplotlib.patches` — used in Cell 13 to create color legend patches (green=correct,  
  red=incorrect) for the sample prediction visualization.
- `seaborn` — built on matplotlib, used specifically for `sns.heatmap()` in the confusion matrix.  
  Its default styling and annotation handling are far cleaner than raw matplotlib for heatmaps.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
```

- `tensorflow` — the deep learning framework. Version 2.x uses eager execution by default  
  (operations run immediately, not deferred). This makes debugging much easier.
- `keras` — TensorFlow's high-level API. Lives inside TF2 as `tf.keras`. Provides `Sequential`  
  and `Functional` model building APIs, layer definitions, and training loops.
- `layers` — contains all neural network layer types: `Conv2D`, `Dense`, `MaxPooling2D`,  
  `Dropout`, `BatchNormalization`, `GlobalAveragePooling2D`, `Activation`.
- `models` — contains `Sequential` (linear stack of layers) and `Model` (functional graph).  
  CNN uses `Sequential` for simplicity; DCNN uses `Model` for residual connections.
- `regularizers` — imported but not directly used in this CNN. Available if you want  
  to add L2 weight regularization: `kernel_regularizer=regularizers.l2(1e-4)`.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
```

- `ImageDataGenerator` — creates an object that reads images from disk in batches,  
  applies augmentation transforms on-the-fly, and yields (image_batch, label_batch) tuples.  
  This is critical because 25,000 images at 224×224×3 would be ~14GB if loaded all at once.
- `load_img` — reads a single image file from disk into a PIL Image object.  
  Used in Cell 13 (predictions) and Cell 14 (feature maps) for individual image inference.
- `img_to_array` — converts a PIL Image to a NumPy array of shape (H, W, C).  
  Required before passing an image to model.predict().

```python
from tensorflow.keras.optimizers import Adam
```

- `Adam` — Adaptive Moment Estimation optimizer. Combines the benefits of RMSProp (adaptive  
  learning rates per parameter) and momentum. It is the default choice for deep learning  
  because it converges faster and more reliably than plain SGD. We initialize it with  
  `learning_rate=1e-3` (0.001) which is Adam's canonical starting point.

```python
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
)
```

These four callbacks run automatically at the end of every epoch:

- `ModelCheckpoint` — saves the model weights to disk whenever the monitored metric improves.  
  Without this, if training crashes at epoch 25 of 30, all progress is lost.
- `EarlyStopping` — stops training when validation loss stops improving for N epochs (patience).  
  Prevents wasted computation and overfitting to the training set.
- `ReduceLROnPlateau` — halves the learning rate when loss plateaus for N epochs.  
  Allows the model to make coarser updates initially and fine-grained updates near convergence.
- `CSVLogger` — writes epoch metrics (loss, accuracy, val_loss, val_accuracy) to a CSV file.  
  Useful for post-training analysis and comparison across experiments.

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, roc_curve
)
```

scikit-learn provides evaluation functions that TensorFlow does not have built in:

- `classification_report` — prints per-class precision, recall, F1, and support in a table.
- `confusion_matrix` — returns an NxN matrix of true vs predicted labels.
- `accuracy_score` — fraction of correctly classified samples.
- `f1_score` — harmonic mean of precision and recall. Better than accuracy for imbalanced classes.
- `roc_auc_score` — area under the ROC curve. 1.0 = perfect, 0.5 = random guess.
- `roc_curve` — returns (fpr, tpr, thresholds) arrays for plotting the ROC curve.

#### Section B: Reproducibility

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

**Why set seeds?** Deep learning models involve randomness at multiple points:  
random weight initialization, random augmentation transforms, random batch sampling,  
and random dropout masks. Without fixed seeds, two runs of the same code will produce  
different models with different accuracies. Seeds make results reproducible — someone else  
running your notebook should get the same numbers.

`42` is a conventional choice (from "The Hitchhiker's Guide to the Galaxy") with no  
mathematical significance — any integer works.

#### Section C: Hyperparameters

```python
IMG_HEIGHT   = 224
IMG_WIDTH    = 224
```

**Why 224×224?**  
This is the standard input size used by VGG, ResNet, EfficientNet, and most ImageNet-trained  
models. Using 224 makes future transfer learning experiments directly compatible.  
The original images are various sizes — `ImageDataGenerator` resizes all of them to this.

```python
BATCH_SIZE   = 32
```

**Why 32?**  
Batch size controls how many images are processed together in one forward/backward pass.  
- Too small (e.g., 4): noisy gradient estimates, slow training, but uses less GPU memory.  
- Too large (e.g., 256): stable gradients but requires more memory and can generalize worse.  
- 32 is the sweet spot empirically confirmed across thousands of papers.

```python
EPOCHS       = 30
```

Maximum number of complete passes through the training data. `EarlyStopping` will stop  
before 30 if validation loss plateaus, so this is a ceiling, not a fixed number.

```python
LEARNING_RATE = 1e-3
```

Controls the step size at each optimization update. `1e-3 = 0.001`.  
Adam's default and most commonly successful starting point for CNNs.

```python
NUM_CLASSES  = 2
DROPOUT_RATE = 0.4
```

- `NUM_CLASSES = 2`: Organic and Recyclable. Controls the final Dense layer size.
- `DROPOUT_RATE = 0.4`: 40% of neurons in the Dense layers are randomly zeroed during  
  training. This forces the network to learn redundant representations and prevents overfitting.

```python
CLASS_NAMES  = ['Organic', 'Recyclable']
CLASS_COLORS = ['#3B6D11', '#185FA5']
```

- `CLASS_NAMES`: displayed in plots and confusion matrix. Must match the alphabetical order  
  that `ImageDataGenerator` assigns (`O` comes before `R`, so index 0=Organic, 1=Recyclable).
- `CLASS_COLORS`: green for organic (nature), blue for recyclable (sustainability).  
  Used consistently across all visualizations for visual coherence.

#### Section D: Dataset Paths

```python
TRAIN_DIR = '/kaggle/input/waste-classification-data/DATASET/TRAIN'
TEST_DIR  = '/kaggle/input/waste-classification-data/DATASET/TEST'
```

**Kaggle-specific paths.** When you add the dataset to a Kaggle notebook, it mounts at  
`/kaggle/input/<dataset-slug>/`. The dataset internally has `DATASET/TRAIN/O/` and  
`DATASET/TRAIN/R/` subfolders. These paths tell `ImageDataGenerator.flow_from_directory()`  
where to look. Modify for local use.

---

### Cell 3 — Dataset Verification

```python
def count_images(directory):
    counts = {}
    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            n = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            counts[class_name] = n
    return counts
```

**What this does:**  
Walks through each class subfolder and counts image files.

**Line-by-line:**
- `sorted(os.listdir(directory))` — lists folder contents alphabetically. Sorted to ensure  
  consistent class ordering across different operating systems.
- `os.path.join(directory, class_name)` — builds full path. Never use string concatenation  
  (`directory + "/" + class_name`) because the separator differs on Windows (`\`) vs Linux (`/`).
- `if os.path.isdir(class_path)` — safety check to skip any stray files (like `.DS_Store`  
  on macOS) that are not class folders.
- `f.lower().endswith(('.jpg', '.jpeg', '.png'))` — case-insensitive extension check.  
  Some datasets have `.JPG` in uppercase — `.lower()` handles that.

**The bar chart that follows** uses different colors per class (`CLASS_COLORS`) and annotates  
each bar with its count. `ax.set_ylim(0, max(values) * 1.15)` adds 15% headroom above  
the tallest bar so annotation text is not cut off.

**Why verify the dataset before training?**  
Common issues: wrong path, missing folders, class imbalance. Catching these before a  
30-epoch training run saves hours.

---

### Cell 4 — Data Augmentation & Preprocessing Pipeline

#### Training Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale            = 1.0 / 255.0,
    rotation_range     = 20,
    width_shift_range  = 0.15,
    height_shift_range = 0.15,
    shear_range        = 0.10,
    zoom_range         = 0.15,
    horizontal_flip    = True,
    vertical_flip      = False,
    brightness_range   = [0.85, 1.15],
    fill_mode          = 'nearest',
    validation_split   = 0.15
)
```

**`rescale = 1.0 / 255.0`**  
Raw images have pixel values 0–255 (uint8). Neural networks work best with inputs in [0, 1]  
or [-1, 1]. Dividing by 255 normalizes to [0, 1]. Without this, the large magnitude gradients  
would destabilize training — BatchNorm would have to compensate heavily.

**`rotation_range = 20`**  
Randomly rotates each image by ±20 degrees. A plastic bottle photographed slightly tilted  
is still a plastic bottle. Teaching the model to ignore orientation improves generalization.

**`width_shift_range = 0.15` and `height_shift_range = 0.15`**  
Randomly shifts the image horizontally/vertically by up to 15% of its dimension.  
Simulates the waste item not being perfectly centered in the camera frame.

**`shear_range = 0.10`**  
Applies a shear transformation — slants the image. Simulates viewing an item at an angle.

**`zoom_range = 0.15`**  
Randomly zooms in or out by up to 15%. Simulates items at different distances from the camera.

**`horizontal_flip = True`**  
Randomly mirrors the image left-to-right. A banana peel flipped horizontally is still organic.  
Doubles effective dataset size with a trivial transform.

**`vertical_flip = False`**  
Intentionally disabled for CNN. While DCNN enables it for waste that can appear upside-down,  
for this binary task it adds unnecessary complexity — organic waste items in their natural  
orientation are more clearly identifiable.

**`brightness_range = [0.85, 1.15]`**  
Randomly adjusts brightness by ±15%. Simulates different lighting conditions:  
a dark recycling bin vs. a bright outdoor environment.

**`fill_mode = 'nearest'`**  
After rotation or shift, empty pixels appear at the image borders. `'nearest'` fills them  
by repeating the nearest edge pixel. Alternatives: `'reflect'`, `'wrap'`, `'constant'`.  
`'nearest'` is generally safest as it does not introduce unrealistic pixel patterns.

**`validation_split = 0.15`**  
Reserves 15% of the TRAIN folder as a validation set. This split is done before augmentation  
is applied — validation images are NEVER augmented (only rescaled), because we want  
validation to reflect real-world performance, not augmented versions.

#### Test Generator

```python
val_test_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)
```

**Only rescaling — no augmentation.** This is critical. If you augment the test set,  
your reported accuracy will not reflect real deployment performance. The model needs to  
be evaluated on images that look exactly like what it will see in production.

#### Flow from Directory

```python
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size   = (IMG_HEIGHT, IMG_WIDTH),
    batch_size    = BATCH_SIZE,
    class_mode    = 'binary',
    subset        = 'training',
    shuffle       = True,
    seed          = SEED,
    interpolation = 'bilinear'
)
```

- `target_size = (IMG_HEIGHT, IMG_WIDTH)` — all images resized to 224×224 on load.
- `class_mode = 'binary'` — labels returned as 0 or 1 (float). Required when using  
  `binary_crossentropy` loss and a Sigmoid output neuron.
- `subset = 'training'` — uses the 85% split (since `validation_split=0.15` was set).
- `shuffle = True` — randomizes batch order each epoch. If False, the model sees the same  
  sequence every epoch, potentially overfitting to the order.
- `seed = SEED` — makes shuffle reproducible, so the validation split is the same each run.
- `interpolation = 'bilinear'` — algorithm used when resizing images. Bilinear is a good  
  balance between quality and speed. Alternatives: `'nearest'` (faster, blockier),  
  `'bicubic'` (slower, smoother).

```python
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    ...
    shuffle = False
)
```

**`shuffle = False` on test generator is non-negotiable.**  
After `model.predict()`, we compare predictions to `test_generator.classes`.  
If shuffle=True, the predictions and the true labels would be in different orders,  
making the confusion matrix completely wrong.

#### Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight
class_weight_values = compute_class_weight(
    class_weight = 'balanced',
    classes      = np.array([0, 1]),
    y            = train_generator.classes
)
CLASS_WEIGHTS = dict(enumerate(class_weight_values))
```

**Why class weights?**  
The dataset has 13,966 organic vs 11,111 recyclable images — a 56/44 split. Without  
compensation, the model will bias toward predicting "Organic" more often.

`'balanced'` mode computes: `weight = total_samples / (n_classes * samples_in_class)`.  
The minority class (Recyclable) gets a higher weight, so misclassifying it costs more  
in the loss function. This forces the model to take both classes equally seriously.

---

### Cell 5 — Visualize Augmented Samples

```python
batch_images, batch_labels = next(generator)
```

**`next()` on a generator** fetches one batch of augmented images. These are already  
transformed (rotated, flipped, etc.) and normalized to [0, 1].

```python
axes[i].imshow(batch_images[i])
```

Matplotlib's `imshow` displays the (224, 224, 3) float array as an image. Since the  
values are already in [0, 1], it renders correctly without additional normalization.

**Purpose of this cell:** Verify augmentation is working as expected before spending  
time training. If rotation_range=90 was accidentally set too high, you would see  
wildly distorted images here and catch the mistake early.

---

### Cell 6 — CNN Architecture

#### Architecture Overview

```
Input (224×224×3)
   ↓
[Conv Block 1] → Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
   ↓
[Conv Block 2] → Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
   ↓
[Conv Block 3] → Conv2D(128) × 2 → BN → ReLU → MaxPool → Dropout(0.30)
   ↓
[Conv Block 4] → Conv2D(256) × 2 → BN → ReLU → MaxPool → Dropout(0.35)
   ↓
GlobalAveragePooling2D
   ↓
Dense(512) → BN → ReLU → Dropout(0.4)
   ↓
Dense(128) → ReLU → Dropout(0.3)
   ↓
Dense(1) → Sigmoid → [0, 1] probability
```

#### Line-by-Line Architecture Explanation

```python
model = models.Sequential(name='WasteCNN')
```

`Sequential` means layers are stacked linearly — the output of one layer goes directly  
into the next. This is sufficient for CNNs without skip connections. `name='WasteCNN'`  
labels the model in summary output.

```python
model.add(layers.Input(shape=input_shape))
```

Declares the input tensor shape (224, 224, 3). Not a computational layer — just tells  
Keras the expected input dimensions so it can build the computation graph.

```python
# Conv Block 1
model.add(layers.Conv2D(32, (3, 3), padding='same', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
```

**`Conv2D(32, (3, 3))`** — 32 filters, each 3×3 pixels.  
- Each filter learns to detect one type of pattern (an edge, a texture, a color gradient).  
- 32 filters means the layer extracts 32 different feature maps from the input.  
- Block 1 uses 32 (fewest) because early layers detect simple low-level features  
  (horizontal edges, color blobs) that do not require many filters.  
- Block 4 uses 256 (most) because deep layers detect complex semantic features  
  (object parts, material textures) that require more representational capacity.

**`padding='same'`** — adds zero-padding around the image borders so the output  
feature map is the same height×width as the input. Without it (`padding='valid'`),  
each conv layer would shrink the spatial dimensions, causing information loss.

**`use_bias=False`** — disables the bias term in Conv2D because BatchNormalization  
(the next layer) has its own learnable bias (`beta` parameter). Adding both is redundant  
and wastes parameters.

**`BatchNormalization()`** — normalizes the activations of the previous layer for each  
batch. It:
1. Subtracts the batch mean and divides by batch standard deviation.
2. Then applies learnable scale (`gamma`) and shift (`beta`) parameters.

**Why BatchNorm?**  
- Stabilizes training by preventing internal covariate shift (the distribution of each  
  layer's inputs changing during training).
- Acts as a mild regularizer (slight generalization improvement).
- Allows higher learning rates, speeding up training.
- Placed BEFORE the activation function (Conv → BN → ReLU pattern).

**`Activation('relu')`** — Rectified Linear Unit: `f(x) = max(0, x)`.  
- Negative values become 0, positive values pass through unchanged.  
- Non-linear activation is what allows neural networks to learn complex functions.  
  Without it, stacking layers would just be matrix multiplication — always linear.  
- ReLU is preferred over sigmoid/tanh in hidden layers because it does not suffer  
  from the vanishing gradient problem at large values.

```python
model.add(layers.MaxPooling2D((2, 2)))
```

Takes the maximum value in each 2×2 window, reducing spatial dimensions by half.  
After MaxPool, a 224×224 image becomes 112×112. After 4 MaxPools, it becomes 14×14.

**Why MaxPooling?**  
1. Reduces computation — fewer pixels to process in subsequent layers.  
2. Provides translation invariance — a feature detected 2 pixels to the right still  
   activates the same max-pooled output.
3. Progressively grows the receptive field — later conv layers effectively "see"  
   more of the original image.

```python
model.add(layers.Dropout(0.25))
```

During training, randomly sets 25% of the layer's output neurons to 0 each forward pass.  
This forces the network to not rely on any single neuron or feature, acting as regularization.

**Dropout rate increases with depth (0.25 → 0.25 → 0.30 → 0.35)** because deeper layers  
have more parameters and are at higher risk of overfitting.

**Dropout is DISABLED during inference** — `model.predict()` automatically uses all neurons.

#### Head Architecture

```python
model.add(layers.GlobalAveragePooling2D())
```

Takes the spatial feature maps from the last conv block (shape: 14×14×256) and computes  
the average of each filter's 14×14 values, producing a 256-element vector.

**Why GlobalAveragePooling instead of Flatten?**  
- `Flatten(14×14×256)` produces a 50,176-element vector → Dense(512) needs 50,176 × 512 = 25.7M parameters.  
- `GlobalAveragePooling(14×14×256)` produces a 256-element vector → Dense(512) needs only 256 × 512 = 131K parameters.  
- Fewer parameters = less overfitting, faster training.

```python
model.add(layers.Dense(512, use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(DROPOUT_RATE))  # 0.4
```

The first fully-connected (Dense) layer.  
512 neurons learn to combine the 256 pooled features into higher-level representations  
relevant to the binary classification task.

`use_bias=False` + `BatchNormalization()` — same reasoning as in Conv layers.

```python
model.add(layers.Dense(128))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.3))
```

Second Dense layer. Compresses 512 features to 128 — a funnel shape.  
No BatchNorm here because it's close to the output and the additional normalization  
can interfere with the final probability calibration.

```python
model.add(layers.Dense(1, activation='sigmoid'))
```

**The output layer.**  
- `1 neuron` — outputs a single scalar (probability).  
- `sigmoid` — squashes output to (0, 1). Values > 0.5 → Recyclable, values < 0.5 → Organic.  
- The choice of 1 neuron + sigmoid vs 2 neurons + softmax is equivalent for binary  
  classification. 1 neuron + sigmoid is more compact and the standard choice.

---

### Cell 7 — Model Compilation & Callbacks

```python
model.compile(
    optimizer = Adam(learning_rate=LEARNING_RATE),
    loss      = 'binary_crossentropy',
    metrics   = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)
```

**`optimizer = Adam(learning_rate=LEARNING_RATE)`**  
Adam updates each weight parameter using its own adaptive learning rate.  
It maintains two running averages per parameter:  
- First moment (mean of gradients) — like momentum.  
- Second moment (mean of squared gradients) — like RMSProp.  
This makes it more efficient than vanilla SGD on most deep learning tasks.

**`loss = 'binary_crossentropy'`**  
The loss function for binary classification.  
Formula: `L = -[y * log(p) + (1-y) * log(1-p)]`  
- When true label `y=1` (Recyclable) and prediction `p` is close to 1 → loss is near 0.  
- When `y=1` and `p` is close to 0 → loss is very high (penalizes confident wrong predictions heavily).  
- Log transform means errors near 0 or 1 are penalized much more than errors near 0.5.

**`Precision`** — of all items predicted as Recyclable, what fraction actually is?  
`TP / (TP + FP)`

**`Recall`** — of all actual Recyclable items, what fraction did we catch?  
`TP / (TP + FN)`

**`AUC`** — Area Under the ROC Curve. Measures discrimination ability across all  
classification thresholds, not just at 0.5. More robust than accuracy for imbalanced data.

#### Callbacks

```python
ModelCheckpoint(
    filepath        = 'checkpoints/waste_cnn_best.keras',
    monitor         = 'val_accuracy',
    save_best_only  = True,
    mode            = 'max',
    verbose         = 1
)
```

- `monitor = 'val_accuracy'` — saves when validation accuracy improves.  
- `save_best_only = True` — overwrites the file with each improvement, keeping only  
  the best weights. Without this, every epoch's weights would be saved, consuming disk space.  
- `mode = 'max'` — accuracy should go up, so "best" means highest.  
  For loss monitoring, use `mode = 'min'`.

```python
EarlyStopping(
    monitor              = 'val_loss',
    patience             = 7,
    restore_best_weights = True,
    verbose              = 1
)
```

- Monitors `val_loss` (not `val_accuracy`) because loss is a smoother signal.  
  Accuracy can plateau while loss still decreases meaningfully.
- `patience = 7` — waits 7 epochs of no improvement before stopping.  
  Too low (e.g., 3) stops prematurely; too high (e.g., 20) wastes time.
- `restore_best_weights = True` — after stopping, rolls back to the weights from the  
  epoch where val_loss was lowest, not the final epoch's weights.

```python
ReduceLROnPlateau(
    monitor   = 'val_loss',
    factor    = 0.5,
    patience  = 4,
    min_lr    = 1e-7,
    verbose   = 1
)
```

- `factor = 0.5` — multiplies the learning rate by 0.5 (halves it) on each trigger.  
- `patience = 4` — reduces LR after 4 epochs of stagnant val_loss.  
- `min_lr = 1e-7` — never reduces below this. Prevents the LR from reaching zero  
  and stopping all learning.

**The interplay of EarlyStopping and ReduceLR:**  
ReduceLR fires at epoch 20 → LR drops → model starts improving again → EarlyStopping  
resets its counter. This pattern allows deep convergence that would otherwise plateau.

---

### Cell 8 — Model Training

```python
history = model.fit(
    train_generator,
    epochs           = EPOCHS,
    validation_data  = val_generator,
    callbacks        = callback_list,
    class_weight     = CLASS_WEIGHTS,
    verbose          = 1
)
```

- `train_generator` — yields batches of (augmented_images, labels) indefinitely.  
  Keras knows one epoch is complete when `len(train_generator)` batches have been consumed.
- `validation_data = val_generator` — evaluated at end of each epoch with no augmentation.
- `class_weight = CLASS_WEIGHTS` — the imbalance correction computed in Cell 4.  
  Passed here rather than in the generator because it affects the loss computation,  
  not the data loading.
- `verbose = 1` — shows a progress bar with metrics for each epoch.

**`history` object** stores all metric values per epoch in `history.history`,  
a dict like `{'loss': [...], 'val_loss': [...], 'accuracy': [...], 'val_accuracy': [...]}`.  
Used in Cell 9 for plotting curves.

---

### Cell 9 — Training Curves Visualization

```python
best_epoch = np.argmax(history.history['val_accuracy']) + 1
```

`np.argmax` returns the index of the maximum value. `+1` converts from 0-based index  
to 1-based epoch number. Used to draw the red vertical line on all three plots.

```python
ax.axvline(best_e, color='#E24B4A', linestyle=':', lw=1.5, label=f'Best epoch ({best_e})')
```

A vertical dotted red line at the best epoch. Visual anchor that shows when the model  
achieved its peak validation performance and where EarlyStopping fired.

**Why plot both train and validation curves?**  
- Train accuracy much higher than val accuracy → overfitting.  
- Both curves still rising → underfitting (need more epochs or bigger model).  
- Both curves converging at similar high values → good generalization.  
- Val loss going up while train loss goes down → classic overfitting signal.

---

### Cell 10 — Full Model Evaluation

```python
model.load_weights('checkpoints/waste_cnn_best.keras')
```

**Critical step.** Loads the weights from when val_accuracy was highest, not the  
final epoch's weights. Without this, if the model overfit in the last few epochs,  
evaluation would use suboptimal weights.

```python
y_pred_proba = model.predict(test_generator, verbose=1)
y_pred       = (y_pred_proba > 0.5).astype(int).flatten()
y_true       = test_generator.classes
```

- `model.predict()` returns raw probabilities (shape: [N, 1]) in range [0, 1].
- `> 0.5` applies the classification threshold — values above 0.5 become Recyclable (1),  
  below become Organic (0).
- `.astype(int).flatten()` converts boolean array to integers and removes the extra  
  dimension so shape is [N] not [N, 1].
- `test_generator.classes` — the true integer labels in the same order as predictions.  
  This works correctly ONLY because `shuffle=False` was set in the test generator.

---

### Cell 11 — Confusion Matrix

```python
cm = confusion_matrix(y_true, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
```

`cm` has shape (2, 2):
```
                 Predicted Organic  Predicted Recyclable
Actual Organic        TN                  FP
Actual Recyclable     FN                  TP
```

`cm_pct`: Divides each row by its row sum — converts counts to percentages.  
`axis=1` means sum across columns (per row). `[:, np.newaxis]` reshapes the 1D  
sum array to (2, 1) for proper broadcasting during division.

```python
tn, fp, fn, tp = cm.ravel()
```

`cm.ravel()` flattens the 2×2 matrix to [TN, FP, FN, TP] for easy unpacking.

- **Sensitivity (Recall) = TP / (TP + FN)** — fraction of actual recyclables correctly caught.
- **Specificity = TN / (TN + FP)** — fraction of actual organics correctly identified.

---

### Cell 12 — Classification Report & ROC Curve

```python
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
```

- `fpr` (False Positive Rate) = FP / (FP + TN) at each threshold.
- `tpr` (True Positive Rate = Recall) = TP / (TP + FN) at each threshold.
- The ROC curve sweeps from threshold=1.0 (all predictions negative) to threshold=0.0  
  (all predictions positive), plotting all (FPR, TPR) pairs.

**Why plot the ROC curve?**  
The default 0.5 threshold may not be optimal. If false negatives (missed recyclables)  
are more costly than false positives, you can choose a lower threshold from the curve  
to improve recall at the cost of precision.

**AUC = 0.5**: no better than random guessing.  
**AUC = 1.0**: perfect classification at some threshold.  
**Typical good range for waste classification**: 0.93–0.99.

---

### Cell 13 — Sample Predictions

```python
border_color = '#1D9E75' if correct else '#E24B4A'
for spine in axes[i].spines.values():
    spine.set_edgecolor(border_color)
    spine.set_linewidth(3)
```

**Why colored borders?** Instantly communicates correct/incorrect prediction without  
reading the title text. Green border = right, red border = wrong — matching traffic light  
intuition. This visualization technique is common in medical imaging papers.

```python
confidence = pred_proba if pred_label == 1 else (1 - pred_proba)
```

`pred_proba` is the probability of being Recyclable (the positive class).  
If the model predicts Organic (pred_label=0), the confidence is `1 - pred_proba`  
(i.e., the probability of being Organic).  
This ensures confidence always reflects how certain the model is about its actual prediction.

---

### Cell 14 — Feature Map Visualization

```python
activation_model = keras.Model(
    inputs  = model.inputs,
    outputs = [l.output for l in conv_layers[:4]]
)
activations = activation_model.predict(img_tensor, verbose=0)
```

Creates a "sub-model" that takes the same input but outputs the activations of  
the first 4 Conv2D layers simultaneously. This is the standard technique for  
inspecting what each convolutional block has learned.

```python
avg_activation = np.mean(act[0], axis=-1)
```

Averages across the filter dimension. A single block activation has shape (H, W, n_filters).  
Averaging across filters collapses this to (H, W) — a single grayscale map showing  
which spatial regions were most activated on average.

**What you expect to see:**  
- **Block 1**: Edges, color gradients, basic textures — fine-grained, high resolution.  
- **Block 2**: Shapes, curves, texture patterns — slightly coarser.  
- **Block 3-4**: Object parts, material-level features — coarse, lower resolution,  
  but semantically richer. Activated regions correspond to discriminative parts of the waste.

---

### Cell 15 — Final Report Card

```python
f1 = f1_score(y_true, y_pred)
```

F1 combines precision and recall into one number: `2 * P * R / (P + R)`.  
It's the harmonic mean — if either precision or recall is very low, F1 is also low.  
For binary classification, this is the most informative single metric.

The full report card aggregates all metrics from the training run, the model configuration,  
and the test set evaluation into one printable summary — useful for documenting results  
in a report or paper.

---

### Cell 16 — Save Model

```python
model.save('saved_models/waste_cnn_final.keras')
```

Saves the complete model: architecture + weights + optimizer state.  
`.keras` is the modern TF2 format (replaces `.h5` for full models).

```python
model.export('saved_models/waste_cnn_savedmodel')
```

Exports to TensorFlow SavedModel format — required for TensorFlow Serving,  
FastAPI with TF Serving backend, and TFLite conversion for mobile apps.

```python
model.save_weights('saved_models/waste_cnn_weights.h5')
```

Saves ONLY the weights, not the architecture. To reload, you must rebuild the exact  
same model architecture first, then call `model.load_weights()`.  
Useful when the architecture code is already available separately.

```python
config = {
    'class_names'   : CLASS_NAMES,
    'class_indices' : train_generator.class_indices,
    'img_height'    : IMG_HEIGHT,
    'img_width'     : IMG_WIDTH,
    'threshold'     : 0.5,
    'model_path'    : 'saved_models/waste_cnn_final.keras'
}
with open('saved_models/model_config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

The JSON config stores everything an inference script needs to know without looking  
at the training code. The FastAPI backend loads this file to know what image size  
to resize to, which class is 0 and which is 1, and where the model file is.

---

### Cell 17 — Inference Helper

```python
def predict_single_image(image_path, model, config):
    img = load_img(image_path, target_size=(config['img_height'], config['img_width']))
    img_array  = img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
```

- `load_img` — reads from disk, resizes to 224×224.
- `img_to_array(img) / 255.0` — converts to NumPy float32 array, normalizes to [0, 1].  
  Must match exactly what the training generator did (rescale=1/255).
- `np.expand_dims(..., axis=0)` — adds a batch dimension. Model expects shape (1, 224, 224, 3),  
  but a single image has shape (224, 224, 3). The batch dimension wraps it: `(1, 224, 224, 3)`.

```python
return {
    'label'      : pred_label,
    'confidence' : round(confidence * 100, 2),
    'probabilities': {
        'Organic'    : round(prob_organic    * 100, 2),
        'Recyclable' : round(prob_recyclable * 100, 2)
    }
}
```

Returns a structured dict — the exact format your FastAPI endpoint would return as JSON.  
Rounded to 2 decimal places to avoid floating-point artifacts like 87.39999999999999%.

---

## 3. DCNN Notebook — Cell-by-Cell

---

### DCNN Cell 1 — Imports & Configuration

Most imports are identical to the CNN. Key additions and differences:

```python
from sklearn.preprocessing import label_binarize
```

Used in Cell 12 for one-vs-rest ROC curves. Converts integer class labels [0, 1, ..., 11]  
to a binary matrix of shape (N, 12) where each row has a single 1.

```python
LEARNING_RATE = 5e-4   # 0.0005 — lower than CNN's 0.001
```

**Why lower?**  
The DCNN has 6 blocks and residual connections — it's a deeper, more sensitive architecture.  
A larger learning rate risks overshooting narrow valleys in the loss landscape.  
Starting at 5e-4 and letting `ReduceLROnPlateau` reduce further gives stable convergence.

```python
NUM_CLASSES   = 12
DROPOUT_RATE  = 0.5    # Higher than CNN's 0.4
```

12-class classification is harder to generalize — more classes means more ways to overfit.  
Higher dropout (50%) forces more redundant feature learning and reduces overfitting risk.

```python
CLASS_NAMES = [
    'battery','biological','brown-glass','cardboard','clothes',
    'green-glass','metal','paper','plastic','shoes','trash','white-glass'
]
```

**Alphabetical order is mandatory** — `ImageDataGenerator.flow_from_directory()` assigns  
class indices alphabetically. `battery=0, biological=1, brown-glass=2...`. If this list  
were in a different order, every class label in visualizations would be wrong.

```python
DATASET_DIR = '/kaggle/input/garbage-classification/garbage classification'
```

Note the space in `"garbage classification"` — this is the actual folder name on Kaggle.  
A common error is using an underscore: `"garbage_classification"` → FileNotFoundError.

---

### DCNN Cell 2 — Dataset Verification

Identical logic to CNN Cell 3, adapted for 12 class folders.

```python
bar = '█' * int(n / total * 40)
print(f'{cls:<18} {n:>7,}  {n/total*100:>5.1f}%  {bar}')
```

A horizontal ASCII bar chart printed directly in the cell output. `int(n/total*40)` maps  
the class proportion to a bar length of 0–40 characters. Quick visual of class imbalance  
without needing a matplotlib figure.

```python
ax.axhline(total/NUM_CLASSES, color='#888780', ls='--', lw=1.2, label='Mean per class')
```

Draws a horizontal dashed line at the perfectly balanced count (total images ÷ 12 classes).  
Classes below this line have fewer samples — the model will naturally perform worse on them.  
`trash` and `battery` are typically underrepresented — this is why class weights are needed.

---

### DCNN Cell 3 — Augmentation Pipeline & Generators

Key differences from CNN's augmentation:

```python
vertical_flip = True,          # Enabled in DCNN (disabled in CNN)
channel_shift_range = 15.0,    # New in DCNN
validation_split = 0.20        # 20% vs CNN's 15%
```

**`vertical_flip = True`** — Enabled for DCNN because waste can genuinely appear upside-down  
(a shoe in a bin, a bottle fallen sideways). The 12-class problem benefits from more  
augmentation diversity to avoid overfitting.

**`channel_shift_range = 15.0`** — Randomly shifts each RGB channel value by up to 15.  
Simulates different camera color balances and lighting color temperatures (incandescent  
vs fluorescent vs sunlight). Helps the model generalize across different imaging conditions.

**`validation_split = 0.20`** — 20% (vs CNN's 15%) because this dataset has no separate  
test split. The validation set serves both as training monitor and final evaluation set.

```python
class_mode = 'categorical'    # DCNN: one-hot encoded labels
```

**Critical difference from CNN.**  
- CNN used `class_mode = 'binary'` → labels returned as scalars (0 or 1).  
- DCNN uses `class_mode = 'categorical'` → labels returned as one-hot vectors  
  e.g., `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]` for class index 3 (cardboard).  
- Required by `categorical_crossentropy` loss and the 12-neuron softmax output.

```python
cw_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=train_gen.classes
)
CLASS_WEIGHTS = dict(enumerate(cw_arr))
```

Same concept as CNN but for 12 classes. `trash` and `battery` will get higher weights  
because they have fewer samples. This is printed to verify before training.

---

### DCNN Cell 4 — Class Sample Grid

```python
fig, axes = plt.subplots(len(cls_dirs), n_per, figsize=(n_per*3.2, len(cls_dirs)*3))
```

Creates a 12×3 grid (12 classes, 3 images each). `len(cls_dirs)` rows ensures one row  
per class regardless of how many classes exist. `figsize` scales with the grid size.

```python
if col == 0:
    axes[row][col].set_ylabel(cls, fontsize=10, fontweight='500',
                               color=CLASS_COLORS[row % len(CLASS_COLORS)])
```

Only the first column of each row gets the class label (as y-axis label).  
`row % len(CLASS_COLORS)` wraps around if there are more classes than colors defined.

---

### DCNN Cell 5 — DCNN Architecture

#### The Residual Block Function

```python
def residual_block(x, filters, prefix):
    shortcut = x   # Save input for the skip connection

    # Main path
    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False, name=f'{prefix}_c1')(x)
    x = layers.BatchNormalization(name=f'{prefix}_bn1')(x)
    x = layers.Activation('relu', name=f'{prefix}_r1')(x)
    x = layers.Conv2D(filters, (3,3), padding='same', use_bias=False, name=f'{prefix}_c2')(x)
    x = layers.BatchNormalization(name=f'{prefix}_bn2')(x)

    # Shortcut projection
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), padding='same',
                                  use_bias=False, name=f'{prefix}_proj')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{prefix}_proj_bn')(shortcut)

    x = layers.Add(name=f'{prefix}_add')([x, shortcut])
    x = layers.Activation('relu', name=f'{prefix}_r2')(x)
    return x
```

**The Vanishing Gradient Problem (why residuals are needed):**  
In deep networks (6+ layers), the gradient signal computed by backpropagation must travel  
back through every layer. Each layer's gradients are multiplied by the layer's weights.  
If these weights are slightly less than 1 (common), the gradient shrinks exponentially:  
`0.9^6 = 0.53`, `0.9^20 = 0.12` — by layer 20, the gradient is 88% smaller than at the output.  
Early layers barely learn at all. This is the vanishing gradient problem.

**How residuals solve it:**  
The skip connection creates a "highway" that bypasses the conv layers:  
`output = F(x, {Wi}) + x`  
During backpropagation, the gradient can flow through the addition operation unchanged  
(gradient of addition is 1 × gradient). Even if the conv path's gradient vanishes,  
the skip path gradient remains intact. This lets gradients reach the early layers.

**`shortcut = x`** — saves the input tensor reference before modifying `x`.

**The main path:** Two Conv→BN operations without the final ReLU.  
ReLU comes AFTER the addition so the skip connection can contribute negative values  
that cancel positive main-path values. If ReLU was before the Add, negative contributions  
from the skip path would be zeroed out.

**Projection shortcut:**
```python
if shortcut.shape[-1] != filters:
    shortcut = layers.Conv2D(filters, (1,1), ...)(shortcut)
```

The input `x` might have a different number of channels than the main path output.  
For example, Block 3 receives 128-channel output from Block 2 but processes with 256 filters.  
Adding tensors of different shapes is impossible — the 1×1 conv projection maps the  
shortcut from 128 → 256 channels with minimal computation (no spatial learning, just  
channel mixing).

**`layers.Add()`** — element-wise addition of two tensors of the same shape.  
Not concatenation (which would double the channel count) — true element-wise addition.

#### DCNN Model Architecture

```python
def build_dcnn(input_shape=(224,224,3), num_classes=12, dropout=0.5):
    inputs = keras.Input(shape=input_shape, name='input')
```

**Functional API instead of Sequential.**  
The CNN used `Sequential` because layers form a linear chain.  
The DCNN uses `keras.Model(inputs, outputs)` because residual blocks have branch points  
(the skip connection is a second path) — impossible to express linearly.

**Stem block:**
```python
x = layers.Conv2D(32, (3,3), padding='same', use_bias=False, name='stem_conv')(inputs)
x = layers.BatchNormalization(name='stem_bn')(x)
x = layers.Activation('relu', name='stem_relu')(x)
```

A single conv layer that does initial feature extraction before the main blocks.  
32 filters is intentionally small here — the stem just needs to extract low-level  
edge information that all subsequent blocks will build upon.

**Block 1 and Block 2 (standard, no residual):**  
64 and 128 filters respectively. No residual connection because at shallow depths  
(only 2 blocks deep), vanishing gradients are not yet a problem. Adding residuals  
here would add complexity without benefit.

**Blocks 3 and 4 (with residual connections):**  
This is where the architecture diverges from the CNN. At 4+ blocks deep, residual  
connections become essential for stable gradient flow.

**Blocks 5 and 6 (512 filters, standard):**  
The deepest blocks learn the most abstract, semantically rich features:  
- "This combination of texture and shape indicates corrugated cardboard"  
- "This sheen pattern is characteristic of PET plastic"  
- "This angular geometry with terminal connections indicates a battery"  
  
512 filters provide the capacity to represent these complex patterns.

**GlobalAveragePooling2D + large Dense head:**  
```python
x = layers.GlobalAveragePooling2D(name='gap')(x)
...
x = layers.Dense(1024, ...)(x)  # Bigger than CNN's 512
x = layers.Dense(256, ...)(x)   # Bigger than CNN's 128
outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
```

Larger Dense layers (1024 vs 512) because 12-class discrimination requires more  
representational capacity in the classification head than binary classification.

**`softmax` output (vs CNN's sigmoid):**  
Softmax computes: `e^xi / Σ(e^xj)` for each class i.  
- Produces a probability distribution summing to 1.0.  
- Exactly one class gets the highest probability.  
- Required for mutually exclusive multi-class classification.  
- Binary sigmoid would produce independent probabilities for each class that don't  
  necessarily sum to 1 — incorrect for single-label classification.

---

### DCNN Cell 6 — Compile & Callbacks

```python
dcnn.compile(
    optimizer = Adam(learning_rate=LEARNING_RATE),
    loss      = 'categorical_crossentropy',
    metrics   = [
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
)
```

**`categorical_crossentropy` (vs binary in CNN):**  
For K classes: `L = -Σ yi * log(pi)` where yi is the one-hot label and pi is the  
predicted probability for class i. Only the term for the true class is non-zero.

**Top-3 accuracy:**  
Checks whether the true class is in the model's top 3 predictions.  
For 12-class problems, top-3 accuracy is often a more realistic metric — in a real  
system, the top-3 predictions might be shown to a human for final decision, so  
getting the answer in the top 3 is a meaningful threshold.

**EarlyStopping patience = 8 (vs CNN's 7):**  
Deeper models need more patience. After a ReduceLR event, a deeper model needs  
more epochs to converge to a new local minimum.

---

### DCNN Cell 7 — Training

Identical structure to CNN. Key difference:

```python
history = dcnn.fit(
    train_gen,
    ...
    class_weight = CLASS_WEIGHTS,   # 12-class balanced weights
)
```

The 12-class weights give higher loss penalties for misclassifying `trash` and `battery`  
(underrepresented classes) than for misclassifying `cardboard` or `paper`.

---

### DCNN Cell 8 — Training Curves

```python
plots = [
    ('accuracy',  'val_accuracy',  'Accuracy'),
    ('loss',      'val_loss',      'Loss'),
    ('top3_acc',  'val_top3_acc',  'Top-3 Accuracy'),
]
```

Three plots instead of CNN's two (accuracy + loss). Top-3 accuracy curve shows  
whether the model at least has the right answer in its top-3 even when uncertain.  
For a 12-class problem, top-3 accuracy should be significantly higher than top-1.

---

### DCNN Cell 9 — Test Evaluation

```python
y_pred_prob = dcnn.predict(test_gen, verbose=1)
y_pred      = np.argmax(y_pred_prob, axis=1)
```

- `model.predict()` returns shape (N, 12) — a probability for each of 12 classes.  
- `np.argmax(axis=1)` takes the index of the highest probability for each sample.  
  This is the predicted class label (0–11).  
- For CNN: `(proba > 0.5)` was used because output was shape (N, 1).  
  For DCNN: `argmax` is needed because output is shape (N, 12).

```python
top3 = np.mean([
    y_true[i] in np.argsort(y_pred_prob[i])[-3:]
    for i in range(len(y_true))
])
```

For each sample: sorts all 12 class probabilities ascending, takes the last 3 indices  
(highest probabilities), checks if the true class is among them. Averages over all samples.

---

### DCNN Cell 10 — Confusion Matrix (12×12)

```python
cm     = confusion_matrix(y_true, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
```

The 12×12 matrix has 144 cells. Key things to look for:

**Strong diagonal**: each class is mostly predicted as itself → good per-class accuracy.

**Off-diagonal hotspots**: cells with high values indicate common confusions:
- `brown-glass` confused with `green-glass` (similar shape, different color)  
- `paper` confused with `cardboard` (same material, different thickness)  
- `clothes` confused with `shoes` (similar web-scraped image style)  
- `trash` confused with multiple classes (heterogeneous content)

`keepdims=True` keeps the sum as a column vector (12, 1) for proper broadcasting.

```python
print(f'{cls:<16}  {correct:>8}  {tot:>6}  {correct/tot*100:>6.1f}%  {"█"*int(correct/tot*20)}')
```

Per-class accuracy printed as an ASCII bar chart. `"█" * int(correct/tot*20)` creates  
a bar of up to 20 characters, proportional to per-class accuracy.

---

### DCNN Cell 11 — Classification Report

```python
rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
f1s   = [rep[c]['f1-score']  for c in CLASS_NAMES]
precs = [rep[c]['precision'] for c in CLASS_NAMES]
recs  = [rep[c]['recall']    for c in CLASS_NAMES]
```

`output_dict=True` returns the report as a Python dict instead of a formatted string,  
making it easy to extract per-class metrics for the grouped bar chart.

The grouped bar chart (precision + recall + F1 per class) immediately shows:
- Classes where precision is high but recall is low → model is conservative (over-predicts other classes)  
- Classes where recall is high but precision is low → model over-predicts this class  
- Classes where all three are low → hardest classes (often `trash` and `clothes`)

---

### DCNN Cell 12 — ROC-AUC One-vs-Rest

```python
y_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
```

Converts labels like `[0, 3, 7, ...]` to a binary matrix:
```
class:    0  1  2  3  4  5  6  7  8  9  10  11
sample1:  1  0  0  0  0  0  0  0  0  0   0   0
sample2:  0  0  0  1  0  0  0  0  0  0   0   0
sample3:  0  0  0  0  0  0  0  1  0  0   0   0
```

Required for computing 12 separate ROC curves (one per class).

```python
for i, (cls, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
    auc_val     = roc_auc_score(y_bin[:, i], y_pred_prob[:, i])
```

For each class `i`, treats it as the positive class and all others as negative.  
`y_bin[:, i]` extracts the column for class i (1 if true class=i, 0 otherwise).  
`y_pred_prob[:, i]` extracts the model's predicted probability for class i.

This is called "One-vs-Rest" (OvR) — each class gets its own binary ROC curve.  
The mean AUC across all 12 classes gives the overall discriminability.

---

### DCNN Cell 13 — Sample Predictions

```python
top3_idx = np.argsort(prob)[::-1][:3]
top3_str = '\n'.join([f'{CLASS_NAMES[j]}: {prob[j]*100:.1f}%' for j in top3_idx])
```

`np.argsort(prob)` sorts indices by probability ascending. `[::-1]` reverses to  
descending (highest first). `[:3]` takes the top 3 indices.

Displaying top-3 predictions per image (not just the top-1) gives insight into  
the model's uncertainty — a model that picks `plastic (85%) / metal (8%) / cardboard (4%)`  
is much more confident than `plastic (35%) / green-glass (30%) / brown-glass (25%)`.

---

### DCNN Cell 14 — Grad-CAM Heatmap

```python
def gradcam(model, img_tensor, pred_class, last_conv='b6_conv'):
    grad_model = keras.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv).output, model.output]
    )
```

Creates a new model that outputs both the last conv layer's feature maps AND  
the final prediction. `b6_conv` is the name given to the last convolutional layer  
in the DCNN architecture (Block 6's conv layer).

```python
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, pred_class]
    grads = tape.gradient(loss, conv_out)
```

`GradientTape` records operations for automatic differentiation.  
- `loss = preds[:, pred_class]` — the predicted score for the true/predicted class.  
- `tape.gradient(loss, conv_out)` — computes the gradient of the class score  
  with respect to each element of the last conv layer's output.

**What this gradient means:**  
A high gradient value at position (h, w, c) means "if the activation at this position  
and channel were larger, the predicted class score would increase more" — this position  
is important for the prediction.

```python
    weights = tf.reduce_mean(grads, axis=(0,1,2))
```

Global average pooling of gradients across spatial dimensions (0=batch, 1=height, 2=width).  
Produces one importance weight per channel. Channels with large mean gradients are  
more important for the prediction.

```python
    cam = conv_out[0] @ weights[..., tf.newaxis]
    cam = tf.squeeze(cam)
    cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
```

- `conv_out[0] @ weights[..., tf.newaxis]` — weighted sum of feature maps.  
  For each spatial position (h, w), sums across all channels weighted by their importance.
- `tf.maximum(cam, 0)` — applies ReLU: only keep positive activations  
  (regions that increase the class score, not ones that suppress it).
- Divide by max: normalize heatmap to [0, 1] range.

```python
hm_col = cv2.applyColorMap(np.uint8(255*hm_rs), cv2.COLORMAP_JET)
hm_col = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
overlay = np.uint8(arr*255*0.55 + hm_col*0.45)
```

- `cv2.applyColorMap(_, COLORMAP_JET)` — maps grayscale values to blue→green→red colors.  
  Blue = low activation (unimportant), Red = high activation (important).
- `cvtColor(_, COLOR_BGR2RGB)` — OpenCV uses BGR channel order; matplotlib uses RGB.  
  Must convert to display correctly.
- `0.55` and `0.45` are the blend weights. 55% original image + 45% heatmap.  
  More original image weight ensures the waste item is still recognizable.

---

### DCNN Cell 15 — CNN vs DCNN Comparison

```python
CNN_ACC  = 0.921    # Replace with actual CNN test accuracy
CNN_F1   = 0.919
```

These are placeholder values from research benchmarks. You replace them with your  
actual CNN notebook results. The comparison chart then shows the true improvement  
achieved by the deeper architecture.

The side-by-side table uses matplotlib's `ax.table()` with colored cells:  
- Blue cells for CNN column: `cell.set_facecolor('#EBF3FB')`  
- Purple cells for DCNN column: `cell.set_facecolor('#F0EEFB')`  
- Header row: dark purple background with white bold text.

---

### DCNN Cell 16 — Final Report Card

```python
delta = (acc - CNN_ACC) * 100
print(f'  Improvement over CNN : {delta:+.2f}% accuracy points')
```

The `+` format specifier in `{delta:+.2f}` always shows the sign:  
positive improvement → `+3.41%`, negative → `-1.20%`.  
This quantifies the depth advantage of DCNN over CNN in concrete terms.

---

### DCNN Cell 17 — Save Model

Identical structure to CNN Cell 16. Additional key line:

```python
'class_indices': {v: k for k, v in train_gen.class_indices.items()},
```

`train_gen.class_indices` maps `{'battery': 0, 'biological': 1, ...}`.  
This inverts it to `{0: 'battery', 1: 'biological', ...}` — maps from index back  
to class name, which is what an inference script needs when converting argmax output  
(an integer) to a human-readable class label.

---

### DCNN Cell 18 — Inference Helper

```python
def predict_waste(image_path, model, config, top_k=3):
    ...
    topk = np.argsort(prob)[::-1][:top_k]
    cls  = config['class_names']
    return {
        'predicted_class': cls[int(np.argmax(prob))],
        'confidence'     : round(float(np.max(prob))*100, 2),
        'top_k'          : [{'class': cls[int(i)], 'probability': round(float(prob[i])*100, 2)}
                             for i in topk]
    }
```

**`int(np.argmax(prob))`** — `np.argmax` returns a numpy int64, not a Python int.  
Explicit `int()` conversion ensures JSON serialization works (json.dumps cannot  
serialize numpy types by default).

**`float(np.max(prob))`** — same reason: numpy float32 → Python float for JSON.

**`round(..., 2)`** — prevents outputs like `87.39999999999999%`.  
Always round floats before returning them to a frontend.

---

## 4. Key Concepts Reference

### Why BatchNormalization always comes before Activation

Original paper (Ioffe & Szegedy, 2015) places BN before ReLU:  
`Conv → BN → ReLU`  
BN normalizes the raw conv output (which can have any distribution), then ReLU  
operates on a well-normalized input. BN after ReLU would normalize only non-negative  
values — less effective because half the distribution is already zeroed.

### Why GlobalAveragePooling vs Flatten

| Property | Flatten | GlobalAveragePooling |
|----------|---------|---------------------|
| Parameters (after 256-ch 14×14 map) | 50,176 → Dense | 256 → Dense |
| Overfitting risk | High | Low |
| Spatial information | Preserved (position matters) | Lost (position averaged) |
| Good for | Object detection | Classification |

For classification, position does not matter. A banana in the top-left vs center is  
still organic. GlobalAveragePooling throws away irrelevant spatial information.

### Why Residual Connections in Blocks 3 & 4 (not 1 & 2)

Blocks 1 and 2 are shallow enough that gradients flow without vanishing.  
Adding residuals to shallow layers introduces unnecessary complexity and actually  
slows convergence slightly. The "skip highway" only helps when the road is long.

### Binary Crossentropy vs Categorical Crossentropy

| | Binary CE | Categorical CE |
|--|-----------|----------------|
| Use when | 2 classes, 1 output neuron | K classes, K output neurons |
| Labels | Scalars (0 or 1) | One-hot vectors |
| Output activation | Sigmoid | Softmax |
| Formula | -[y·log(p) + (1-y)·log(1-p)] | -Σ yi·log(pi) |

### The Threshold (0.5) and Why It Matters

The default threshold 0.5 assumes equal cost for false positives and false negatives.  
In real waste management:  
- A battery classified as "recyclable" (false positive for the battery class) could  
  cause a facility fire → very high false negative cost.  
- An organic item classified as "recyclable" → slightly contaminates recyclables  
  → moderate false positive cost.  
  
The ROC curve in Cell 12 allows you to choose a non-0.5 threshold to optimize for  
the actual cost structure of your deployment.

---

## 5. CNN vs DCNN Decision Logic

```
Input image received
        │
        ▼
   ┌─────────────────────────────────────────────────┐
   │             CNN (Phase 1)                       │
   │  4 conv blocks → binary sigmoid → P(recyclable)│
   │  Output: Organic OR Recyclable                  │
   └────────────────────┬────────────────────────────┘
                        │
           ┌────────────┴─────────────┐
           │                          │
      P ≤ 0.5                    P > 0.5
   (Organic)                  (Recyclable)
           │                          │
           ▼                          ▼
   Organic processing          ┌─────────────────────────────────────────────────┐
   (compost/biogas)            │              DCNN (Phase 2)                     │
                               │  6 conv blocks → softmax (12 classes)           │
                               │  Output: battery/biological/glass/cardboard...   │
                               └────────────────────────────────────────────────┘
                                                   │
                          ┌────────┬───────┬───────┴──────┬──────┬──────┬────────┐
                          │        │       │              │      │      │        │
                       battery  paper  cardboard  plastic  metal  glass  trash...
                          │        │       │              │      │      │        │
                          ▼        ▼       ▼              ▼      ▼      ▼        ▼
                       Hazmat   Shred   Flatten       Granulate Smelt  Furnace  Landfill
```

The CNN performs cheap, fast binary triage.  
The DCNN performs expensive, detailed classification only on the recyclable stream.  
Together they form a cost-efficient cascade architecture for real deployment.

---

*Documentation version 1.0 — Covers CNN notebook (17 cells, 784 lines) and DCNN notebook (18 cells, 761 lines)*  
*Total code documented: 1,545 lines across both notebooks*
