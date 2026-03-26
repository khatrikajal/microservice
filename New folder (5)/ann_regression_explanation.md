# ANN Regression - Battery Health Prediction: Explanation

This document explains each step in the notebook, why it is done, and how the operation works internally.

---

## 1. Import Libraries

```python
import numpy, pandas, matplotlib, seaborn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers
```

**Why:** Each library has a specific role.

- `numpy` handles fast numerical computation on arrays.
- `pandas` holds and inspects the dataset as a DataFrame.
- `matplotlib` and `seaborn` produce visualizations.
- `sklearn` provides data splitting, scaling, and evaluation metrics.
- `tensorflow / keras` is the deep learning framework used to build, train, and run the ANN.

`warnings.filterwarnings('ignore')` suppresses non-critical warnings so the output stays clean during training.

---

## 2. Create Custom Dataset

```python
np.random.seed(7)
charge_cycles = np.random.randint(10, 1500, n)
...
capacity_remaining_pct = (100 - 0.025 * charge_cycles - ...).clip(20, 100)
```

**Why a custom dataset?** Real battery degradation data is proprietary and large. A synthetic dataset lets us control exactly which features affect the target and by how much, making it ideal for learning.

**How it works:**

- `np.random.seed(7)` fixes the randomness so results are reproducible every run.
- Each feature is sampled from a realistic range using `randint` (integers) or `uniform` (floats).
- `internal_resistance_mohm` is not fully random — it is linked to `charge_cycles` with a small noise term, mimicking real physics where resistance grows as a battery ages.
- The target `capacity_remaining_pct` is computed from a linear combination of all features, each with a weight that reflects real-world battery science (e.g., more cycles = lower capacity, higher temperature = faster degradation).
- `np.random.normal(0, 1.5, n)` adds Gaussian noise to simulate measurement error and real-world unpredictability.
- `.clip(20, 100)` ensures capacity stays within physical bounds (a battery cannot have more than 100% or less than 20% useful capacity in this scenario).

---

## 3. Exploratory Data Analysis (EDA)

**Why:** Before training any model, it is important to understand the data — its shape, distribution, missing values, and relationships between features and the target.

### df.describe()
Provides count, mean, standard deviation, min, max for each column. Helps spot if any feature is on a very different scale (e.g., `charge_cycles` goes up to 1500 while `depth_of_discharge` is between 0.2 and 1.0).

### df.isnull().sum()
Checks for missing values. If any exist, they must be handled before training — models cannot process NaN values.

### Scatter plots (Feature vs Target)
Each feature is plotted against `capacity_remaining_pct`. This shows whether the relationship is linear, non-linear, or has no visible pattern. Features with a clear slope are likely to be predictive.

### Correlation heatmap (seaborn)
The heatmap shows the Pearson correlation coefficient between all pairs of columns. Values close to +1 or -1 indicate strong linear relationships. For example, `charge_cycles` and `internal_resistance_mohm` are positively correlated because we built that into the formula. Features highly correlated with the target are strong candidates for good predictors.

---

## 4. Data Preprocessing

### Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Why:** The model must be evaluated on data it has never seen during training. Splitting 80% for training and 20% for testing gives the model enough data to learn while keeping a fair evaluation set.

`random_state=42` makes the split reproducible.

### StandardScaler

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
```

**Why:** Neural networks are sensitive to the scale of input features. If one feature ranges from 10 to 1500 and another from 0.2 to 1.0, the large-scale feature dominates gradient updates and the model learns poorly.

**How it works:** StandardScaler converts each feature to have mean = 0 and standard deviation = 1 using the formula:

```
z = (x - mean) / std
```

`fit_transform` computes the mean and std from training data and applies the transformation.
`transform` (on test data) applies the same mean/std — it does not refit. This is critical: fitting on test data would cause data leakage.

---

## 5. Build the ANN Model

```python
model = keras.Sequential([
    layers.Input(shape=(7,)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    ...
    layers.Dense(1, activation='linear')
])
```

**Why a Sequential model?** The data flows straight through the layers in order — no branching. Sequential is the right choice for this kind of straightforward regression.

### Dense Layers

A Dense layer connects every neuron in the previous layer to every neuron in this layer. Each connection has a weight. The layer computes:

```
output = activation(W * input + b)
```

Where `W` are the learned weights and `b` is the bias. More neurons = more capacity to learn complex patterns, but also more risk of overfitting.

The architecture grows from 128 → 256 → 128 → 64 → 32 neurons. This funnel shape forces the network to learn compressed representations of the input before making the final prediction.

### ReLU Activation

```
ReLU(x) = max(0, x)
```

Used in all hidden layers. ReLU introduces non-linearity, allowing the network to learn curved, complex relationships between features and the target. Without it, stacking Dense layers would just be equivalent to a single linear transformation — no more powerful than linear regression.

### BatchNormalization

Normalizes the outputs of a layer across the current mini-batch (subtracts batch mean, divides by batch std, then applies learnable scale and shift). This stabilizes training by preventing internal covariate shift — a problem where the distribution of each layer's inputs changes during training. It also acts as a mild regularizer.

### Dropout

```python
layers.Dropout(0.2)
```

During each training step, randomly sets 20% of neurons to zero. This forces the network not to rely too heavily on any single neuron. It is a strong regularization technique that reduces overfitting. Dropout is disabled during prediction (inference).

### Output Layer

```python
layers.Dense(1, activation='linear')
```

A single neuron with linear activation (no transformation). This outputs a real-valued number — exactly what regression needs. If this were classification, we would use `softmax` or `sigmoid` here instead.

### Compile

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
```

- `loss='mse'` (Mean Squared Error) is the function the optimizer minimizes. It penalizes large errors more than small ones due to the square.
- `metrics=['mae']` (Mean Absolute Error) is tracked for readability — it is in the same units as the target (%).
- `Adam` (Adaptive Moment Estimation) is an advanced gradient descent optimizer. It adjusts the learning rate for each weight individually based on historical gradients, converging faster and more reliably than plain SGD.

---

## 6. Train the Model

```python
history = model.fit(
    X_train_sc, y_train,
    validation_split=0.15,
    epochs=300,
    batch_size=32,
    callbacks=[early_stop, lr_scheduler]
)
```

**How training works:**

1. The model makes predictions on a batch of 32 samples.
2. It computes the MSE loss between predictions and true values.
3. Backpropagation computes the gradient of the loss with respect to every weight in the network.
4. Adam updates the weights in the direction that reduces loss.
5. This repeats for every batch — one full pass through all training data is one epoch.

`validation_split=0.15` holds out 15% of training data as a validation set. After every epoch, the model is evaluated on this data without updating weights. This tells us if the model is overfitting (training loss goes down but validation loss goes up).

### EarlyStopping Callback

```python
keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
```

Monitors validation loss each epoch. If it does not improve for 25 consecutive epochs, training stops automatically. `restore_best_weights=True` rolls back to the epoch where validation loss was lowest. This prevents the model from training past the optimal point.

### ReduceLROnPlateau Callback

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
```

If validation loss does not improve for 10 epochs, the learning rate is halved (`factor=0.5`). A smaller learning rate means smaller weight updates — useful when the model is close to a good solution and large updates would cause it to overshoot.

---

## 7. Training History Plots

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
```

**Why:** These plots show whether training was healthy.

- A good training run: both train and val loss decrease smoothly and converge close together.
- Overfitting: train loss keeps going down but val loss flattens or rises.
- Underfitting: both losses are high and plateau early.

Two subplots — one for MSE loss and one for MAE — give complementary views of the same learning process.

---

## 8. Model Evaluation

```python
y_pred = model.predict(X_test_sc).flatten()
```

The model runs forward propagation on the scaled test features and outputs predictions. `.flatten()` converts the 2D output array (shape [240, 1]) into a 1D array for easier comparison with `y_test`.

### Metrics

| Metric | Formula | Meaning |
|---|---|---|
| MSE | mean((y - y_pred)^2) | Penalizes large errors heavily |
| RMSE | sqrt(MSE) | Same unit as target (%) |
| MAE | mean(abs(y - y_pred)) | Average absolute error in % |
| R2 | 1 - SS_res / SS_tot | Proportion of variance explained (1.0 = perfect) |

R2 is the most intuitive — an R2 of 0.95 means the model explains 95% of the variation in battery capacity.

### Actual vs Predicted Plot

Points scattered close to the diagonal red dashed line (`y = x`) mean the model predicts accurately. Points far from the line are errors. A good model will show a tight cluster along the diagonal.

### Residual Plot

`residuals = y_true - y_pred`. Plotted against predicted values. A good model shows residuals randomly scattered around zero with no visible pattern. If residuals fan out or curve, it indicates the model is missing some structure in the data.

### Residual Distribution

A histogram of residuals should be bell-shaped and centered at zero. This confirms errors are random and symmetric — no systematic over- or under-prediction.

---

## 9. Permutation Feature Importance

```python
for i, feat in enumerate(X.columns):
    X_perm = X_test_sc.copy()
    np.random.shuffle(X_perm[:, i])
    perm_pred = model.predict(X_perm)
    importances[feat] = sqrt(MSE(y_test, perm_pred)) - base_rmse
```

**Why:** Neural networks do not natively give feature importance like decision trees. Permutation importance is a model-agnostic method.

**How it works:**

1. Compute the baseline RMSE on the original test set.
2. For each feature, randomly shuffle its values (breaking any relationship with the target).
3. Recompute RMSE with that feature shuffled.
4. The increase in RMSE = how much the model depended on that feature.

A feature that causes a large RMSE increase when shuffled is highly important. A feature with near-zero increase can be removed without hurting the model much.

---

## 10. Prediction on New Data

```python
new_sc   = scaler.transform(new_batteries)
new_pred = model.predict(new_sc)
```

**Why transform first?** The model was trained on scaled data. Passing raw (unscaled) values would give wrong predictions because the model's weights were optimized for standardized inputs. We always apply the same scaler that was fitted on training data.

Three example batteries are passed — a new one, a mid-life one, and a heavily aged one — to show how the predicted capacity degrades with increasing stress conditions.

---

## Overall Flow Summary

```
Raw features (7)
      |
      v
StandardScaler  -->  zero mean, unit variance
      |
      v
ANN Input Layer (7 neurons)
      |
      v
Dense(128) + BatchNorm + Dropout(0.2)   -> learn low-level patterns
      |
Dense(256) + BatchNorm + Dropout(0.3)   -> learn deeper combinations
      |
Dense(128) + BatchNorm + Dropout(0.2)   -> compress representation
      |
Dense(64)  + BatchNorm + Dropout(0.1)   -> refine
      |
Dense(32)                                -> distill
      |
Dense(1, linear)                         -> output: capacity %
      |
      v
MSE Loss  -->  Adam optimizer  -->  backpropagation  -->  weight updates
      |
EarlyStopping + ReduceLROnPlateau (prevent overfitting, fine-tune lr)
      |
      v
Evaluation: RMSE, MAE, R2
      |
Permutation Feature Importance
```

Each layer in the ANN learns progressively more abstract features — the early layers detect simple linear combinations of inputs, while deeper layers compose those into non-linear patterns that better predict battery degradation.
