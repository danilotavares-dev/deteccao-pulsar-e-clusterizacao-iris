# 🌌 Unsupervised Learning Challenges

> Clustering and anomaly detection using GMM and LOF on real-world datasets.

---

## 📁 Project Structure

```
├── clusterizacao-iris-gmm.ipynb       # Challenge 1 — GMM clustering on Iris
├── deteccao_anomalia_pulsar.ipynb     # Challenge 2 — LOF anomaly detection on Pulsar
└── datasets/
    └── pulsar.csv                     # Pulsar stars dataset
```

---

## Challenge 1 — GMM Clustering on the Iris Dataset

### Objective

Apply **unsupervised learning** to group flower samples from the Iris dataset using the **Gaussian Mixture Model (GMM)** algorithm, with **hyperparameter optimization via Optuna**.

### Dataset

The Iris dataset contains 150 samples of flowers from three species: *Setosa*, *Versicolor*, and *Virginica*. Each sample has four numerical features:

| Feature | Description |
|---|---|
| `sepal length (cm)` | Length of the sepal |
| `sepal width (cm)` | Width of the sepal |
| `petal length (cm)` | Length of the petal |
| `petal width (cm)` | Width of the petal |

> Target labels (0, 1, 2) are used **only for evaluation**, not for training.

### Approach

1. Load the dataset directly from `scikit-learn`
2. Perform EDA (shape, types, basic statistics)
3. Preprocess with `StandardScaler`
4. Optimize hyperparameters using **Optuna GridSampler**:
   - `n_components`: number of Gaussian components (2–10)
   - `covariance_type`: `full`, `tied`, `diag`, `spherical`
5. Select the best model by **minimizing BIC** (Bayesian Information Criterion)
6. Predict cluster assignments and visualize results

### Key Concepts

- **GMM** is a probabilistic clustering method — each point belongs to a cluster with a certain probability (`predict_proba`), unlike K-Means which assigns hard labels
- **BIC** penalizes model complexity; lower BIC = better generalization
- `StandardScaler` is required because GMM is density-based and sensitive to feature scale

---

## Challenge 2 — Anomaly Detection on the Pulsar Dataset

### Objective

Apply **anomaly detection** using the **LOF (Local Outlier Factor)** algorithm to identify pulsar stars in a dataset of astronomical observations, simulating a fully unsupervised scenario.

### Dataset

The Pulsar dataset contains 17,898 observations with 8 numerical features extracted from radio signal time series:

| Feature | Description |
|---|---|
| `Mean of the integrated profile` | Mean of the integrated signal profile |
| `Standard deviation of the integrated profile` | Std of the integrated profile |
| `Excess kurtosis of the integrated profile` | Kurtosis of the integrated profile |
| `Skewness of the integrated profile` | Skewness of the integrated profile |
| `Mean of the DM-SNR curve` | Mean of the DM-SNR curve |
| `Standard deviation of the DM-SNR curve` | Std of the DM-SNR curve |
| `Excess kurtosis of the DM-SNR curve` | Kurtosis of the DM-SNR curve |
| `Skewness of the DM-SNR curve` | Skewness of the DM-SNR curve |
| `target_class` | 1 = pulsar star, 0 = non-pulsar (**evaluation only**) |

**Class distribution:** ~9.2% pulsars (1,639) vs ~90.8% non-pulsars (16,259) — heavily imbalanced.

### Approach

1. Load and explore the dataset (`.info()`, `.describe()`, `value_counts()`)
2. Remove `target_class` from features — unsupervised training only
3. Preprocess with `StandardScaler` via `ColumnTransformer`
4. Train **LOF** with `n_neighbors=30` and `contamination=0.1`
5. Map predictions to match real labels (`-1` = anomaly = pulsar)
6. Evaluate with `recall_score`

### Evaluation

Since the dataset is heavily imbalanced, **recall** is the primary metric:

```
Recall = True Positives / (True Positives + False Negatives)
```

> A model that flags everything as normal achieves 90%+ accuracy but 0% recall for anomalies.

**Important:** Always pass `pos_label=-1` when computing recall for the anomaly class:

```python
recall_score(y_true, y_pred, pos_label=-1)
```

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd deteccao-pulsar-e-clusterizacao-iris

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter
python -m jupyter notebook
```

---

## Dependencies

```
pandas
numpy
scikit-learn
optuna
plotly
jupyter
```

---

## References

- [Scikit-learn: LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- [Scikit-learn: GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [HTRU2 Pulsar Dataset](https://archive.ics.uci.edu/ml/datasets/HTRU2)
