# KNN Regressor from Scratch (Python + NumPy)

Ce projet propose une implémentation simple de l'algorithme **K-Nearest Neighbors (KNN)** pour la **régression** et la **classification**, réalisée entièrement en Python avec NumPy, sans bibliothèques de machine learning comme scikit-learn.

## 🔢 Principe

L'algorithme KNN pour la régression prédit une valeur réelle pour un échantillon test en calculant la **moyenne** des valeurs cibles de ses `k` plus proches voisins dans l'ensemble d'entraînement.

Formule :
\[
\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_i
\]

## 📁 Structure du projet

- `knn_regressor.py` : contient la classe `KNNRegressor` avec :
  - `fit()`
  - `predict()`
  - `mean_squared_error()`
  - `mean_absolute_error()`
  - `r2_score()`

- `example.py` : script de test avec un jeu de données synthétique (`make_regression` de scikit-learn).

## ✅ Fonctionnalités

- Métrique de distance : Euclidienne
- Métriques de performance :
  - **Mean Squared Error (MSE)**
  - **Mean Absolute Error (MAE)**
  - **R² Score**

## 🚀 Exemple d'utilisation

```python
from knn_regressor import KNNRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Générer des données de test
X, y = make_regression(n_samples=100, n_features=1, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Créer et entraîner le modèle
model = KNNRegressor(k=5)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
print("MSE:", model.mean_squared_error(y_test, y_pred))
print("MAE:", model.mean_absolute_error(y_test, y_pred))
print("R² Score:", model.r2_score(y_test, y_pred))
