#  LPBF Relative Density Prediction using Gradient Boosting

##  Project Overview

This project develops a **Machine Learning model** to predict **Relative Density (%)** in **Laser Powder Bed Fusion (LPBF)** using process parameters:

* Laser Power (W)
* Scan Speed (m/s)

The model uses **Gradient Boosting Regression** with **hyperparameter tuning** and **cross-validation** to ensure reliable and unbiased predictions.

---

##  Workflow Summary

```
Data → Model → GridSearchCV → Best Model → Cross-Validation → Metrics → Prediction → Visualization
```

---

##  Code Explanation

###  Load Dataset

```python
df = pd.read_csv("rp_lpbf_dataset.csv")
X = df[["Laser_Power(W)", "Scan Speed(m/s)"]]
y = df["Relative_Density_%"]
```

* Reads dataset into a DataFrame
* Separates:

  * **X (features)** → Input variables
  * **y (target)** → Output variable

---

###  Define Model

```python
gbr = GradientBoostingRegressor(random_state=42)
```

* Uses Gradient Boosting algorithm
* `random_state` ensures reproducibility

---

###  Define Hyperparameter Grid

```python
param_grid = {
    "learning_rate": [0.01, 0.05, 0.75, 1],
    "max_depth": [2, 3],
    "n_estimators": [50, 75, 100, 150, 175],
    "subsample": [0.8, 1.0]
}
```

* Specifies combinations of parameters to test
* Helps find optimal model configuration

---

###  Cross-Validation Strategy

```python
cv = KFold(n_splits=5, shuffle=True, random_state=42)
```

* Splits dataset into 5 folds
* Ensures model is evaluated on unseen data

---

###  GridSearchCV (Hyperparameter Tuning + CV)

```python
grid = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid.fit(X, y)
```

 What happens:

* Tries all parameter combinations
* Performs cross-validation for each
* Selects best configuration based on **R² score**
* Retrains best model on full dataset (default `refit=True`)

---

###  Best Model Selection

```python
best_model = grid.best_estimator_
```

* Returns model with optimal hyperparameters

---

### Cross-Validated Predictions

```python
y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)
```

* Generates predictions for each sample
* Ensures predictions are made on **unseen folds**
* Prevents data leakage

---

### Performance Metrics

```python
r2 = r2_score(y, y_pred_cv)
mae = mean_absolute_error(y, y_pred_cv)
mse = mean_squared_error(y, y_pred_cv)
rmse = np.sqrt(mse)
```

Metrics used:

* **R² Score** → Model accuracy
* **MAE** → Average error
* **MSE** → Squared error
* **RMSE** → Interpretable error magnitude

---

###  Prediction for New Input

```python
new_input = pd.DataFrame({
    "Laser_Power(W)": [150],
    "Scan Speed(m/s)": [1]
})

prediction = best_model.predict(new_input)
```

* Predicts relative density for new LPBF parameters
* Represents real-world model usage

---

### Visualization (Actual vs Predicted)

```python
plt.scatter(y, y_pred_cv)
```

Plot includes:

* Scatter points (Actual vs Predicted)
* Ideal line (x = y)
* Deviation lines (error visualization)

 Purpose:

* Evaluate model accuracy visually
* Check closeness to ideal prediction

---

##  Key Concepts Used

* Gradient Boosting Regression
* Hyperparameter Tuning (GridSearchCV)
* K-Fold Cross-Validation
* Overfitting Prevention
* Model Evaluation Metrics
* Data Visualization

---

## Important Notes

* Cross-validation ensures **realistic performance**
* Final model is trained using **full dataset**
* CV predictions are used for **honest evaluation**
* Training predictions may appear overly optimistic

---

##  Possible Improvements

* Add more features:

  * Volumetric Energy Density (VED)
  * Hatch spacing
  * Layer thickness

* Try other models:

  * Random Forest
  * XGBoost

---

##  Conclusion

This project demonstrates a robust ML pipeline for LPBF parameter prediction using:

* Reliable model selection
* Proper validation techniques
* Practical prediction capability

---

## 👩‍💻 Author

Srijoni Ghosh
