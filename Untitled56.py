#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
df = pd.read_csv("C:/Users/Hp/Downloads/rp_lpbf_dataset.csv")  # change file name

X = df[["Laser_Power(W)", "Scan Speed(m/s)"]]
y = df["Relative_Density_%"]   # or porosity

# -------------------------------------------------
# 2. MODEL
# -------------------------------------------------
gbr = GradientBoostingRegressor(random_state=42)

# small, controlled grid (prevents overfitting)
param_grid = {
    "learning_rate": [0.01, 0.05, 0.75, 1],
    "max_depth": [2, 3],
    "n_estimators": [50, 75, 100, 150, 175],
    "subsample": [0.8, 1.0]
}

# -------------------------------------------------
# 3. CROSS-VALIDATION STRATEGY (IMPORTANT FIX)
# -------------------------------------------------
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# -------------------------------------------------
# 4. GRID SEARCH (WITH CV ONLY)
# -------------------------------------------------
grid = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid.fit(X, y)

print("\nBest Parameters:", grid.best_params_)
print("Best CV R2:", grid.best_score_)

# -------------------------------------------------
# 5. BEST MODEL
# -------------------------------------------------
best_model = grid.best_estimator_

# -------------------------------------------------
# 6. TRUE CROSS-VALIDATED PREDICTIONS (NO LEAKAGE)
# -------------------------------------------------
y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)

# -------------------------------------------------
# 7. METRICS (REALISTIC PERFORMANCE)
# -------------------------------------------------
r2 = r2_score(y, y_pred_cv)
mae = mean_absolute_error(y, y_pred_cv)
mse = mean_squared_error(y, y_pred_cv)
rmse = np.sqrt(mse)

print("\n===== CROSS-VALIDATION PERFORMANCE =====")
print(f"R2   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")

#POROSITY_PREDICTION
new_input = pd.DataFrame({
    "Laser_Power(W)": [150],
    "Scan Speed(m/s)": [1] }) 

prediction = best_model.predict(new_input)

print("Predicted Relative Density:", prediction[0])

# -------------------------------------------------
# 8. ACTUAL vs PREDICTED PLOT (CV-BASED)
# -------------------------------------------------
plt.figure(figsize=(7,6))

plt.scatter(y, y_pred_cv, color='blue', label='CV Predictions')

# ideal line (x = y)
min_val = min(min(y), min(y_pred_cv))
max_val = max(max(y), max(y_pred_cv))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (x = y)')

# deviation lines
for i in range(len(y)):
    plt.plot([y.iloc[i], y.iloc[i]],
             [y.iloc[i], y_pred_cv[i]],
             'gray', alpha=0.3)

plt.xlabel("Actual Relative Density")
plt.ylabel("Predicted Relative Density")
plt.title("Actual vs Predicted (Cross-Validated Gradient Boosting)")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




