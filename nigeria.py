import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from scipy.stats import skew
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'nga_median_spend_model_train_data.xlsx'
data = pd.read_excel(file_path)

# Visualize and check skewness of median_spend
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['median_spend'], kde=True, bins=30)
plt.title('Histogram of Median Spend')

plt.subplot(1, 2, 2)
sns.boxplot(x=data['median_spend'])
plt.title('Boxplot of Median Spend')

plt.tight_layout()
plt.show()

# Apply log transformation since skewed
data['log_median_spend'] = np.log1p(data['median_spend'])

# Step 1: Check for multicollinearity using VIF
X = data.drop(columns=['median_spend', 'log_median_spend'])
y = data['log_median_spend']

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("Variance Inflation Factors:")
print(vif_data)

# We have some highly collinear features (VIF > 5), makes sense since it is spatial data

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data for Ridge and Lasso
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression (L2 Regularization)
ridge = Ridge()
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_search = GridSearchCV(ridge, ridge_params, scoring='neg_root_mean_squared_error', cv=5)
ridge_search.fit(X_train_scaled, y_train)
ridge_best = ridge_search.best_estimator_

# Lasso Regression (L1 Regularization)
lasso = Lasso(max_iter=10000)
lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_search = GridSearchCV(lasso, lasso_params, scoring='neg_root_mean_squared_error', cv=5)
lasso_search.fit(X_train_scaled, y_train)
lasso_best = lasso_search.best_estimator_

# Evaluate Ridge and Lasso on test set
ridge_pred_log = ridge_best.predict(X_test_scaled)
lasso_pred_log = lasso_best.predict(X_test_scaled)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred_log))
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred_log))

print(f"Ridge RMSE (Log Scale): {ridge_rmse}")
print(f"Lasso RMSE (Log Scale): {lasso_rmse}")

# Back-transform predictions
ridge_pred_original = np.expm1(ridge_pred_log)
lasso_pred_original = np.expm1(lasso_pred_log)

# Evaluate on original scale
ridge_rmse_original = np.sqrt(mean_squared_error(np.expm1(y_test), ridge_pred_original))
lasso_rmse_original = np.sqrt(mean_squared_error(np.expm1(y_test), lasso_pred_original))

print(f"Ridge RMSE (Original Scale): {ridge_rmse_original}")
print(f"Lasso RMSE (Original Scale): {lasso_rmse_original}")

# XGBoost Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 10]
}

xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(estimator=xgb_base, param_grid=param_grid, cv=5, 
                           scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)

# Best parameters from grid search
best_model = grid_search.best_estimator_
y_pred_log = best_model.predict(X_test)

# Back-transform predictions to original scale
y_pred_original = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# Calculate RMSE in original scale
rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
print(f"Optimized XGBoost RMSE (Original Scale): {rmse_original}")

# Calculate the RMSE of a baseline model (predicting the mean of the training set)
baseline_pred = np.full_like(y_test_original, y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_test_original, baseline_pred))
print(f"Baseline RMSE (Original Scale): {baseline_rmse}")

# Feature importance visualization
xgb.plot_importance(best_model)
plt.title('Feature Importance')
plt.show()
