from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
data = fetch_california_housing()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["PRICE"] = data.target

# Show first rows
print(df.head())
print(df.info())

# -----------------------------
# 3. Prepare the data
# -----------------------------

# Select features for X
X = df[['MedInc', 'HouseAge', 'AveRooms']]

# Target variable
y = df['PRICE']

from sklearn.model_selection import train_test_split

# Split dataset: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# -----------------------------
# -----------------------------
# 7. Normalize features
# -----------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same transformation on test data
X_test_scaled = scaler.transform(X_test)

print("Scaled shapes:", X_train_scaled.shape, X_test_scaled.shape)

# 4. Train Linear Regression Model
# -----------------------------
from sklearn.linear_model import LinearRegression

# Train again with scaled data
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model trained with scaled data!")

# -----------------------------

# -----------------------------
# 5. Make predictions (use scaled test data)
# -----------------------------
y_pred = model.predict(X_test_scaled)   # <-- حتما از X_test_scaled استفاده کن
print("First 10 predictions (scaled model):", y_pred[:10])
print("First 10 real prices:", list(y_test[:10]))

# -----------------------------
# 6. Evaluate the model (scaled)
# -----------------------------
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

mse2 = mean_squared_error(y_test, y_pred)
rmse2 = np.sqrt(mse2)
r2_2 = r2_score(y_test, y_pred)

print("After Scaling -> RMSE:", rmse2)
print("After Scaling -> R2  :", r2_2)

# Ridge (already trained on scaled data)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Ridge RMSE:", rmse_ridge)
print("Ridge R2:", r2_ridge)

# -----------------------------
# 6. Polynomial Regression
# -----------------------------
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train_scaled)
X_poly_test = poly.transform(X_test_scaled)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

y_pred_poly = poly_model.predict(X_poly_test)

mse_poly = mean_squared_error(y_test, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("Polynomial RMSE:", rmse_poly)
print("Polynomial R2:", r2_poly)

# ----------------------------
#  EVALUATE, SAVE BEST MODEL, PREDICT SAMPLE (fixed names & handling)
# ----------------------------
import matplotlib.pyplot as plt
import joblib
import os

def report_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:12s} -> RMSE: {rmse:.6f}   R2: {r2:.6f}")
    return rmse, r2

# use the actual model variable names you have
models = {
    'Linear': model,
    'Ridge': ridge,
    'Poly'  : poly_model
}

results = {}
for name, mdl in models.items():
    if name == 'Poly':
        X_for_pred = X_poly_test
    else:
        X_for_pred = X_test_scaled
    y_pred_model = mdl.predict(X_for_pred)
    results[name] = report_metrics(name, y_test, y_pred_model)

# choose best by RMSE
best_name = min(results.keys(), key=lambda k: results[k][0])
best_rmse, best_r2 = results[best_name]
best_model = models[best_name]
print("\nBest model based on RMSE:", best_name, f"(RMSE={best_rmse:.6f}, R2={best_r2:.6f})")

# save model and scaler (and polynomial transformer if used)
os.makedirs("saved_models", exist_ok=True)
model_path = f"saved_models/best_model_{best_name}.joblib"
joblib.dump(best_model, model_path)
print("Saved best model to:", model_path)
scaler_path = "saved_models/scaler.joblib"
joblib.dump(scaler, scaler_path)
print("Saved scaler to:", scaler_path)
# save the poly transformer if poly was used
poly_path = "saved_models/poly_transformer.joblib"
joblib.dump(poly, poly_path)
print("Saved poly transformer to:", poly_path)

# helper to load and predict a single sample dict
def load_model_and_predict(sample_dict):
    mdl = joblib.load(model_path)
    sc = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    poly_t = joblib.load(poly_path) if os.path.exists(poly_path) else None
    feature_order = list(X.columns)
    x_list = [sample_dict[k] for k in feature_order]
    x_arr = np.array(x_list).reshape(1, -1)
    if sc is not None:
        x_arr = sc.transform(x_arr)
    if best_name == 'Poly' and poly_t is not None:
        x_arr = poly_t.transform(x_arr)
    pred = mdl.predict(x_arr)
    return float(pred[0])

# example prediction using first test row
example = {col: float(X_test.iloc[0][col]) for col in X.columns}
print("\nExample test-row (used as sample):", example)
print("Predicted (example):", load_model_and_predict(example))

# residual plot for best model
if best_name == 'Poly':
    y_pred_best = best_model.predict(X_poly_test)
else:
    y_pred_best = best_model.predict(X_test_scaled)

res = y_test - y_pred_best
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(y_pred_best, res, alpha=0.4)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title(f'Residuals vs Predicted ({best_name})')
plt.subplot(1,2,2)
plt.hist(res, bins=30)
plt.title('Residuals distribution')
plt.tight_layout()
plt.show()
