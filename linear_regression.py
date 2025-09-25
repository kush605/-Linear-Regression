import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (replace with dataset path after download)
data = pd.read_csv("Housing.csv")

# Display basic info
print("Dataset shape:", data.shape)
print(data.head())

# Select features and target
X = data[["area", "bedrooms", "bathrooms"]]   # Example features
y = data["price"]                             # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Coefficients interpretation
coefficients = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print("\nModel Coefficients:")
print(coefficients)

# Plot regression (only if single feature for visualization)
plt.scatter(X_test["area"], y_test, color="blue", label="Actual")
plt.scatter(X_test["area"], y_pred, color="red", label="Predicted")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("Linear Regression - House Price Prediction")
plt.legend()
plt.show()