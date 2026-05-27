from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

# Load dataset
data = fetch_california_housing()

X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = XGBRegressor()

# Train model
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"R2 Score: {r2}")
print(f"MAE: {mae}")

# Save model
joblib.dump(model, "app/model.pkl")

print("Model saved successfully!")