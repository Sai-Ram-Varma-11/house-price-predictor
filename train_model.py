# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# 🏠 STEP 1: Load dataset safely (encoding fix)
try:
    df = pd.read_csv("houses.csv", encoding='latin1')
except UnicodeDecodeError:
    df = pd.read_csv("houses.csv", encoding='utf-8-sig')
except FileNotFoundError:
    print("❌ ERROR: Could not find 'houses.csv'. Make sure the file is in the same folder as this Python file.")
    exit()

print("✅ Dataset loaded successfully!\n")
print("Here are the first 5 rows of your data:")
print(df.head())

# 🧩 STEP 2: Select features (X) and target (y)
# ⚠️ Make sure these column names match your CSV exactly!
X = df[["sqft", "bedrooms", "bathrooms"]]  # input features
y = df["price"]                            # target/output

# 🧹 STEP 3: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📊 Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# 🤖 STEP 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n✅ Model training complete!")

# 📈 STEP 5: Predict on the test set
y_pred = model.predict(X_test)

# 🧮 STEP 6: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# 🧠 STEP 7: Display model coefficients
print("\n🔢 Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print("Intercept:", model.intercept_)

# 🎨 STEP 8: Plot actual vs predicted prices
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color="purple", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("price_prediction_plot.png")
print("\n🖼️ Plot saved as 'price_prediction_plot.png'")

# 💾 STEP 9: Save trained model
joblib.dump(model, "house_price_model.joblib")
print("💾 Model saved as 'house_price_model.joblib'")

# 🧮 STEP 10: Example prediction
example = pd.DataFrame([[1400, 3, 2]], columns=["sqft", "bedrooms", "bathrooms"])
prediction = model.predict(example)
print(f"\n🏡 Example Prediction → For a 1400 sqft, 3 bed, 2 bath house: ₹{prediction[0]:,.2f}")
