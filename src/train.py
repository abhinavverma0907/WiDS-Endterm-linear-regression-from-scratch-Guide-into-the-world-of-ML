import os
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "data", "data.csv")

data = pd.read_csv(data_path)

X = data["SquareFeet"].values.astype(float)
y = data["Price"].values.astype(float)

model = LinearRegression(learning_rate=0.00000001, epochs=1000)
model.fit(X, y)

y_pred = model.predict(X)

plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("House Price Prediction using Linear Regression (From Scratch)")
plt.legend()
plt.savefig(os.path.join(BASE_DIR, "..", "plots", "regression_line.png"))
plt.show()

print("Slope (m):", model.m)
print("Intercept (b):", model.b)
