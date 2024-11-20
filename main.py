import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Generate the dataset
num_samples = 10_000_000
X = np.random.randint(-1_000_000, 1_000_000, size=(num_samples, 2))
y = X[:, 0] + X[:, 1]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
model_filename = 'linear_regression_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to '{model_filename}'")

# Later on, or in a different script, load the model
# Load the saved model
model = joblib.load(model_filename)
print(f"Model loaded from '{model_filename}'")

# Test the loaded model
a, b = -12345, 67890
prediction = model.predict([[a, b]])
print(f"The loaded model predicts that {a} + {b} = {prediction[0]:.2f}")
print(f"Actual sum is {a + b}")
