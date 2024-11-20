import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import argparse
import time

# Start timer


# Set up argument parser
parser = argparse.ArgumentParser(description='Load a model from a file.')
parser.add_argument('model_filename', type=str, help='The filename of the model to load')

# Parse arguments
args = parser.parse_args()

model = joblib.load(args.model_filename)
print(f"Model loaded from '{args.model_filename}'")
start_time = time.time()
a, b = 0,0 
prediction = model.predict([[a, b]])
print(f"The loaded model predicts that {a} + {b} = {prediction[0]:.2f}")
# print(f"Actual sum is {a + b}")

# End timer
end_time = time.time()
print(f"Execution time: {end_time - start_time:.10f} seconds")