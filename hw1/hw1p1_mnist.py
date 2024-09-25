"""
Fetch Dataset
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# Fetch the MNIST dataset
ds = fetch_openml("mnist_784", as_frame=False)
x, x_test , y, y_test = train_test_split (ds.data , ds.target , test_size =0.2, random_state =42)

# Convert labels to integers
y = y.astype(int)
y_test = y_test.astype(int)

# Initialize lists to store the sampled data and labels
x_sampled = []
y_sampled = []

# Sample 1000 examples per class
for i in range(10):
    class_indices = np.where(y == i)[0]  # Get indices for class i
    sampled_indices = np.random.choice(class_indices, 200, replace=False)  # Randomly sample 1000 indices
    x_sampled.append(x[sampled_indices])  # Add the sampled data to the list
    y_sampled.append(y[sampled_indices])  # Add the corresponding labels to the list

# Convert lists to arrays
x_sampled = np.concatenate(x_sampled, axis=0)
print(x_sampled.max())

x_sampled_resized = np.zeros((x_sampled.shape[0], 14, 14), dtype=float)
import cv2
for i in range(x_sampled.shape[0]):
    a = x_sampled[i].reshape(28, 28).astype(float)
    # print(a.shape)
    x_sampled_resized[i] = cv2.resize(a, (14, 14))
x_sampled = x_sampled_resized.reshape(-1, 14*14)
y_sampled = np.concatenate(y_sampled, axis=0)

# Split the sampled data into training (80%) and validation (20%) sets
x_train, x_val, y_train, y_val = train_test_split(x_sampled, y_sampled, test_size=0.5)

print(f"x_train shape: {x_train.shape}")
print(f"x_val shape: {x_val.shape}")
print(f"y_train distribution: {np.bincount(y_train)}")
print(f"y_val distribution: {np.bincount(y_val)}")

np.save("hw1/p1/x2_train.npy", x_train)
np.save("hw1/p1/y2_train.npy", y_train)
np.save("hw1/p1/x2_val.npy", x_val)
np.save("hw1/p1/y2_val.npy", y_val)