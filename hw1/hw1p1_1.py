"""
Fetch Dataset
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# Fetch the MNIST dataset
# ds = fetch_openml("mnist_784", as_frame=False)
# ds = fetch_openml("mnist_784", as_frame = False )
# x, x_test , y, y_test = train_test_split (ds.data , ds.target , test_size =0.2, random_state =42)

# # Convert labels to integers
# y = y.astype(int)
# y_test = y_test.astype(int)

# # Initialize lists to store the sampled data and labels
# x_sampled = []
# y_sampled = []

# # Sample 1000 examples per class
# for i in range(10):
#     class_indices = np.where(y == i)[0]  # Get indices for class i
#     sampled_indices = np.random.choice(class_indices, 1000, replace=False)  # Randomly sample 1000 indices
#     x_sampled.append(x[sampled_indices])  # Add the sampled data to the list
#     y_sampled.append(y[sampled_indices])  # Add the corresponding labels to the list

# # Convert lists to arrays
# x_sampled = np.concatenate(x_sampled, axis=0)
# print(x_sampled.max())

# x_sampled_resized = np.zeros((x_sampled.shape[0], 14, 14), dtype=float)
# import cv2
# for i in range(x_sampled.shape[0]):
#     a = x_sampled[i].reshape(28, 28).astype(float)
#     # print(a.shape)
#     x_sampled_resized[i] = cv2.resize(a, (14, 14))
# x_sampled = x_sampled_resized.reshape(-1, 14*14)
# y_sampled = np.concatenate(y_sampled, axis=0)

# # Split the sampled data into training (80%) and validation (20%) sets
# x_train, x_val, y_train, y_val = train_test_split(x_sampled, y_sampled, test_size=0.2, random_state=42)

# print(f"x_train shape: {x_train.shape}")
# print(f"x_val shape: {x_val.shape}")
# print(f"y_train distribution: {np.bincount(y_train)}")
# print(f"y_val distribution: {np.bincount(y_val)}")

# np.save("hw1/p1/x1_train.npy", x_train)
# np.save("hw1/p1/y1_train.npy", y_train)
# np.save("hw1/p1/x1_val.npy", x_val)
# np.save("hw1/p1/y1_val.npy", y_val)
# np.save("hw1/p1/x1_test.npy", x_test)
# np.save("hw1/p1/y1_test.npy", y_test)




# """
# Read constructed dataset downsampled.
# """

import numpy as np
from sklearn.svm import SVC

x_train = np.load("hw1/p1/x1_train.npy").reshape(-1, 14*14).astype(float) / 255.
y_train = np.load("hw1/p1/y1_train.npy")

print(x_train.shape, y_train.shape)
x_val = np.load("hw1/p1/x1_val.npy").reshape(-1, 14*14).astype(float) / 255.    
y_val = np.load("hw1/p1/y1_val.npy")
print(x_val.shape, y_val.shape)

x_test = np.load("hw1/p1/x1_test.npy")
y_test = np.load("hw1/p1/y1_test.npy")

classifier = SVC(C=1.0, kernel="rbf", gamma="auto")
# C = 1.0, gamma = "scale" is default, which is used for 

# Fit the classifier to the training data
classifier.fit(x_train, y_train)

# Predict the labels on the validation set
y_val_pred = classifier.predict(x_val)
# # print(np.unique(y_val_pred))
# # # Calculate validation error
print(y_val_pred, y_val)
val_error = np.mean(y_val_pred != y_val)
print(f"Validation Error: {val_error:.4f}") # Error rate: 0.0785

# # # Calculate the ratio of support vectors to the total number of training samples
support_vectors_ratio = len(classifier.support_) / x_train.shape[0]
print(f"Ratio of Support Vectors to Training Samples: {support_vectors_ratio:.4f}") # 0.5804


# reshape x_test
print(x_test.shape)
x_test_resized = np.zeros((x_test.shape[0], 14, 14), dtype=float)
for i in range(x_test.shape[0]):
    a = x_test[i].reshape(28, 28).astype(float)
    # print(a)
    x_test_resized[i] = cv2.resize(a, (14, 14))
x_test = x_test_resized.reshape(-1, 14*14) / 255.
print(x_test.shape)
y_test_pred = classifier.predict(x_test)
test_error = np.mean(y_test != y_test_pred)
print(f"test Error: {test_error:.4f}") # Error rate: 0.0856

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

print(y_test_pred, y_test)
cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(10))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))

plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
