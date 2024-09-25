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
