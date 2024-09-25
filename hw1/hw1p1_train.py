

import numpy as np
from sklearn.svm import SVC

x_train = np.load("hw1/p1/x_gabor.npy")
y_train = np.load("hw1/p1/y2_train.npy")

x_val = np.load("hw1/p1/x_gabor.npy")
y_val = np.load("hw1/p1/y2_val.npy")

classifier = SVC(C=1.0, kernel="rbf", gamma="auto")
# C = 1.0, gamma = "scale" is default, which is used for 

# Fit the classifier to the training data
print(np.max(x_train))
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