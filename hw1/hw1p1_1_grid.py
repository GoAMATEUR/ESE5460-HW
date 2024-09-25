"""
Fetch Dataset
"""
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

# """
# Read constructed dataset downsampled.
# """

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

x_train = np.load("hw1/p1/x1_train.npy").reshape(-1, 14*14).astype(float) / 255.
y_train = np.load("hw1/p1/y1_train.npy")

print(x_train.shape, y_train.shape)
x_val = np.load("hw1/p1/x1_val.npy").reshape(-1, 14*14).astype(float) / 255.    
y_val = np.load("hw1/p1/y1_val.npy")
print(x_val.shape, y_val.shape)


# Define the parameter grid for C
param_grid = {'C': [0.1, 1, 10, 100, 1000]} 

svc = SVC(kernel='rbf', gamma='auto')

grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy')

grid_search.fit(x_train, y_train)

print("Grid Search Results:")
for param, mean_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
    print(f"Parameter: {param}, Cross-Validation Accuracy: {mean_score:.4f}")

# Get the best parameter combination
best_params = grid_search.best_params_
print(f"\nBest Parameters: {best_params}")

# Evaluate the best model on the validation set



# best_model = grid_search.best_estimator_
# y_val_pred = best_model.predict(x_val)
# val_error = np.mean(y_val != y_val_pred)
# print(f"Validation Set Accuracy with Best C: {val_error:.4f}")

print("Validation Set Performance of Each Estimator:")
for idx, estimator_params in enumerate(grid_search.cv_results_['params']):  
    # Access the corresponding estimator
    estimator = grid_search.cv_results_['params'][idx]
    # Manually set the parameters to the SVM
    estimator_svc = SVC(kernel='rbf', gamma='auto', C=estimator['C'])
    # Train on the entire training set
    estimator_svc.fit(x_train, y_train)
    # Predict on the validation set
    y_val_pred = estimator_svc.predict(x_val)
    # Compute accuracy
    err = np.mean(y_val != y_val_pred)
    
    print(f"Estimator {idx+1}: Params: {estimator_params}, Validation Accuracy: {err:.4f}")

"""
Result:

Grid Search Results:
Parameter: {'C': 0.1}, Cross-Validation Accuracy: 0.8183
Parameter: {'C': 1}, Cross-Validation Accuracy: 0.9045
Parameter: {'C': 10}, Cross-Validation Accuracy: 0.9319
Parameter: {'C': 100}, Cross-Validation Accuracy: 0.9427
Parameter: {'C': 1000}, Cross-Validation Accuracy: 0.9406

Best Parameters: {'C': 100}
Validation Set Performance of Each Estimator:
Estimator 1: Params: {'C': 0.1}, Validation Accuracy: 0.1505
Estimator 2: Params: {'C': 1}, Validation Accuracy: 0.0785
Estimator 3: Params: {'C': 10}, Validation Accuracy: 0.0590
Estimator 4: Params: {'C': 100}, Validation Accuracy: 0.0520
Estimator 5: Params: {'C': 1000}, Validation Accuracy: 0.0525

"""


