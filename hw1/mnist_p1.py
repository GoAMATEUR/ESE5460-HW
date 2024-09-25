import torchvision as thv
import torch

train = thv.datasets.MNIST ("./hw1/data", download =True , train = True )
val = thv.datasets.MNIST ("./hw1/data", download =True , train = False )

print("Number of training examples:", len(train))
print("Number of validation examples:", len(val))

## Extract half of the data with even classes.
import numpy as np
import matplotlib.pyplot as plt


## 1. construct dataset for training

num_classes = 10
total_samples = 2000
samples_per_class = 200
print(samples_per_class)
# exit()
# Initialize lists to hold the selected data and labels
x_train_balanced = np.zeros((total_samples, 14, 14), dtype=float)
y_balanced = np.zeros((total_samples), dtype=float)

# x_val_balanced = np.zeros()

# Create a figure with 10x10 subplots
fig, axs = plt.subplots(10, 10, figsize=(10, 10))

# Hide axes for all subplots
for ax in axs.flatten():
    ax.axis('off')

import cv2
# Iterate over each class and randomly select samples
for label in range(num_classes):
    print("Label:", label)
    # Find indices of the current class
    class_indices = np.where(val.targets == label)[0]
    print(class_indices)
    
    # Randomly select the required number of samples from this class
    selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
    
    # Append the selected data and labels to the lists
    train_data = val.data[selected_indices].numpy()
    # downsample to (14, 14)
    for i in range(train_data.shape[0]):
        x_train_balanced[label * samples_per_class + i] = cv2.resize(train_data[i], (14, 14))
    y_balanced[label * samples_per_class: (label+1) * samples_per_class] = val.targets[selected_indices].numpy()
    # print(selected_x.shape)
    for i in range(10):
        axs[label, i].imshow(x_train_balanced[label * samples_per_class + i], cmap="gray")
plt.savefig("hw1/p1_eval_visualize.jpg")

np.save("hw1/data/x1_val.npy", x_train_balanced)
np.save("hw1/data/y1_val.npy", y_balanced)




