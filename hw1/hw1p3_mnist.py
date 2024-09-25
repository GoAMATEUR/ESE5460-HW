import torchvision as thv
import torch

train = thv.datasets.MNIST ("./hw1/data", download =True , train = True )
val = thv.datasets.MNIST ("./hw1/data", download =True , train = False )

print("Number of training examples:", len(train))
print("Number of validation examples:", len(val))

## Extract half of the data with even classes.
import numpy as np
import matplotlib.pyplot as plt

# num_classes = 10
# total_data = len(train.targets)
# total_samples = total_data // 2
# samples_per_class = total_data // num_classes // 2
# print(samples_per_class)

# # Initialize lists to hold the selected data and labels
# x_train_balanced = np.zeros((total_samples, 28, 28), dtype=float)
# y_balanced = np.zeros((total_samples), dtype=float)

# x_val_balanced = np.zeros()

# # Create a figure with 10x10 subplots
# fig, axs = plt.subplots(10, 10, figsize=(10, 10))

# # Hide axes for all subplots
# for ax in axs.flatten():
#     ax.axis('off')

# # Iterate over each class and randomly select samples
# for label in range(num_classes):
#     print("Label:", label)
#     # Find indices of the current class
#     class_indices = np.where(train.targets == label)[0]
#     print(class_indices)
    
#     # Randomly select the required number of samples from this class
#     selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
    
#     # Append the selected data and labels to the lists
#     x_train_balanced[label * samples_per_class: (label+1) * samples_per_class, :, :] = train.data[selected_indices].numpy()
#     y_balanced[label * samples_per_class: (label+1) * samples_per_class] = train.targets[selected_indices].numpy()
#     # print(selected_x.shape)
#     for i in range(10):
#         axs[label, i].imshow(x_train_balanced[label * samples_per_class + i], cmap="gray")
# plt.savefig("hw1/p3_visualize.jpg")

# np.save("hw1/data/x_train.npy", x_train_balanced)
# np.save("hw1/data/y_train.npy", y_balanced)


num_classes = 10
total_data = len(val.targets)
total_samples = total_data // 2
samples_per_class = total_data // num_classes // 2
print(samples_per_class)
# exit()
# Initialize lists to hold the selected data and labels
x_train_balanced = np.zeros((total_samples, 28, 28), dtype=float)
y_balanced = np.zeros((total_samples), dtype=float)

# x_val_balanced = np.zeros()

# Create a figure with 10x10 subplots
fig, axs = plt.subplots(10, 10, figsize=(10, 10))

# Hide axes for all subplots
for ax in axs.flatten():
    ax.axis('off')

# Iterate over each class and randomly select samples
for label in range(num_classes):
    print("Label:", label)
    # Find indices of the current class
    class_indices = np.where(val.targets == label)[0]
    print(class_indices)
    
    # Randomly select the required number of samples from this class
    selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
    
    # Append the selected data and labels to the lists
    x_train_balanced[label * samples_per_class: (label+1) * samples_per_class, :, :] = val.data[selected_indices].numpy()
    y_balanced[label * samples_per_class: (label+1) * samples_per_class] = val.targets[selected_indices].numpy()
    # print(selected_x.shape)
    for i in range(10):
        axs[label, i].imshow(x_train_balanced[label * samples_per_class + i], cmap="gray")
plt.savefig("hw1/p3_eval_visualize.jpg")

np.save("hw1/data/x_val.npy", x_train_balanced)
np.save("hw1/data/y_val.npy", y_balanced)




