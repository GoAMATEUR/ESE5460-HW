import numpy as np

x = np.load("hw1/p1/x2_train.npy").reshape(-1, 14*14).astype(float) / 255.
y = np.load("hw1/p1/y2_train.npy")
x_val = np.load("hw1/p1/x2_val.npy").reshape(-1, 14*14).astype(float) / 255.
y_val = np.load("hw1/p1/y2_val.npy")

from skimage . filters import gabor_kernel , gabor
import numpy as np
import matplotlib.pyplot as plt


# freqs = [0.1, 1, 5, 10]
# theta = np.pi / 4
# bandwidth = 0.1

# fig, axs = plt.subplots(6, 6, figsize=(10, 10))

# # Hide axes for all subplots
# for ax in axs.flatten():
#     ax.axis('off')



# Define the parameter ranges
thetas = np.arange(0, np.pi, np.pi/4)
frequencies = np.arange(0.05, 0.5, 0.15)
bandwidths = np.arange(0.3, 1, 0.3)

# Create an empty list to store the Gabor kernels
i = 0



x_gabor = np.zeros((x.shape[0], 14*14*36))

print(x_gabor.shape)
for i in range(x.shape[0]):
    if i // 100 == 0:
        print(i)
    image = x[i]. reshape ((14 ,14))
    image_features = []
    for theta in thetas:
        for frequency in frequencies:
            for bandwidth in bandwidths:
                # Create a Gabor kernel with the current parameters
                # gk = gabor_kernel(frequency=frequency, theta=theta, bandwidth=bandwidth)
                # gabor_filters.append((gk, frequency, theta, bandwidth))
                coeff_real , _ = gabor (image , frequency =frequency , theta =theta , bandwidth = bandwidth )
                image_features.append(coeff_real.reshape(-1))
                # axs[i//6, i%6].imshow(coeff_real, cmap='gray')
                i += 1
    image_features_flattened = np.concatenate(image_features)
    x_gabor[i] = image_features_flattened
np.save("hw1/p1/x_gabor.npy", x_gabor)
# plt.savefig("hw1/gabor_filters.png")
    