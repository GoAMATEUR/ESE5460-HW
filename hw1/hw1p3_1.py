import numpy as np


class LinearLayer:
    def __init__(self, input_dim: int, output_dim: int):
        """
        bias term is included in W (extended by 1 dimension)
        """
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(1, output_dim)
        # self.W = np.zeros((output_dim, input_dim), dtype=float)
        # self.b = np.zeros((1, output_dim), dtype=float)
        self.h_l = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        

    def forward(self, h_l: np.ndarray):
        """
        h_l_1 = h_l @ W.T
        
        Input: h_l (np.array) - shape (batch_size, input_dim)
        Output: h_l_1 (np.array) - shape (batch_size, output_dim)
        """
        h_l_1 = h_l @ self.W.T + self.b
        self.h_l = h_l
        return h_l_1

    def backward(self, dh_l_1: np.ndarray):
        """
        dh_l_1 : shape (batch_size, output_dim)
        """
        self.dW += dh_l_1.T @ self.h_l / dh_l_1.shape[0]
        self.db += np.mean(dh_l_1, axis=0, keepdims=True)
        dh_l = dh_l_1 @ self.W
        return dh_l
        

    def zero_grad(self):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def __call__(self, h_l: np.ndarray):
        return self.forward(h_l)

class ReLU:
    def __init__(self):
        self.h_l = None

    def forward(self, h_l: np.ndarray):
        """
        h_l_1 = max(0, h_l)
        
        Input: h_l (np.array) - shape (batch_size, input_dim)
        Output: h_l_1 (np.array) - shape (batch_size, input_dim)
        """
        self.h_l = h_l
        h_l_1 = np.maximum(0, h_l)
        return h_l_1

    def backward(self, dh_l_1: np.ndarray):
        dh_l = dh_l_1.copy()
        dh_l[self.h_l < 0] = 0
        return dh_l

    def zero_grad(self):
        self.h_l = None


# class Softmax:
#     def __init__(self):
#         self.probs = None

#     def forward(self, h_l: np.ndarray):
#         """
#         h_l_1 = softmax(h_l)
        
#         Input: h_l (np.array) - shape (batch_size, input_dim)
#         Output: h_l_1 (np.array) - shape (batch_size, input_dim)
#         """
#         self.probs = np.exp(h_l) / np.sum(np.exp(h_l), axis=1, keepdims=True)
#         return self.probs

#     def backward(self, dh_l_1: np.ndarray):
#         """_
#         dh_l_1 : shape (batch_size, output_dim)
#         """
#         sum = np.sum(dh_l_1 * self.probs, axis=1, keepdims=True)
#         dh_l = self.probs * (dh_l_1 - sum)
#         return dh_l

class SoftmaxCrossEntropy:
    def __init__(self):
        # no parameters, nothing to initialize
        self.probs = None
        self.y = None

    def forward(self, h_l: np.ndarray, y: np.ndarray):
        """
        Compute the forward pass for softmax cross-entropy loss.
        
        h_l : shape (batch_size, num_classes)
        y : shape (batch_size, num_classes) - one-hot encoded labels
        
        Returns:
        ell : scalar - average loss over the mini-batch
        error : scalar - classification error over the mini-batch
        """
        # Compute softmax probabilities
        # print(h_l)
        probs = np.exp(h_l) / np.sum(np.exp(h_l), axis=1, keepdims=True)
        self.probs = probs
        self.y = y
        # print(probs)
        # Compute cross-entropy loss
        
        
        ell = -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))
        
        # Compute classification error
        predictions = np.argmax(probs, axis=1)
        targets = np.argmax(y, axis=1)
        error = np.mean(predictions != targets)
        
        return ell, error

    def backward(self):
        """
        Compute the backward pass for softmax cross-entropy loss.
        
        Returns:
        dh_l : shape (batch_size, num_classes) - gradient of the loss with respect to h_l
        """
        dh_l = self.probs - self.y
        return dh_l

    def zero_grad(self):
        self.probs = None
        self.y = None


def validation(w, b):
    

if __name__ == "__main__":
    #### Test Layers
    # np.random.seed(0)
    
    # k = 2
    # dh_l_1 = np.zeros((1, 3), dtype=float)
    # dh_l_1[0, k] = 1
    
    # h_1 = np.array([[1, 2, 3, 4]], dtype=float)
    
    # import random
    # i = k
    # j = 3
    # epsilon = np.random.randn(3, 4) * 1e-5
    # epsilon_mtx = np.zeros((3, 4), dtype=float)
    # epsilon_mtx[i, j] = epsilon[i, j]
    
    
    # linear = LinearLayer(4, 3)
    # h_l_1 = linear(h_1)
    # linear.zero_grad()
    # linear.backward(dh_l_1)
    
    # est = ((h_1 @ ((linear.W + epsilon_mtx).T))[0, k] - (h_1 @ (linear.W - epsilon_mtx).T)[0, k]) / epsilon_mtx[i, j] / 2
    
    
    # print("Implemented dW_ij:", linear.dW)
    # print("Estimated dW_ij:", est)
    
    ## Training
    ## 1. load data
    x_train = np.load("hw1/data/x_train.npy").astype(float) # shape (num_samples, 28, 28)
    y_train = np.load("hw1/data/y_train.npy").astype(int) # shape (num_samples, )
    print(x_train.shape, y_train.shape)
    
    ## 2. One-hot encoding
    y_train_one_hot = np.zeros((len(y_train), 10), dtype=float)
    y_train_one_hot[np.arange(len(y_train)), y_train] = 1.0
    # x_val = np.load("hw1/data/x_val.npy")
    # y_val = np.load("hw1/data/y_val.npy")
    batch_size = 32
    
    
    l1 , l2 , l3 = LinearLayer(28*28, 10) , ReLU() , SoftmaxCrossEntropy()
    net = [l1 , l2 , l3]
    losses = []
    errors = []
    # curriculum = 
    # lr = 0.1
    
    for epoch in range(30000):
        lr = 0.1
        # Extract mini-batch
        # randomly choose indices
        batch_indices = np.random.choice(len(x_train), batch_size, replace=False)
        x = x_train[batch_indices].reshape(batch_size, -1) / 255.0
        
        y_gt = y_train_one_hot[batch_indices]
        # 2. zero gradient buffer
        for l in net:
            l.zero_grad()
        # 3. forward pass
        h1 = l1.forward (x)
        h2 = l2.forward (h1)
        ell , error = l3.forward(h2, y_gt)
        # 4. backward pass
        dh2 = l3.backward()
        dh1 = l2.backward(dh2)
        dx = l1.backward(dh1)
        # 5. gather backprop gradients
        dw , db = l1.dW , l1.db
        # 6. print some quantities
        print(epoch, ell , error)
        # print(epoch, np.linalg.norm(dw/l1.W), np.linalg.norm(db/l1.b))
        l1.W -= lr * dw
        l1.b -= lr * db
        losses.append(ell)
        errors.append(error)
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.savefig("hw1/p3_loss.jpg")
    plt.clf()
    plt.plot(errors)
    plt.savefig("hw1/p3_error.jpg")
    
    print("err on train set:")
    x = x_train.reshape(len(x_train), -1) / 255.0
    y_gt = y_train_one_hot
    for l in net:
        l.zero_grad()
    # 3. forward pass
    h1 = l1.forward (x)
    h2 = l2.forward (h1)
    ell , error = l3.forward(h2, y_gt)
    # 4. backward pass
    dh2 = l3.backward()
    dh1 = l2.backward(dh2)
    dx = l1.backward(dh1)
    # 5. gather backprop gradients
    dw , db = l1.dW , l1.db
    # 6. print some quantities
    print(epoch, ell , error)
            
        
    
    