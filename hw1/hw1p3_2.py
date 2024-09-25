import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.softmax(x)
        return x


if __name__ == "__main__":
    x = np.load("hw1/data/x_train.npy") / 255.0
    y = np.load("hw1/data/y_train.npy")
    x = torch.tensor(x).reshape(-1, 784).float()
    y = torch.tensor(y).long()
    
    
    
    model = Classification().to("mps")
    
    loss = nn.CrossEntropyLoss()
    
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = dataloader.DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(10):
        for i, (x, y) in enumerate(loader):
            x = x.to("mps")
            y = y.to("mps")
            y_pred = model(x)
            loss_val = loss(y_pred, y)
            
            print(loss_val.item())
            # optimize weights
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
    
    
    