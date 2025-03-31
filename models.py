import torch.nn as nn
import torch.nn.functional as F
import torch

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 32*32*3 corresponds to the size/shape of img. 
        # W=32 ; H=32 ; 3 Color image ; 
        # output of 10 corresponding to the 10 classes in CIFAR10
        self.fc = nn.Linear(32*32*3, 10)

    def forward(self, x):
        x = torch.flatten(x, 0)
        x = self.fc(x)
        return x
    
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(32*32*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.model(x)


# Trying No-Batch Normalization Version of Convolution Network
class ConvolutionNetwork(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, padding="same", input_size=32):
        super().__init__()

        self.model = nn.Sequential(
            # First Convolutional layer
            nn.Conv2d(in_chan, out_chan*4, kernel_size, stride, padding=padding, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling reduces size by 2

            # Second Convolutional layer
            nn.Conv2d(out_chan*4, out_chan*8, kernel_size, stride, padding=padding, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling reduces size by 2

            # Flatten the output for the fully connected layer
            nn.Flatten(),

            nn.Linear(out_chan * 8 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # Output layer for 10 classes (for example, in classification tasks)
        )

    def forward(self, x):
        return self.model(x)
    
# CNN with Batch Normalization
class ConvolutionBatchNetwork(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, padding="same", input_size=32):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_chan, out_chan*4, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_chan*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_chan*4, out_chan*8, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_chan*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(out_chan * 8 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)