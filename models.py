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
        x = torch.flatten(x, 1)
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
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, padding="same", input_size=32, bias=True):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_chan, out_chan*4, kernel_size, stride, padding, bias=bias), # Want to try with bias = False | The current save is with bias = True
            nn.BatchNorm2d(out_chan*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(out_chan*4, out_chan*8, kernel_size, stride, padding, bias=bias),
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

class ResNet(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride = 1, padding = "same", ):
        super().__init__()

        self.learning_block = nn.Sequential(
            nn.Conv2d(in_chan, out_chan*4, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_chan*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # At this point, I: 16*16 || C: 12

            nn.Conv2d(out_chan * 4, out_chan*8, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_chan*8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # At this point, I: 8*8 || C: 24
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_chan, out_chan*8, kernel_size=1, stride=4),
            nn.BatchNorm2d(out_chan * 8)
        )

        self.middle_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * out_chan * 8, 128),
        )

        # Resnet Code
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(128, 10)

    def forward(self, x):
        identity = x
        identity = self.downsample(identity)  
        
        x = self.learning_block(x)            
        x = self.relu(x + identity)           
        
        x = self.middle_block(x)              # Flatten + FC to 128
        x = self.output_layer(x)              # FC to 10
        
        return x