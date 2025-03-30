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


class ArchitectureXYZ(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass