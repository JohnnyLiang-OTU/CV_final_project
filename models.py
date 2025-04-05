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
    

class DeepCNN(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1):
        super().__init__()
        print("updated2")
        self.conv1 = nn.Conv2d(in_chan, out_chan*36, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # 
        self.bNorm1 = nn.BatchNorm2d(out_chan*36)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 32x32x6 -> 16x16x6

        self.conv2 = nn.Conv2d(out_chan*36, out_chan*60, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # 
        self.bNorm2 = nn.BatchNorm2d(out_chan*60)

        self.conv3 = nn.Conv2d(out_chan*60, out_chan*84, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # 
        self.bNorm3 = nn.BatchNorm2d(out_chan*84)

        self.conv4 = nn.Conv2d(out_chan*84, out_chan*108, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # 
        self.bNorm4 = nn.BatchNorm2d(out_chan*108)

        self.conv5 = nn.Conv2d(out_chan*108, out_chan*132, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # 
        self.bNorm5 = nn.BatchNorm2d(out_chan*132)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8*8*out_chan*132, 512)
        self.out = nn.Linear(512, 10)

    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bNorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bNorm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bNorm3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.bNorm4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv5(x)
        x = self.bNorm5(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.out(x)

        return x

class ModifiedDeepCNN(nn.Module):
    def __init__(self, in_chan, out_chan=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # print("debug msg")
        
        # Same channel dimensions as ImprovedDeepResnet
        self.exp1 = out_chan * 12  # 12x out_chan
        self.exp2 = out_chan * 24  # 24x out_chan
        self.exp3 = out_chan * 36  # 36x out_chan
        self.exp4 = out_chan * 48  # 48x out_chan
        self.exp5 = out_chan * 60  # 60x out_chan

        # Conv blocks (same as ImprovedDeepResnet but without residual shortcuts)
        self.conv1 = nn.Conv2d(in_chan, self.exp1, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.exp1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(self.exp1, self.exp2, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(self.exp2)

        self.conv3 = nn.Conv2d(self.exp2, self.exp3, kernel_size, stride, padding, bias=False)
        self.bn3 = nn.BatchNorm2d(self.exp3)

        self.conv4 = nn.Conv2d(self.exp3, self.exp4, kernel_size, stride, padding, bias=False)
        self.bn4 = nn.BatchNorm2d(self.exp4)

        self.conv5 = nn.Conv2d(self.exp4, self.exp5, kernel_size, stride, padding, bias=False)
        self.bn5 = nn.BatchNorm2d(self.exp5)

        # Same dropout and classifier as ImprovedDeepResnet
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 8 * self.exp5, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  

        # Block 2 (no residual)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Block 3 (no residual)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)  

        # Block 4 (no residual)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # Block 5 (no residual)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)

        return x

class ImprovedDeepResnet(nn.Module):
    def __init__(self, in_chan, out_chan=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        print("Improved DeepResnet with more residual connections")
        
        self.exp1 = out_chan * 12
        self.exp2 = out_chan * 24
        self.exp3 = out_chan * 36
        self.exp4 = out_chan * 48
        self.exp5 = out_chan * 60

        # Initial conv block
        self.conv1 = nn.Conv2d(in_chan, self.exp1, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.exp1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        # Residual blocks with shortcuts
        self.conv2 = nn.Conv2d(self.exp1, self.exp2, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(self.exp2)
        self.res1 = nn.Conv2d(self.exp1, self.exp2, kernel_size=1, stride=1, bias=False)

        self.conv3 = nn.Conv2d(self.exp2, self.exp3, kernel_size, stride, padding, bias=False)
        self.bn3 = nn.BatchNorm2d(self.exp3)
        self.res2 = nn.Conv2d(self.exp2, self.exp3, kernel_size=1, stride=1, bias=False)

        self.conv4 = nn.Conv2d(self.exp3, self.exp4, kernel_size, stride, padding, bias=False)
        self.bn4 = nn.BatchNorm2d(self.exp4)
        self.res3 = nn.Conv2d(self.exp3, self.exp4, kernel_size=1, stride=1, bias=False)

        self.conv5 = nn.Conv2d(self.exp4, self.exp5, kernel_size, stride, padding, bias=False)
        self.bn5 = nn.BatchNorm2d(self.exp5)
        self.res4 = nn.Conv2d(self.exp4, self.exp5, kernel_size=1, stride=1, bias=False)

        # Regularization
        self.dropout = nn.Dropout(0.25)  

        # Classifier
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 8 * self.exp5, 512)
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual block 1
        residual = self.res1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)

        # Residual block 2
        residual = self.res2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual block 3
        residual = self.res3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x += residual
        x = self.relu(x)

        # Residual block 4
        residual = self.res4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x += residual
        x = self.relu(x)

        # Classifier
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)

        return x


# class BlockA(nn.Module):
#     def __init__(self, in_chan, out_chan):
#         super().__init__()

#         self.model = nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         )

#     def forward(self, x):
#         return self.model(x)


# class BlockB(nn.Module):
#     def __init__(self, in_chan, out_chan):
#         super().__init__()

#         self.model = nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         return self.model(x)


# class DeepResnet(nn.Module):
#     def __init__(self, in_chan, out_chan):
#         super().__init__()
#         print("aa")
#         self.downsample1 = nn.Sequential(
#             nn.Conv2d(3, out_chan*60, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_chan*60)
#         )

#         self.downsample2 = nn.Sequential(
#             nn.Conv2d(3, out_chan*108, kernel_size=3, stride=4, padding=1),
#             nn.BatchNorm2d(out_chan*108)
#         )

#         self.relu = nn.ReLU()
#         self.layer1 = self.build_block(in_chan, out_chan*36, isA=False)
#         self.layer2 = self.build_block(in_chan*36, out_chan*60, isA=False)
#         self.layer3 = self.build_block(in_chan*60, out_chan*84, isA=True)
#         self.layer4 = self.build_block(in_chan*84, out_chan*108, isA=True)
#         self.layer5 = self.build_block(in_chan*108, out_chan*132, isA=False)

#         self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(132*out_chan, 512)
#         self.out = nn.Linear(512, 10)

#     def forward(self, x):
#         id1 = self.downsample1(x)
#         id2 = self.downsample2(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = x + id1
#         x = self.relu(x)

#         x = self.layer3(x)
#         x = self.layer4(x)
#         x + x + id2
#         x = self.relu(x)

#         x = self.layer5(x)

#         x = self.global_avg_pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         x = self.out(x)

#         return x

#     def build_block(self, in_chan, out_chan, isA=True):
#         if isA:
#             return BlockA(in_chan, out_chan)
#         return BlockB(in_chan, out_chan)
