import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

class Dilated_Net(nn.Module):
    def __init__(self):
        super(Dilated_Net, self).__init__()
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=1, padding=1, bias=False),  #32>>28 | 1>>7 | 1>>1
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1),
            nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),#28>>26 | 7>>11 | 1>>1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=2, bias=False),#26>>24 | 11>>15 | 1>>1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
        )
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False), #24>>24 | 15>>17 | 1>>1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, 1, stride=1, padding=1, bias=False), #26>>26 | 17>>17 | 1>>1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #nn.Dropout(0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1,dilation=4, bias=False), #20>>20 | 17>>25 | 1>>1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),

        )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False), #20>>20 | 20>>20 | 1>>1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.Conv2d(64, 32, 1, stride=1, padding=1, bias=False), #20>>22 | 20>>22 | 1>>1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout(0.1),
            nn.Conv2d(32, 64, 3, stride=1, padding=1,dilation=8, bias=False), #22>>8 | 22>>43 | 1>>1
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),


        )
        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, 3, stride=1, padding=1, bias=False), #8>>8 | 43>>45 | 8>>8
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.Dropout(0.1),
            nn.Conv2d(96, 128, 1, stride=1, padding=1, bias=False), #8>>10 | 45>>45 | 8>>8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            #nn.Dropout(0.1),

        )

        self.fc = nn.Sequential(
            nn.Conv2d(128, 10, 1, stride=1, padding=0, bias=False), 
            #nn.ReLU(),
            #nn.BatchNorm2d(1),
            #nn.Dropout(0.05),
            #nn.Linear(10, 10)

        )
        self.gap = nn.Sequential(
            #nn.AvgPool2d(kernel_size=16)
            nn.AdaptiveAvgPool2d((1, 1))
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)