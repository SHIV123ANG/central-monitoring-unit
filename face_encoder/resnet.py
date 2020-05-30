import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s):
        super().__init__()
        if in_channels == out_channels and s == 1:
            self.residual = True
        else:
            self.residual = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2*in_channels, kernel_size=1, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(2*in_channels), 
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, stride=s, padding=1, bias=False), 
            nn.BatchNorm2d(2*in_channels), 
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.conv3(z)
        if self.residual:
            z = x + z
        return z


class ResNet50(nn.Module):
    '''
    Input shape of Network should be Nx3x112x112
    '''
    def __init__(self):
        super().__init__()
        convs = []
        ########## Stage 1 ##########
        convs += [nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )]
        ########## Stage 2 ##########
        convs += [ResidualBlock(64, 64, 2)]
        for _ in range(2):
            convs += [ResidualBlock(64, 64, 1)]
        ########## Stage 3 ##########
        convs += [ResidualBlock(64, 128, 2)]
        for _ in range(3):
            convs += [ResidualBlock(128, 128, 1)]
        ########## Stage 4 ##########
        convs += [ResidualBlock(128, 256, 2)]
        for _ in range(5):
            convs += [ResidualBlock(256, 256, 1)]
        ########## Stage 5 ##########
        convs += [ResidualBlock(256, 512, 2)]
        for _ in range(2):
            convs += [ResidualBlock(512, 512, 1)]
        self.convs = nn.Sequential(*convs)
        self.bn = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc = nn.Sequential(
            nn.Linear(4*4*512, 512, bias=False), 
            nn.BatchNorm1d(512)
        )
    
    def forward(self, x):
        x = self.convs(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ResNet101(nn.Module):
    def __init__(self):
        super().__init__()
        convs = []
        ########## Stage 1 ##########
        convs += [nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False), 
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )]
        ########## Stage 2 ##########
        convs += [ResidualBlock(64, 64, 2)]
        for _ in range(2):
            convs += [ResidualBlock(64, 64, 1)]
        ########## Stage 3 ##########
        convs += [ResidualBlock(64, 128, 2)]
        for _ in range(3):
            convs += [ResidualBlock(128, 128, 1)]
        ########## Stage 4 ##########
        convs += [ResidualBlock(128, 256, 2)]
        for _ in range(22):
            convs += [ResidualBlock(256, 256, 1)]
        ########## Stage 5 ##########
        convs += [ResidualBlock(256, 512, 2)]
        for _ in range(2):
            convs += [ResidualBlock(512, 512, 1)]
        self.convs = nn.Sequential(*convs)
        self.bn = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc = nn.Sequential(
            nn.Linear(4*4*512, 512, bias=False), 
            nn.BatchNorm1d(512)
        )
    
    def forward(self, x):
        x = self.convs(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
