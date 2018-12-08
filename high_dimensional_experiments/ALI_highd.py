import torch.nn as nn
import torch.nn.functional as F

class GeneratorX(nn.Module):
    def __init__(self, zd=16, xd=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zd, 16*zd),
            nn.BatchNorm1d(zd*16),
            nn.LeakyReLU(0.02),

            #nn.Dropout(0.2),
            nn.Linear(16*zd, 16*zd),
            nn.BatchNorm1d(zd*16),
            nn.LeakyReLU(0.02),

            #nn.Dropout(0.2),
            nn.Linear(16*zd, 16*zd),
            nn.BatchNorm1d(zd*16),
            nn.LeakyReLU(0.02),

            #nn.Dropout(0.2),
            nn.Linear(16*zd, 16*zd),
            nn.BatchNorm1d(zd*16),
            nn.LeakyReLU(0.02),

            nn.Linear(16*zd, xd),
        )

    def forward(self, x):
        return self.net(x)

class GeneratorZ(nn.Module):
    def __init__(self, zd=16, xd=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xd, zd*8),
            nn.BatchNorm1d(zd*8),
            nn.LeakyReLU(0.02),

            #nn.Dropout(0.2),
            nn.Linear(zd*8, 4*zd),
            nn.BatchNorm1d(zd*4),
            nn.LeakyReLU(0.02),

            #nn.Dropout(0.2),
            nn.Linear(4*zd, zd*4),
            nn.BatchNorm1d(zd*4),
            nn.LeakyReLU(0.02),

            #nn.Linear(4*zd, zd*4),
            nn.Linear(zd*4, zd*2)
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorX(nn.Module):
    def __init__(self, xd = 256, zd = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xd, xd //4),
            nn.BatchNorm1d(xd // 4),
            nn.LeakyReLU(0.02),
            #nn.Dropout(0.2),

            nn.Linear(xd//4, xd//8),
            nn.LeakyReLU(0.02),

            nn.Linear(xd//8, zd),
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorXZ(nn.Module):
    def __init__(self, zd=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zd * 2, zd * 2, bias=False),
            nn.LeakyReLU(0.02),
            #nn.Dropout(0.2),

            nn.Linear(zd * 2, zd, bias=False),
            nn.LeakyReLU(0.02),

            nn.Linear(zd, 1)
        )

    def forward(self, x):
        return self.net(x)
