import torch.nn as nn
import torch.nn.functional as F

class GeneratorX(nn.Module):
    def __init__(self, zd=32, xd=500):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zd, 2* zd),
            nn.BatchNorm1d(zd*2),
            nn.LeakyReLU(0.02),

            nn.Linear(2* zd, 4*zd),
            nn.BatchNorm1d(zd*4),
            nn.LeakyReLU(0.02),
            nn.Linear(4*zd, xd),

        )

    def forward(self, x):
        return F.tanh(self.net(x))

class GeneratorZ(nn.Module):
    def __init__(self, zd=32, xd=500):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xd, zd*4),
            nn.BatchNorm1d(zd*4),
            nn.LeakyReLU(0.02),

            nn.Linear(zd*4, 2* zd),
            nn.BatchNorm1d(zd*2),
            nn.LeakyReLU(0.02),

            nn.Linear(2* zd, zd),
            nn.BatchNorm1d(zd),
            nn.LeakyReLU(0.02),



            nn.Linear(zd, zd)
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorX(nn.Module):
    def __init__(self, xd = 500):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xd, xd //4),
            nn.LeakyReLU(0.02),

            nn.Linear(xd//4, xd//4),
            nn.LeakyReLU(0.02),

            nn.Linear(xd//4, xd //8),
            nn.LeakyReLU(0.02),


            nn.Linear(xd//8, 1)
        )

    def forward(self, x):
        return self.net(x)