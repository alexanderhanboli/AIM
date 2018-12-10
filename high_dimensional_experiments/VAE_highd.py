import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class GeneratorX(nn.Module):
    def __init__(self, zd=16, xd=256):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(zd, xd, bias=False),

            nn.Linear(zd, 16*zd),
            nn.BatchNorm1d(zd*16),
            nn.LeakyReLU(0.02),

            nn.Linear(16*zd, 16*zd),
            nn.BatchNorm1d(zd*16),
            nn.LeakyReLU(0.02),

            nn.Linear(16*zd, xd),
            nn.BatchNorm1d(xd),
            nn.LeakyReLU(0.02),

            nn.Linear(xd, xd),
            nn.Tanh(), # since the data has been rescaled
        )

    def forward(self, x):
        return self.net(x)

class GeneratorZ(nn.Module):
    def __init__(self, zd=16, xd=256):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(xd, zd*4, bias=False),
            # nn.Linear(zd*4, zd*2),

            nn.Linear(xd, zd*16, bias=False),
            # nn.BatchNorm1d(zd*8),
            # nn.LeakyReLU(0.02),

            #nn.Dropout(0.2),
            nn.Linear(zd*16, 4*zd, bias=False),
            # nn.BatchNorm1d(zd*4),
            # nn.LeakyReLU(0.02),

            nn.Linear(zd*4, zd*2)
        )

    def forward(self, x):
        return self.net(x)
