import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class GeneratorX(nn.Module):
    def __init__(self, zd=16, xd=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zd, xd, bias=False),

            # nn.Linear(zd, 4*zd),
            # nn.BatchNorm1d(zd*4),
            # nn.LeakyReLU(0.02),
            #
            # nn.Linear(4*zd, 8*zd),
            # nn.BatchNorm1d(zd*8),
            # nn.LeakyReLU(0.02),
            #
            # nn.Linear(8*zd, 16*zd),
            # nn.BatchNorm1d(zd*16),
            # nn.LeakyReLU(0.02),
            #
            # #nn.Dropout(0.2),
            # nn.Linear(16*zd, 16*zd),
            # nn.BatchNorm1d(zd*16),
            # nn.LeakyReLU(0.02),
            #
            # #nn.Dropout(0.2),
            # nn.Linear(16*zd, 16*zd),
            # nn.BatchNorm1d(zd*16),
            # nn.LeakyReLU(0.02),
            #
            # #nn.Dropout(0.2),
            # nn.Linear(16*zd, 16*zd),
            # nn.BatchNorm1d(zd*16),
            # nn.LeakyReLU(0.02),
            #
            # nn.Linear(16*zd, xd),
        )

    def forward(self, x):
        return self.net(x)

class Feature(nn.Module):
    def __init__(self, zd=16, xd=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xd, zd*4, bias=False),

            #nn.BatchNorm1d(zd*8),
            #nn.LeakyReLU(0.02),

            #nn.Dropout(0.2),
            #nn.Linear(zd*8, 4*zd),
        )

    def forward(self, x):
        return self.net(x)

# class GeneratorZ(nn.Module):
#     def __init__(self, zd=16, xd=256):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(xd, zd*8),
#             nn.BatchNorm1d(zd*8),
#             nn.LeakyReLU(0.02),
#
#             #nn.Dropout(0.2),
#             nn.Linear(zd*8, 4*zd),
#             nn.BatchNorm1d(zd*4),
#             nn.LeakyReLU(0.02),
#
#             #nn.Dropout(0.2),
#             nn.Linear(4*zd, zd*4),
#             nn.BatchNorm1d(zd*4),
#             nn.LeakyReLU(0.02),
#
#             #nn.Linear(4*zd, zd*4),
#             nn.Linear(zd*4, zd*2)
#         )
#
#     def forward(self, x):
#         return self.net(x)

class GeneratorZ(nn.Module):
    def __init__(self, zd=16, xd=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4*zd, zd*4, bias=False),
            #nn.BatchNorm1d(zd*4),
            #nn.LeakyReLU(0.02),

            #nn.Linear(4*zd, zd*4),
            nn.Linear(zd*4, zd*2)
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorX(nn.Module):
    def __init__(self, zd = 16, xd = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zd*4, xd//8, bias=False),
            nn.LeakyReLU(0.02),

            nn.Linear(xd//8, xd //16, bias=False),
            nn.LeakyReLU(0.02),
            # nn.Dropout(0.1),

            nn.Linear(xd//16, 1)
        )

    def forward(self, x):
        return self.net(x)

# class GeneratorX(nn.Module):
#     def __init__(self, zd=128, ch=1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.ConvTranspose2d(zd, zd, 4, 1),
#             nn.BatchNorm2d(zd),
#             nn.LeakyReLU(0.02),
#
#             nn.ConvTranspose2d(zd, zd//2, 5, 2),
#             nn.BatchNorm2d(zd//2),
#             nn.LeakyReLU(0.02),
#
#             nn.ConvTranspose2d(zd//2, zd//4, 5, 2),
#             nn.BatchNorm2d(zd//4),
#             nn.LeakyReLU(0.02),
#
#             nn.ConvTranspose2d(zd//4, ch, 4, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# class GeneratorZ(nn.Module):
#     def __init__(self, zd=128, ch=1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(ch, zd//4, 5, 1),
#             nn.BatchNorm2d(zd//4),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv2d(zd//4, zd//2, 5, 2),
#             nn.BatchNorm2d(zd//2),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv2d(zd//2, zd, 3, 2),
#             nn.BatchNorm2d(zd),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv2d(zd, zd*2, 4, 1),
#             nn.BatchNorm2d(zd*2),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv2d(zd*2, zd*2, 1, 1),
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# class DiscriminatorX(nn.Module):
#     def __init__(self, zd=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(1, zd//4, 5, 1),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv2d(zd//4, zd//2, 5, 2),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv2d(zd//2, zd, 3, 2),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv2d(zd, zd, 4, 1),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv2d(zd, zd, 1, 1),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv2d(zd, zd, 1, 1),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv2d(zd, 1, 1, 1),
#         )
#
#     def forward(self, x):
#         return self.net(x)
