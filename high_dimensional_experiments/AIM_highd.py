import torch.nn as nn
import torch.nn.functional as F

class GeneratorX(nn.Module):
    def __init__(self, zd=32, xd=500):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(zd, 8*zd),
            nn.BatchNorm1d(zd*8),
            nn.LeakyReLU(0.02),

            nn.Linear(8*zd, 8*zd),
            nn.BatchNorm1d(zd*8),
            nn.LeakyReLU(0.02),

            nn.Linear(8*zd, 8*zd),
            nn.BatchNorm1d(zd*8),
            nn.LeakyReLU(0.02),

            nn.Linear(8*zd, 8*zd),
            nn.BatchNorm1d(zd*8),
            nn.LeakyReLU(0.02),

            nn.Linear(8*zd, xd),
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

            nn.Linear(zd*4, 4* zd),
            nn.BatchNorm1d(zd*4),
            nn.LeakyReLU(0.02),

            nn.Linear(4*zd, zd*4),
            nn.BatchNorm1d(zd*4),
            nn.LeakyReLU(0.02),

            nn.Linear(zd*4, zd*2)
        )

    def forward(self, x):
        return self.net(x)

class DiscriminatorX(nn.Module):
    def __init__(self, xd = 500):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xd, xd //4),
            nn.LeakyReLU(0.02),

            # nn.Linear(xd//4, xd//4),
            # nn.LeakyReLU(0.02),

            nn.Linear(xd//4, xd //8),
            nn.LeakyReLU(0.02),

            nn.Linear(xd//8, 1)
        )

    def forward(self, x):
        return self.net(x)



# import torch.nn as nn
# import torch.nn.functional as F
#
# class GeneratorX(nn.Module):
#     def __init__(self, zd=32, ch=1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.ConvTranspose1d(zd, zd, 4, 1),
#             nn.BatchNorm2d(zd),
#             nn.LeakyReLU(0.02),
#
#             nn.ConvTranspose1d(zd, zd//2, 5, 2),
#             nn.BatchNorm2d(zd//2),
#             nn.LeakyReLU(0.02),
#
#             nn.ConvTranspose1d(zd//2, zd//4, 5, 2),
#             nn.BatchNorm2d(zd//4),
#             nn.LeakyReLU(0.02),
#
#             nn.ConvTranspose1d(zd//4, ch, 4, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# class GeneratorZ(nn.Module):
#     def __init__(self, zd=32, ch=1):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(ch, zd//4, 5, 1),
#             nn.BatchNorm2d(zd//4),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv1d(zd//4, zd//2, 5, 2),
#             nn.BatchNorm2d(zd//2),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv1d(zd//2, zd, 3, 2),
#             nn.BatchNorm2d(zd),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv1d(zd, zd*2, 4, 1),
#             nn.BatchNorm2d(zd*2),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv1d(zd*2, zd*2, 1, 1),
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# class DiscriminatorX(nn.Module):
#     def __init__(self, zd=128):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(1, zd//4, 5, 1),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv1d(zd//4, zd//2, 5, 2),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv1d(zd//2, zd, 3, 2),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv1d(zd, zd, 4, 1),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv1d(zd, zd, 1, 1),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv1d(zd, zd, 1, 1),
#             nn.LeakyReLU(0.02),
#
#             nn.Conv1d(zd, 1, 1, 1),
#         )
#
#     def forward(self, x):
#         return self.net(x)
