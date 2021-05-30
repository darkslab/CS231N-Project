from torch import nn


class MsPacManGenesisModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent_dim_pi = 1024
        self.latent_dim_vf = 1024

        self.common_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),     #   3x224x224 ->  64x224x224
            nn.SELU(),                          #  64x224x224 ->  64x224x224
            nn.Conv2d(64, 64, 3, padding=1),    #  64x224x224 ->  64x224x224
            nn.SELU(),                          #  64x224x224 ->  64x224x224

            nn.MaxPool2d(2),                    #  64x224x224 ->  64x112x112

            nn.Conv2d(64, 128, 3, padding=1),   #  64x112x112 -> 128x112x112
            nn.SELU(),                          # 128x112x112 -> 128x112x112
            nn.Conv2d(128, 128, 3, padding=1),  # 128x112x112 -> 128x112x112
            nn.SELU(),                          # 128x112x112 -> 128x112x112

            nn.MaxPool2d(2),                    # 128x112x112 -> 128 x56 x56

            nn.Conv2d(128, 256, 3, padding=1),  # 128 x56 x56 -> 256 x56 x56
            nn.SELU(),                          # 256 x56 x56 -> 256 x56 x56
            nn.Conv2d(256, 256, 3, padding=1),  # 256 x56 x56 -> 256 x56 x56
            nn.SELU(),                          # 256 x56 x56 -> 256 x56 x56

            nn.MaxPool2d(2),                    # 256 x28 x28 -> 256 x28 x28

            nn.Conv2d(256, 512, 3, padding=1),  # 128 x28 x28 -> 512 x28 x28
            nn.SELU(),                          # 512 x28 x28 -> 512 x28 x28
            nn.Conv2d(512, 512, 3, padding=1),  # 512 x28 x28 -> 512 x28 x28
            nn.SELU(),                          # 512 x28 x28 -> 512 x28 x28

            nn.MaxPool2d(2),                    # 512 x28 x28 -> 512 x14 x14

            nn.Conv2d(512, 512, 3, padding=1),  # 512 x14 x14 -> 512 x14 x14
            nn.SELU(),                          # 512 x14 x14 -> 512 x14 x14
            nn.Conv2d(512, 512, 3, padding=1),  # 512 x14 x14 -> 512 x14 x14
            nn.SELU(),                          # 512 x14 x14 -> 512 x14 x14

            nn.MaxPool2d(2),                    # 512 x14 x14 -> 512  x7  x7

            nn.Conv2d(512, 512, 3),             # 512  x7  x7 -> 512  x5  x5
            nn.SELU(),                          # 512  x5  x5 -> 512  x5  x5
            nn.Conv2d(512, 512, 3),             # 512  x5  x5 -> 512  x3  x3
            nn.SELU(),                          # 512  x3  x3 -> 512  x3  x3
            nn.Conv2d(512, 512, 3),             # 512  x3  x3 -> 512  x1  x1
            nn.SELU(),                          # 512  x1  x1 -> 512  x1  x1
        )

        self.policy_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, self.latent_dim_pi),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, self.latent_dim_vf),
            nn.ReLU(),
        )

    def forward(self, X):
        N, C, H, W = X.shape
        X = self.common_net(X)
        return self.policy_net(X), self.value_net(X)
