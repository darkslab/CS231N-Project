from torch import nn


# SEGA Genesis display resolution is 320x224x3

class TestGenesisModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent_dim_pi = 64
        self.latent_dim_vf = 64

        self.policy_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 320, self.latent_dim_pi),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 320, self.latent_dim_vf),
            nn.ReLU(),
        )

    def forward(self, X):
        N, C, H, W = X.shape
        return self.policy_net(X), self.value_net(X)
