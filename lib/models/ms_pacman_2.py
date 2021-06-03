from torch import nn
import torchvision


class MsPacMan2GenesisModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent_dim_pi = 1024
        self.latent_dim_vf = 1024

        self.common_net = torchvision.models.vgg11(pretrained=True)
        for param in self.common_net.parameters():
            param.requires_grad = False

        self.common_net.classifier[6] = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )

        self.policy_net = nn.Sequential(
            nn.Linear(2048, self.latent_dim_pi),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(2048, self.latent_dim_vf),
            nn.ReLU(),
        )

    def forward(self, X):
        N, C, H, W = X.shape
        X = self.common_net(X)
        return self.policy_net(X), self.value_net(X)
