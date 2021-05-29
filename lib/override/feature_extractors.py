import gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ProjectFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations / 255 * 2 - 1
