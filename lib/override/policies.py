import gym
from torch.optim import Adam
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from .feature_extractors import ProjectFeatureExtractor
from ..models import *


class ProjectActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        **kwargs,
    ):
        restricted_kwargs = {
            "net_arch":                   None,
            "activation_fn":              None,
            "ortho_init":                 False,
            "use_sde":                    False,
            "log_std_init":               0.0,
            "full_std":                   False,
            "sde_net_arch":               None,
            "use_expln":                  False,
            "squash_output":              False,
            "features_extractor_class":   ProjectFeatureExtractor,
            "normalize_images":           False,
        }
        default_kwargs = {
            "features_extractor_kwargs":  None,
            "optimizer_class":            Adam,
            "optimizer_kwargs":           {},
        }

        sanitized_kwargs = {}
        for restricted_kwarg, restricted_kwval in restricted_kwargs.items():
            sanitized_kwargs[restricted_kwarg] = kwargs.get(restricted_kwarg, restricted_kwval)
            if sanitized_kwargs[restricted_kwarg] != restricted_kwval:
                raise AssertionError(f"{type(self).__name__}: {restricted_kwarg} has to be {restricted_kwval}")
        for default_kwarg, default_kwval in default_kwargs.items():
            sanitized_kwargs[default_kwarg] = kwargs.get(default_kwarg, default_kwval)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **sanitized_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = TestGenesisModel()
