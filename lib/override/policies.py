import gym
from torch.optim import Adam
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from .feature_extractors import ProjectFeatureExtractor


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
        for kwarg, kwarg_val in restricted_kwargs.items():
            sanitized_kwargs = kwargs.get(kwarg, kwarg_val)
            if sanitized_kwargs[kwarg] != kwarg_val:
                raise AssertionError(f"{type(self).__name__}: {kwarg} has to be {kwarg_val}")
        for kwarg, kwarg_val in default_kwargs.items():
            sanitized_kwargs = kwargs.get(kwarg, kwarg_val)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **sanitized_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        # DARK_NEXT  implement this!
        raise NotImplementedError()
        self.mlp_extractor = None
