from retro.examples.discretizer import Discretizer
from retro.examples.brute import Frameskip


def project_env_wrapper(env):
    env = MsPacManGenesisDiscretizer(env)
    env = Frameskip(env, skip=4)
    return env


class MsPacManGenesisDiscretizer(Discretizer):
    def __init__(self, env):
        super().__init__(
            env,
            [
                [ "UP" ],
                [ "RIGHT" ],
                [ "DOWN" ],
                [ "LEFT" ],
            ],
        )
