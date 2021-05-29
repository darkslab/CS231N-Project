import os
import retro
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO


retro.data.Integrations.add_custom_path(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "games",
))

env = make_vec_env(
    retro.make,
    n_envs=1,
    env_kwargs={
        "game": "MsPacMan-Genesis",
        "inttype": retro.data.Integrations.CUSTOM_ONLY
    },
)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10e6)
model.save("trained/baseline/PPO-MsPacMan-Genesis-001")
