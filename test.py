import os
import retro
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from lib.override.policies import ProjectActorCriticPolicy
from lib.override.env_wrappers import project_env_wrapper
from lib.models.ms_pacman import MsPacManGenesisModel


retro.data.Integrations.add_custom_path(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "games",
))

env = make_vec_env(
    retro.make,
    n_envs=1,
    wrapper_class=project_env_wrapper,
    env_kwargs={
        "game": "MsPacMan-Genesis",
        "inttype": retro.data.Integrations.CUSTOM_ONLY
    },
)

model = PPO(
    ProjectActorCriticPolicy,
    env,
    # learning_rate=3e-4,
    # n_steps=2048,
    # batch_size=128,
    # n_epochs=10,
    # gamma=0.99,
    # tensorboard_log=None,
    # create_eval_env=None,
    policy_kwargs={
        "project_model_class": MsPacManGenesisModel,
        # optimizer_class=torch.optim.Adam,
        # optimizer_kwargs=None,
    },
    verbose=1,
    # seed=None,
)
model.learn(total_timesteps=8192)
model.save("trained/model/PPO-MsPacMan-Genesis-001")


del model
model = PPO.load("trained/model/PPO-MsPacMan-Genesis-001")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
