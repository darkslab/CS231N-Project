import retro
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from lib.override.policies import ProjectActorCriticPolicy


env = make_vec_env(
    retro.make,
    n_envs=1,
    env_kwargs={
        "game": "Airstriker-Genesis",
    },
)

model = PPO(
    ProjectActorCriticPolicy,
    env,
    # learning_rate=3e-4,
    # n_steps=2048,
    batch_size=128,
    # n_epochs=10,
    # gamma=0.99,
    # tensorboard_log=None,
    # create_eval_env=None,
    # policy_kwargs=None,
        # optimizer_class=torch.optim.Adam,
        # optimizer_kwargs=None,
    verbose=1,
    # seed=None,
)
# model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=8192)
model.save("trained/PPO-Airstriker-Genesis")


# # del model
# model = PPO.load("trained/PPO-Airstriker-Genesis")
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
