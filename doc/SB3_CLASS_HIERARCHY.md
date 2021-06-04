### Relevant Class Hierarchy of `stable-baselines3` package ###

All class implementations are in separate files with the class name as the file name for easier
reverse engineering and implementation reading.


#### Policy ####

    BaseModel(torch.nn.Module, abc.ABC)
     |
     v
    BasePolicy
     |
     v
    ActorCriticPolicy


#### Algorithm ####

    BaseAlgorithm(abc.ABC)
     |
     v
    OnPolicyAlgorithm
     |
     v
    PPO
