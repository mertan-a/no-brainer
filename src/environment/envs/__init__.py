
# import envs and necessary gym packages
from environment.envs.custom_env import BASIC_ENVIRONMENT, BASIC_ENVIRONMENT_TEST
from environment.envs.basic_directional_env import DIRECTIONAL_ENVIRONMENT, DIRECTIONAL_ENVIRONMENT_TEST
from environment.envs.logic_env import LOGIC_ENVIRONMENT, LOGIC_ENVIRONMENT_TEST
from gym.envs.registration import register

# register the env using gym's interface
register(
    id = 'BasicEnv-v0',
    entry_point = 'environment.envs.custom_env:BASIC_ENVIRONMENT',
    max_episode_steps = 1000
)

register(
    id = 'BasicEnvTest-v0',
    entry_point = 'environment.envs.custom_env:BASIC_ENVIRONMENT_TEST',
    max_episode_steps = 1000
)

register(
        id = 'BasicDirectionalEnv-v0',
        entry_point = 'environment.envs.basic_directional_env:DIRECTIONAL_ENVIRONMENT',
        max_episode_steps = 1000
)

register(
        id = 'BasicDirectionalEnvTest-v0',
        entry_point = 'environment.envs.basic_directional_env:DIRECTIONAL_ENVIRONMENT_TEST',
        max_episode_steps = 1000
)

register(
        id = 'LogicEnv-v0',
        entry_point = 'environment.envs.logic_env:LOGIC_ENVIRONMENT',
        max_episode_steps = 1000
)

register(
        id = 'LogicEnvTest-v0',
        entry_point = 'environment.envs.logic_env:LOGIC_ENVIRONMENT_TEST',
        max_episode_steps = 1000
)
