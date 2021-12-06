import random

import pettingzoo

from alg_GLOBALS import *
from alg_env_wrapper import MultiAgentParallelEnvWrapper


def load_and_play(env_to_play, times, path_to_load_model):
    # Example runs
    load_dict = torch.load(path_to_load_model)
    model = load_dict['model']
    model.eval()
    # env_to_play.state_stat.running_mean = load_dict['mean']
    # env_to_play.state_stat.running_std = load_dict['std']
    # env_to_play.state_stat.len = load_dict['len']

    play_parallel_env(env_to_play, True, times, model=model)


def play_parallel_env(parallel_env, render=True, episodes=10, model=None):
    max_cycles = 500
    for episode in range(episodes):
        observations = parallel_env.reset()
        result_dict = {agent: 0 for agent in parallel_env.agents}
        for step in range(max_cycles):
            if model:
                actions = {agent: parallel_env.action_spaces(agent).sample() for agent in parallel_env.agents}
                pass
            else:
                actions = {agent: parallel_env.action_spaces(agent).sample() for agent in parallel_env.agents}
            observations, rewards, dones, infos = parallel_env.step(actions)
            for agent in parallel_env.agents:
                result_dict[agent] += rewards[agent]

            if render:
                parallel_env.render()
            if False not in dones.values():
                break

        print(f'[{episode + 1}] Game finished with result:')
        pprint(result_dict)
        print('---')
    parallel_env.close()


if __name__ == '__main__':
    # ENV = simple_v2.parallel_env(max_cycles=25, continuous_actions=True)
    ENV = simple_v2.parallel_env(max_cycles=25, continuous_actions=True)
    ENV = MultiAgentParallelEnvWrapper(ENV)

    # SEED
    SEED = 111
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    ENV.seed(SEED)

    n = 2
    # NOT FOR PARALLEL
    if isinstance(ENV, pettingzoo.AECEnv):
        print('Not parallel env')
        random_demo(ENV, render=True, episodes=n)
    if isinstance(ENV, pettingzoo.ParallelEnv) or isinstance(ENV, MultiAgentParallelEnvWrapper):
        print('Parallel env')
        play_parallel_env(ENV, render=True, episodes=n)
