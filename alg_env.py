from alg_GLOBALS import *


class FedRLEnv:
    """
    The main duties of this wrapper:

    1. Receive inputs and transform outputs as tensors
    2. Normalize states of observations

    """
    def __init__(self, max_steps=25):
        self.steps_counter = 0
        self.max_steps = max_steps
        self.name = 'FedRL_env'

        # FOR RENDERING

    def observation_space(self, agent):
        return

    def observation_size(self, agent):
        return

    def action_space(self, agent):
        return

    def action_size(self, agent):
        return

    def reset(self):
        self.steps_counter = 0
        return

    def step(self, actions=None):
        observations, rewards, done, infos = {}, {}, False, {}
        self.steps_counter += 1
        if self.steps_counter == self.max_steps:
            done = True
        # print(f'step {self.steps_counter}')
        return observations, rewards, done, infos

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def _prepare_action(self):
        pass

    def render(self, mode='human'):
        pass
