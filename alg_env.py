from alg_GLOBALS import *


class Position:
    def __init__(self, x, y, block=False):
        self.x = x
        self.y = y
        self.block = block
        self.occupied = block

        self.color = 'k' if self.block else 'lightgray'
        self.marker = 's' if self.block else '.'


class Agent:
    def __init__(self, x, y, agent_type, metric_radius=1):
        self.x = x
        self.y = y
        self.type = agent_type
        self.metric_radius = metric_radius
        self. domain = []
        self.distance_type = 'chebyshev'
        # self.distance_type = 'cityblock'
        self.marker = 'p'

        if self.type == 'alpha':
            self.color = 'r'

        elif self.type == 'beta':
            self.color = 'b'

        else:
            raise RuntimeError('Unknown type!')


def get_domain_of_agents(agents, positions):
    for agent in agents:
        agent.domain = []
        pos_dict = {(pos.x, pos.y): pos for pos in positions}
        # self position
        agent.domain.append(pos_dict[(agent.x, agent.y)])

        for pos in positions:
            dist = cdist([[agent.x, agent.y]], [[pos.x, pos.y]], agent.distance_type)[0, 0]
            if dist <= agent.metric_radius:
                if not pos.occupied:
                    agent.domain.append(pos)


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
        self.side_size = 32
        self.positions = []
        self.agents = []

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

        # CREATE A NEW LIST OF POSITIONS
        self.positions = []
        for point_x in range(self.side_size):
            for point_y in range(self.side_size):
                block = True if random.random() > 0.9 else False
                self.positions.append(Position(x=point_x, y=point_y, block=block))

        # CREATE AGENTS
        self.agents = []
        free_positions = list(filter(lambda x: x.occupied, self.positions))
        pos1, pos2 = random.sample(free_positions, 2)
        self.agents.append(Agent(pos1.x, pos1.y, 'alpha', metric_radius=1))
        self.agents.append(Agent(pos2.x, pos2.y, 'beta', metric_radius=2))
        pos1.occupied = True
        pos2.occupied = True
        get_domain_of_agents(self.agents, self.positions)
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
