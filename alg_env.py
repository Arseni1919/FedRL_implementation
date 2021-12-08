import math

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
    def __init__(self, x, y, agent_id, agent_type, metric_radius=1):
        self.x = x
        self.y = y
        self.id = agent_id
        self.type = agent_type
        self.metric_radius = metric_radius
        self.state_side_size = self.metric_radius * 2 + 1
        self.state_size = self.state_side_size ** 2
        # self.name = f'agent_{self.type}_{self.id}'
        self.name = f'{self.type}'
        self. domain = []
        self.state = []
        self.distance_type = 'chebyshev'
        # self.distance_type = 'cityblock'
        self.marker = 'p'

        if self.type == 'alpha':
            self.color = 'r'

        elif self.type == 'beta':
            self.color = 'b'

        else:
            raise RuntimeError('Unknown type!')


def set_domain_and_state_of_agents(agents, positions):
    for agent in agents:
        agent.domain = []
        pos_dict = {(pos.x, pos.y): pos for pos in positions}
        # self position
        agent.domain.append(pos_dict[(agent.x, agent.y)])

        for pos in positions:
            dist = cdist([[agent.x, agent.y]], [[pos.x, pos.y]], agent.distance_type)[0, 0]
            if dist <= agent.metric_radius:
                # if not pos.occupied:
                agent.domain.append(pos)

        side = agent.metric_radius * 2 + 1
        agent.state = np.zeros((side, side))
        for pos in agent.domain:
            agent.state[agent.metric_radius - (agent.y - pos.y), agent.metric_radius - (agent.x - pos.x)] = float(pos.block)


def distance(pos1, pos2):
    return math.sqrt(((pos1.x - pos2.x)**2) + ((pos1.y - pos2.y)**2))


class FedRLEnv:
    """
    The main duties of this wrapper:

    1. Receive inputs and transform outputs as tensors
    2. Normalize states of observations

    """
    def __init__(self, max_steps=25, side_size=32):
        self.steps_counter = 0
        self.max_steps = max_steps
        self.name = 'FedRL_env'
        self.side_size = side_size
        self.positions = []
        self.agents = []
        self.agent_dict = {}
        self.pos_dict = {}

        # FOR RENDERING
        self.reset()

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
        free_positions = list(filter(lambda x: not x.occupied, self.positions))
        pos1, pos2 = random.sample(free_positions, 2)
        self.agents.append(Agent(pos1.x, pos1.y, 101, 'alpha', metric_radius=1))
        self.agents.append(Agent(pos2.x, pos2.y, 201, 'beta', metric_radius=2))
        pos1.occupied = True
        pos2.occupied = True
        set_domain_and_state_of_agents(self.agents, self.positions)

        self.agent_dict = {agent.name: agent for agent in self.agents}
        self.pos_dict = {(pos.x, pos.y): pos for pos in self.positions}

        t_observations = {agent.name: torch.tensor(agent.state) for agent in self.agents}

        return t_observations

    def step(self, t_actions):
        actions = {agent_name: action.detach().item() for agent_name, action in t_actions.items()}
        # ACTION: 0,1,2,3,4 = stay ! ,  east > , south v , west < , north ^
        observations, done, infos = {}, False, {}
        rewards = {agent.name: 0 for agent in self.agents}

        # EXECUTE ACTIONS
        for agent_name, action in actions.items():
            r_l_1 = self._execute_action(agent_name, action)
            rewards[agent_name] += r_l_1
        set_domain_and_state_of_agents(self.agents, self.positions)

        # OBSERVATIONS
        observations = {agent.name: agent.state for agent in self.agents}

        # REWARDS
        dist = cdist([[self.agents[0].x, self.agents[0].y]], [[self.agents[1].x, self.agents[1].y]], 'cityblock')[0, 0]
        r_g = 50 if dist <= 2 else 0
        r_g += self.side_size / dist
        for agent in self.agents:
            rewards[agent.name] += r_g

        # DONE
        self.steps_counter += 1
        if self.steps_counter == self.max_steps or dist <= 2:
            done = True

        # INFO
        pass

        # TO TENSOR
        t_observations = {agent_name: torch.tensor(obs) for agent_name, obs in observations.items()}
        t_rewards = {agent_name: torch.tensor(reward) for agent_name, reward in rewards.items()}

        return t_observations, t_rewards, done, infos

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def _execute_action(self, agent_name, action):
        # ACTION: 0,1,2,3,4 = stay ! ,  east > , south v , west < , north ^
        agent = self.agent_dict[agent_name]
        return_value = -1
        # stay
        if action != 0:
            new_x, new_y = agent.x, agent.y
            curr_pos = self.pos_dict[(new_x, new_y)]

            new_x = new_x + 1 if action == 1 else new_x  # east >
            new_y = new_y - 1 if action == 2 else new_y  # south v
            new_x = new_x - 1 if action == 3 else new_x  # west <
            new_y = new_y + 1 if action == 4 else new_y  # north ^

            if (new_x, new_y) in self.pos_dict:
                pos = self.pos_dict[(new_x, new_y)]
                if not pos.block:
                    # print(f'occ: {pos.occupied}, block: {pos.block}')
                    curr_pos.occupied = False
                    agent.x = new_x
                    agent.y = new_y
                    pos.occupied = True
                else:
                    return_value = -10

        # t_return_value = torch.tensor(return_value)
        return return_value

    def render(self, mode='human'):
        return 'Plot is unavailable'

    def observation_spaces(self):
        return {agent.name: agent.state for agent in self.agents}

    def observation_sizes(self):
        return {agent.name: agent.state_size for agent in self.agents}

    def action_spaces(self):
        return {agent.name: list(range(5)) for agent in self.agents}

    def action_sizes(self):
        return {agent.name: 5 for agent in self.agents}
