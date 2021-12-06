from alg_GLOBALS import *
from alg_plotter import ALGPlotter
from alg_env import FedRLEnv
from alg_nets import CriticNet, ActorNet

def train():
    best_score = - math.inf
    for i_episode in range(M_EPISODE):

        done = False
        steps = 0
        scores = []
        observations = env.reset()

        while not done:
            actions = {agent.name: random.choice(env.action_spaces()[agent.name]) for agent in env.agents}
            new_observations, rewards, done, infos = env.step(actions)

            observations = new_observations
            scores.append(sum(rewards.values()))
            steps += 1
            plotter.plot(steps, env, scores)
            print('', end='')
        print(f'Finished episode {i_episode} with reward: {sum(scores)}')


    # SAVE
    # average_score = sum(average_result_dict.values())
    # if average_score > best_score:
    #     best_score = average_score
    #     save_results(SAVE_PATH, actor)


def load_and_play(env_to_load, times, path):
    # load env
    # TODO

    # play
    best_score = - math.inf
    for i_episode in range(times):

        done = False
        steps = 0
        scores = []
        observations = env.reset()

        while not done:
            actions = {agent.name: random.choice(env.action_spaces()[agent.name]) for agent in env.agents}
            new_observations, rewards, done, infos = env.step(actions)

            observations = new_observations
            scores.append(sum(rewards.values()))
            steps += 1
            plotter.plot(steps, env, scores)
            print('', end='')
        print(f'Finished episode {i_episode} with reward: {sum(scores)}')


if __name__ == '__main__':
    # --------------------------- # PARAMETERS # -------------------------- #
    N_UPDATES = 20
    M_EPISODE = 20
    BATCH_SIZE = 64  # size of the batches
    LR_CRITIC = 1e-2  # learning rate
    LR_ACTOR = 1e-2  # learning rate
    GAMMA = 0.95  # discount factor

    # --------------------------- # CREATE ENV # -------------------------- #
    NUMBER_OF_AGENTS = 1
    MAX_STEPS = 10
    # SIDE_SIZE = 8
    SIDE_SIZE = 16
    # SIDE_SIZE = 32
    ENV_NAME = ''
    env = FedRLEnv(max_steps=MAX_STEPS, side_size=SIDE_SIZE)

    NUMBER_OF_GAMES = 10

    # --------------------------- # NETS # -------------------------- #
    Q_alpha, Q_f_alpha, Q_beta, Q_f_beta = None, None, None, None
    for i_agent in env.agents:
        if i_agent.type == 'alpha':
            Q_alpha = CriticNet(i_agent.state_size, 5)
            Q_f_alpha = CriticNet(i_agent.state_size, 6)
        if i_agent.type == 'beta':
            Q_beta = CriticNet(i_agent.state_size, 5)
            Q_f_beta = CriticNet(i_agent.state_size, 6)

    # --------------------------- # OPTIMIZERS # -------------------------- #
    Q_alpha_optim = torch.optim.Adam(Q_alpha.parameters(), lr=LR_CRITIC)
    Q_f_alpha_optim = torch.optim.Adam(Q_f_alpha.parameters(), lr=LR_CRITIC)
    Q_beta_optim = torch.optim.Adam(Q_beta.parameters(), lr=LR_CRITIC)
    Q_f_beta_optim = torch.optim.Adam(Q_f_beta.parameters(), lr=LR_CRITIC)

    # --------------------------- # REPLAY BUFFER # -------------------------- #
    replay_buffer_alpha = deque(maxlen=10000)
    replay_buffer_beta = deque(maxlen=10000)

    # --------------------------- # FOR PLOT # -------------------------- #
    PLOT_PER = 1
    NEPTUNE = False
    PLOT_LIVE = True
    SAVE_RESULTS = True
    SAVE_PATH = f'data/actor_{ENV_NAME}.pt'
    plotter = ALGPlotter(plot_life=PLOT_LIVE, plot_neptune=NEPTUNE, name='my_run_FedRL', tags=[ENV_NAME])
    plotter.neptune_init()

    # --------------------------- # PLOTTER INIT # -------------------------- #

    # --------------------------- # SEED # -------------------------- #
    SEED = 111
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    env.seed(SEED)
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #

    # MAIN PROCESS
    train()

    # Example Plays
    print(colored('Example run...', 'green'))
    load_and_play(env, 1, SAVE_PATH)
