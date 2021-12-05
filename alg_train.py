from alg_GLOBALS import *
from alg_plotter import ALGPlotter
from alg_env import FedRLEnv


def train():
    best_score = - math.inf
    for i_episode in range(M_EPISODE):

        done = False
        steps = 0
        env.reset()

        while not done:
            actions = {}
            observations, rewards, done, infos = env.step()

            steps += 1
            plotter.plot(steps, env, [], [])
        print(f'Finished episode {i_episode} with reward: ')


    # SAVE
    # average_score = sum(average_result_dict.values())
    # if average_score > best_score:
    #     best_score = average_score
    #     save_results(SAVE_PATH, actor)


def load_and_play(env_to_load, times, path):
    pass


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
    ENV_NAME = ''
    env = FedRLEnv()

    NUMBER_OF_GAMES = 10

    # --------------------------- # NETS # -------------------------- #
    # critic = CriticNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    # actor = ActorNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    # actor_old = ActorNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    # --------------------------- # OPTIMIZERS # -------------------------- #
    # critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    # actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)

    # --------------------------- # REPLAY BUFFER # -------------------------- #
    # replay_buffer = ReplayBuffer()

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
