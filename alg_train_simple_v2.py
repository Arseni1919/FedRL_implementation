import math

from alg_GLOBALS import *
from alg_nets import *
from alg_plotter import ALGPlotter
from alg_env_wrapper import MultiAgentParallelEnvWrapper
from alg_env_demo import play_parallel_env, load_and_play


def train():
    best_score = - math.inf

    for i_update in range(N_UPDATES):

        with torch.no_grad():
            # SAMPLE TRAJECTORIES
            average_result_dict = get_trajectories()

            # COMPUTE RETURNS AND ADVANTAGES
            compute_returns_and_advantages()
            pass

        # UPDATE CRITIC
        update_critic()

        # UPDATE ACTOR
        update_actors()

        # PLOTTER, NEPTUNE
        # plotter.plot()

        # RENDER, PRINT
        if i_update > N_UPDATES - 5:
            print(f'Update {i_update + 1}:')
            play_parallel_env(env, True, 1, actor_old)
        else:
            print(f'Update {i_update + 1}, average results: {average_result_dict} \n---')

        # SAVE
        average_score = sum(average_result_dict.values())
        if average_score > best_score:
            best_score = average_score
            save_results(SAVE_PATH, actor)

    env.close()


def get_trajectories():
    # PLAY EPISODE
    observations = env.reset()
    result_dict = {agent: 0 for agent in env.agents}
    while True:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, dones, infos = env.step(actions)
        for agent in env.agents:
            result_dict[agent] += rewards[agent]

        if False not in dones.values():
            break
    return result_dict


def compute_returns_and_advantages():
    pass


def update_critic():
    pass


def update_actors():
    pass


def save_results(path, model_to_save):
    # SAVE
    if SAVE_RESULTS:
        # SAVING...
        print(colored(f"Saving model...", 'green'))
        torch.save({
            'model': model_to_save,
            # 'len': env.state_stat.len,
            # 'mean': env.state_stat.running_mean,
            # 'std': env.state_stat.running_std,
        }, path)
    return path


if __name__ == '__main__':

    # --------------------------- # PARAMETERS # -------------------------- #
    N_UPDATES = 20
    BATCH_SIZE = 64  # size of the batches
    REPLAY_BUFFER_SIZE = BATCH_SIZE * 1000
    LR_CRITIC = 1e-2  # learning rate
    LR_ACTOR = 1e-2  # learning rate
    GAMMA = 0.95  # discount factor
    ACT_NOISE = 0.5  # actuator noise
    POLYAK = 0.99
    VAL_EVERY = 2000
    TRAIN_EVERY = 100

    # --------------------------- # CREATE ENV # -------------------------- #
    NUMBER_OF_AGENTS = 1
    # ENV = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=MAX_CYCLES, continuous_actions=True)
    ENV_NAME = 'simple_v2'
    ENV = simple_v2.parallel_env(max_cycles=25, continuous_actions=True)
    env = MultiAgentParallelEnvWrapper(ENV)

    NUMBER_OF_GAMES = 10

    # --------------------------- # NETS # -------------------------- #
    # critic = CriticNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    actor = ActorNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    actor_old = ActorNet(obs_size=env.observation_size(env.agents[0]), n_actions=env.action_size(env.agents[0]))
    # --------------------------- # OPTIMIZERS # -------------------------- #
    # critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)

    # --------------------------- # REPLAY BUFFER # -------------------------- #
    # replay_buffer = ReplayBuffer()

    # --------------------------- # FOR PLOT # -------------------------- #
    PLOT_PER = 1
    NEPTUNE = False
    PLOT_LIVE = True
    SAVE_RESULTS = True
    SAVE_PATH = f'data/actor_{ENV_NAME}.pt'
    plotter = ALGPlotter(plot_life=PLOT_LIVE, plot_neptune=NEPTUNE, name='my_run_ppo', tags=[ENV_NAME])

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


# print(colored(f'\n~[WARNING]: {message}', 'yellow'), end=end)


























