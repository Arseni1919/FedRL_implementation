import torch

from alg_GLOBALS import *
from alg_plotter import ALGPlotter
from alg_env import FedRLEnv
from alg_nets import CriticNet, ActorNet
from alg_functions import load_and_play


def train():
    best_score = - math.inf
    global_steps = 0
    for i_episode in range(M_EPISODE):

        done = False
        steps = 0
        scores = []
        t_observations = env.reset()

        while not done:

            # PREPARATION
            epsilon = EPSILON_MAX - global_steps / (M_EPISODE * MAX_STEPS) * (EPSILON_MAX - EPSILON_MIN)
            t_alpha_obs = t_observations['alpha']
            t_beta_obs = t_observations['beta']

            # COMPUTE C_BETA
            C_beta, t_beta_action = beta_compute_q_beta(t_beta_obs, epsilon)

            # SELECT ACTION
            if random.random() < epsilon:
                t_alpha_action = torch.tensor(random.choice(env.action_spaces()['alpha']))
            else:
                t_alpha_action_q_values = Q_alpha(t_alpha_obs)
                q_f_alpha_values_list = []
                for t_action_q_value in t_alpha_action_q_values:
                    input_q_f = torch.stack((t_action_q_value, C_beta))
                    q_f_alpha_values_list.append(Q_f_alpha(input_q_f))
                q_f_alpha_values_list = torch.stack(q_f_alpha_values_list)
                t_alpha_action = torch.argmax(q_f_alpha_values_list)
            t_actions = {'alpha': t_alpha_action, 'beta': t_beta_action}

            # EXECUTE ACTION
            t_new_observations, t_rewards, done, infos = env.step(t_actions)

            # OBSERVE AND STORE
            replay_buffer_alpha.append((t_alpha_obs, t_alpha_action, t_rewards['alpha'], t_new_observations['alpha']))

            # SAMPLE
            sample_index = random.randint(0, len(replay_buffer_alpha) - 1)
            sample_tuple_alpha = replay_buffer_alpha[sample_index]
            t_sample_alpha_obs, t_sample_alpha_action, t_sample_alpha_reward, t_sample_alpha_new_observation = sample_tuple_alpha

            # CALL C_BETA
            C_beta = beta_compute_q_beta_j(sample_index)

            # COMPUTE Y
            t_sample_alpha_action_q_values = Q_alpha(t_sample_alpha_new_observation)  # !
            q_f_alpha_values_list = []
            for t_sample_action_q_value in t_sample_alpha_action_q_values:
                input_q_f = torch.stack((t_sample_action_q_value, C_beta))
                q_f_alpha_values_list.append(Q_f_alpha(input_q_f))
            q_f_alpha_values_list = torch.stack(q_f_alpha_values_list)
            max_C_f_alpha = torch.max(q_f_alpha_values_list)

            y_j = t_sample_alpha_reward + GAMMA * max_C_f_alpha
            # t_sample_alpha_max_a = torch.argmax(Q_f_alpha(t_sample_alpha_beta_obs))

            # UPDATE ALPHA
            alpha_update_q(y_j, t_sample_alpha_obs, t_sample_alpha_action, C_beta)

            # COMPUTE C_ALPHA
            C_alpha = Q_alpha(t_sample_alpha_obs)[t_sample_alpha_action]

            # UPDATE BETA
            beta_update_q(y_j, sample_index, C_alpha)

            # UPDATE OBSERVATIONS VARIABLE
            t_observations = t_new_observations

            # PLOT
            scores.append(sum(t_rewards.values()).item())
            steps += 1
            global_steps += 1
            plotter.plot(steps, env, scores)
            plotter.neptune_plot({'epsilon': epsilon})

        # PRINT AND SAVE
        print(f'Finished episode {i_episode} with reward: {sum(scores)}')
        # average_score = sum(average_result_dict.values())
        # if average_score > best_score:
        #     best_score = average_score
        #     save_results(SAVE_PATH, actor)


def beta_compute_q_beta(t_beta_obs, epsilon):

    if random.random() < epsilon:
        t_beta_action = torch.tensor(random.choice(env.action_spaces()['beta']))
        C_beta = Q_beta(t_beta_obs)[t_beta_action]
    else:
        t_beta_action = torch.argmax(Q_beta(t_beta_obs))
        C_beta = torch.max(Q_beta(t_beta_obs))
    replay_buffer_beta.append((t_beta_obs, t_beta_action))

    return C_beta, t_beta_action


def beta_compute_q_beta_j(sample_index):
    sample_tuple_beta = replay_buffer_beta[sample_index]
    t_beta_obs, t_beta_action = sample_tuple_beta
    C_beta = Q_beta(t_beta_obs)[t_beta_action]
    return C_beta


def beta_update_q(y_j, sample_index, C_alpha):
    sample_tuple_beta = replay_buffer_beta[sample_index]
    t_sample_beta_obs, t_sample_beta_action = sample_tuple_beta

    # UPDATE BETA
    C_beta = Q_beta(t_sample_beta_obs)[t_sample_beta_action]
    input_q_f = torch.stack((C_alpha, C_beta))
    q_f_beta = Q_f_beta(input_q_f).squeeze().float()
    y_j = y_j.detach().float()
    beta_loss = nn.MSELoss()(q_f_beta, y_j)
    Q_beta_optim.zero_grad()
    beta_loss.backward()
    Q_beta_optim.step()


def alpha_update_q(y_j, t_sample_alpha_obs, t_sample_alpha_action, C_beta):
    C_alpha = Q_alpha(t_sample_alpha_obs)[t_sample_alpha_action]
    input_q_f = torch.stack((C_alpha, C_beta))
    q_f_alpha = Q_f_alpha(input_q_f).squeeze().float()
    y_j = y_j.detach().float()
    alpha_loss = nn.MSELoss()(q_f_alpha, y_j)
    Q_alpha_optim.zero_grad()
    alpha_loss.backward()
    Q_alpha_optim.step()


if __name__ == '__main__':
    # --------------------------- # PARAMETERS # -------------------------- #
    N_UPDATES = 20
    M_EPISODE = 20
    BATCH_SIZE = 64  # size of the batches
    LR_CRITIC = 1e-2  # learning rate
    LR_ACTOR = 1e-2  # learning rate
    GAMMA = 0.95  # discount factor
    EPSILON_MAX = 0.9
    EPSILON_MIN = 0.01

    # --------------------------- # CREATE ENV # -------------------------- #
    NUMBER_OF_AGENTS = 1
    MAX_STEPS = 10
    # SIDE_SIZE = 8
    SIDE_SIZE = 16
    # SIDE_SIZE = 32
    ENV_NAME = 'grid'
    env = FedRLEnv(max_steps=MAX_STEPS, side_size=SIDE_SIZE)

    NUMBER_OF_GAMES = 10

    # --------------------------- # NETS # -------------------------- #
    Q_alpha, Q_f_alpha, Q_beta, Q_f_beta = None, None, None, None
    for i_agent in env.agents:
        if i_agent.type == 'alpha':
            Q_alpha = ActorNet(i_agent.state_size, 5)
            Q_f_alpha = ActorNet(2, 1)
        if i_agent.type == 'beta':
            Q_beta = ActorNet(i_agent.state_size, 5)
            Q_f_beta = ActorNet(2, 1)

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
    SAVE_PATH = f'data/critic_{ENV_NAME}.pt'
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
    load_and_play(env, 1, SAVE_PATH, plotter, env)
