from alg_GLOBALS import *


def load_and_play(env_to_load, times, path, plotter, env=None):
    # load env
    # TODO
    if not env:
        return

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