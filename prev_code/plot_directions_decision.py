import math
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import random

from fisher_function import create_X_matrix


def count_time(function):
    def wrapper(*args, **kwargs):
        time_start = time.time()
        func_output = function(*args, **kwargs)
        time_end = time.time()
        print(f'time to finish: {time_end - time_start : .2f}')
        return func_output
    return wrapper


class Agent:
    def __init__(self, name):
        self.x = -1
        self.y = -1
        self.name = name


class Robot(Agent):
    def __init__(self, name, cred=0):
        super(Robot, self).__init__(name)
        self.cred = cred
        self.marker = "o"
        self.color = 'blue'
        self.dx = 0
        self.dy = 0


class Target(Agent):
    def __init__(self, name, req=0):
        super(Target, self).__init__(name)
        self.req = req
        self.marker = "s"
        self.color = 'orange'


def get_distance(agent1, agent2):
    initial_distance = math.sqrt(math.pow(agent1.x - agent2.x, 2) + math.pow(agent1.y - agent2.y, 2))
    final_distance = initial_distance if initial_distance > 1 else 1
    return final_distance


def assign_random_positions(robots, targets, field):
    all_agents = []
    all_agents.extend(robots)
    all_agents.extend(targets)
    while len(all_agents) > 0:
        rand_x, rand_y = random.randint(0, FIELD_WIDTH-1), random.randint(0, FIELD_WIDTH-1)
        if field[rand_x, rand_y] == 0:
            agent = all_agents[0]
            agent.x = rand_x
            agent.y = rand_y
            field[rand_x, rand_y] = 1
            all_agents.remove(agent)


def get_weighted_coverage_list(coverage_list):
    weighted_coverage_list = []
    item = coverage_list[0]
    alpha = 0.9
    for next_item in coverage_list:
        item = item * alpha + (1 - alpha) * next_item
        weighted_coverage_list.append(item)
    return weighted_coverage_list


# @count_time
def plot_field(fig, ax, robots, targets, coverage_list):
    # CLEAR THE PLOT BEFORE
    ax_list = fig.axes
    ax_list[0].cla()
    ax_list[1].cla()

    # PLOT FIELD
    for point_x in range(FIELD_WIDTH):
        for point_y in range(FIELD_WIDTH):
            ax_list[0].plot(point_x, point_y, color='lightgray', marker='.', markersize=1)

    # PLOT ROBOTS WITH DIRECTIONS AND TARGETS
    for agent in [*robots, *targets]:
        ax_list[0].plot(agent.x, agent.y, color=agent.color, marker=agent.marker, markersize=12)
        if 'robot' in agent.name:
            ax_list[0].annotate("", xy=(agent.dx, agent.dy), xytext=(agent.x, agent.y),
                                textcoords='data', xycoords='data', arrowprops=dict(connectionstyle="arc3"),)

    # PLOT REQS AND CREDS
    for robot in robots:
        ax_list[0].text(robot.x - 2, robot.y - 2, f'{robot.cred}', dict(size=9), weight="bold", color='b')
    for target in targets:
        ax_list[0].text(target.x + 2, target.y + 2, f'{remained_coverage_for_target(target, robots): .2f}',
                        dict(size=9), weight="bold", color='brown')

    # PLOT COVERAGE
    ax_list[1].plot(coverage_list, label='Rem_Cov')
    ax_list[1].plot(get_weighted_coverage_list(coverage_list), color='r', label='weighted average Rem_Cov')

    # LABELS
    ax_list[0].title.set_text('Field')
    ax_list[1].title.set_text('Remained Coverage')
    ax_list[1].legend()

    # SHOWING..
    plt.pause(0.005)
    # time.sleep(1)


def create_input_to_fisher(robots, targets):
    mat = []
    for target in targets:
        mat.append([])
        for robot in robots:
            distance = get_distance(target, robot)
            demand = MIN_DEMAND if distance > SR else robot.cred/distance
            # demand = robot.cred/distance
            mat[-1].append(demand)
    return mat


def set_dx_dy_for_robots(robots, targets, output_fisher):
    # print(sum(x[1] for x in output_fisher))
    # print(sum(x[2] for x in output_fisher))
    for robot_indx, robot in enumerate (robots):
        to_x, to_y = 0, 0
        for target_indx, target in enumerate(targets):
            distance = get_distance(robot, target)
            dx = (target.x - robot.x) / distance
            dy = (target.y - robot.y) / distance
            weight = output_fisher[target_indx][robot_indx]
            to_x += dx * weight
            to_y += dy * weight
        robot.dx = robot.x + to_x * STEP
        robot.dy = robot.y + to_y * STEP


def move(robots, targets, field):
    for robot in robots:
        robot.x = robot.dx
        robot.y = robot.dy


def close():
    plt.close()


def remained_coverage_for_target(target, robots):
    initial_req = target.req
    for robot in robots:
        distance = get_distance(robot, target)
        if distance <= SR:
            initial_req = max(0, initial_req - (robot.cred / distance))
            # initial_req -= (robot.cred / distance)
        # else:
        #     initial_req -= MIN_DEMAND
    return initial_req


def get_coverage(robots, targets):
    remained_coverage = 0
    for target in targets:
        initial_req = remained_coverage_for_target(target, robots)
        remained_coverage += initial_req
    return remained_coverage


def create_robots_and_targets():
    if LOAD_PREV:
        print('loading previous robots and targets...')
        with open(f'data/robots.info', "rb") as input_file:
            robots = pickle.load(input_file)
        with open(f'data/targets.info', "rb") as input_file:
            targets = pickle.load(input_file)
        with open(f'data/field.info', "rb") as input_file:
            field = pickle.load(input_file)
    else:
        print('creating new robots and targets...')
        robots = [Robot(f'robot_{x}', random.randint(10, 30)) for x in range(N_ROBOTS)]
        targets = [Target(f'target_{x}', 100) for x in range(N_TARGETS)]
        field = np.zeros((FIELD_WIDTH, FIELD_WIDTH))
        assign_random_positions(robots, targets, field)
        if SAVE_PREV:
            with open(f'data/robots.info', "wb") as output_file:
                pickle.dump(robots, output_file)
            with open(f'data/targets.info', "wb") as output_file:
                pickle.dump(targets, output_file)
            with open(f'data/field.info', "wb") as output_file:
                pickle.dump(field, output_file)

    return robots, targets, field


def get_fisher(input_fisher):
    return create_X_matrix(input_fisher)


def main():
    pass
    # CREATE ROBOTS AND TARGETS
    robots, targets, field = create_robots_and_targets()

    # SETTING FOR GRAPHS
    coverage_list = []
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    for iteration in range(ITERATIONS):
        # CREATE INPUT FISHER MATRIX
        input_fisher = create_input_to_fisher(robots, targets)

        # GET OUTPUT FISHER MATRIX
        output_fisher = get_fisher(input_fisher)

        # DECIDE ON DIRECTION
        set_dx_dy_for_robots(robots, targets, output_fisher)

        # CALCULATE COVERAGE
        remained_coverage = get_coverage(robots, targets)
        coverage_list.append(remained_coverage)

        # PLOT THE DECISION
        if iteration % PLOT_EVERY == 0 or iteration + 1 == ITERATIONS:
            plot_field(fig, ax, robots, targets, coverage_list)
            print(f'\r(iter: {iteration}) remained_coverage: {remained_coverage}')

        # CHANGE POSITIONS
        move(robots, targets, field)

    # WAIT..
    input()

    # END OF SIMULATION
    close()


if __name__ == '__main__':
    FIELD_WIDTH = 50
    N_ROBOTS = 10
    N_TARGETS = 10
    SR = 50
    STEP = 0.2
    ITERATIONS = 2000
    PLOT_EVERY = 20
    MIN_DEMAND = 0.001
    SAVE_PREV = True
    LOAD_PREV = True
    # LOAD_PREV = False

    main()

