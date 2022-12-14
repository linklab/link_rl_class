import numpy as np
import random

from online_learning.common.a_grid_word import GridWorld
from online_learning.common.e_util import draw_grid_world_action_values_image

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]


def get_exploring_start_state():
    while True:
        i = random.randrange(GRID_HEIGHT)
        j = random.randrange(GRID_WIDTH)
        if (i, j) not in TERMINAL_STATES:
            break
    return (i, j)


# 환경에서 무작위로 에피소드 생성
def generate_random_episode_and_state_actions(env):
    episode = []
    visited_state_actions = []

    # 탐험적 시작 기반 몬테카를로 제어
    initial_state = get_exploring_start_state()
    env.moveto(initial_state)

    state = initial_state
    done = False
    while not done:
        # 상태에 관계없이 항상 4가지 행동 중 하나를 선택하여 수행
        action = random.randrange(env.NUM_ACTIONS)

        next_state, reward, done, _ = env.step(action)

        episode.append(((state, action), reward))
        visited_state_actions.append((state, action))

        state = next_state

    return episode, visited_state_actions


# 첫 방문 행동 가치 MC 예측
def first_visit_mc_prediction(env, gamma, num_iter):
    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_action_values = np.zeros((GRID_HEIGHT, GRID_WIDTH, env.NUM_ACTIONS))
    returns = dict()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            for action in env.ACTIONS:
                returns[((i, j), action)] = list()

    for i in range(num_iter):
        episode, visited_state_actions = generate_random_episode_and_state_actions(env)

        G = 0
        for idx, ((state, action), reward) in enumerate(reversed(episode)):
            G = gamma * G + reward

            value_prediction_conditions = [
                (state, action) not in \
                visited_state_actions[:len(visited_state_actions) - idx - 1],
                state not in TERMINAL_STATES
            ]

            if all(value_prediction_conditions):
                returns[(state, action)].append(G)
                state_action_values[state[0], state[1], action] \
                    = np.mean(returns[(state, action)])

        if i % 1000 == 0:
            print("Iteration: {0}".format(i))

    print("Iteration: {0}".format(i))

    return state_action_values, returns


# 모든 방문 행동 가치 MC 예측
def every_visit_mc_prediction(env, gamma, num_iter):
    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_action_values = np.zeros((GRID_HEIGHT, GRID_WIDTH, env.NUM_ACTIONS))
    returns = dict()
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            for action in env.ACTIONS:
                returns[((i, j), action)] = list()

    for i in range(num_iter):
        episode, _ = generate_random_episode_and_state_actions(env)

        G = 0
        for idx, ((state, action), reward) in enumerate(reversed(episode)):
            G = gamma * G + reward

            value_prediction_conditions = [
                state not in TERMINAL_STATES
            ]

            if all(value_prediction_conditions):
                returns[(state, action)].append(G)
                state_action_values[state[0], state[1], action] \
                    = np.mean(returns[(state, action)])

        if i % 1000 == 0:
            print("Iteration: {0}".format(i))

    print("Iteration: {0}".format(i))

    return state_action_values, returns


def action_value_prediction_main():
    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=None,
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )
    env.reset()

    state_action_values, returns = first_visit_mc_prediction(env, 1.0, 10000)
    draw_grid_world_action_values_image(
        state_action_values,
        GRID_HEIGHT, GRID_WIDTH,
        env.NUM_ACTIONS,
        env.ACTION_SYMBOLS
    )

    state_action_values, returns = every_visit_mc_prediction(env, 1.0, 10000)
    draw_grid_world_action_values_image(
        state_action_values,
        GRID_HEIGHT, GRID_WIDTH,
        env.NUM_ACTIONS,
        env.ACTION_SYMBOLS
    )


if __name__ == "__main__":
    action_value_prediction_main()