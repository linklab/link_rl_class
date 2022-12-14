import numpy as np

from online_learning.common.a_grid_word import GridWorld
from online_learning.common.e_util import draw_grid_world_optimal_policy_image

EPSILON = 0.1

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT-1, GRID_WIDTH-1)]
DISCOUNT_RATE = 1.0
MAX_EPISODES = 100


# 비어있는 행동 가치 테이블을 0으로 초기화하며 생성함
def generate_initial_q_value_and_return(env):
    state_action_values = np.zeros((GRID_HEIGHT, GRID_WIDTH, env.NUM_ACTIONS))
    returns = dict()

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            for action in env.ACTIONS:
                returns[((i, j), action)] = list()

    return state_action_values, returns


# 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
# 초기에 각 행동의 선택 확률은 모두 같음
def generate_initial_random_policy(env):
    policy = dict()

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            actions = []
            prob = []
            for action in env.ACTIONS:
                actions.append(action)
                prob.append(0.25)
            policy[(i, j)] = (actions, prob)

    return policy


# 환경에서 현재 정책에 입각하여 에피소드(현재 상태, 행동, 다음 상태, 보상) 생성
def generate_episode(env, policy):
    episode = []
    visited_state_actions = []

    state = env.reset()  # 초기 상태 고정 (0, 1)

    done = False
    while not done:
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]

        next_state, reward, done, _ = env.step(action)

        episode.append(((state, action), reward))
        visited_state_actions.append((state, action))

        state = next_state

    return episode, visited_state_actions


# 첫 방문 행동 가치 MC 추정 함수
def first_visit_mc_prediction(
    state_action_values, returns, episode, visited_state_actions):
    G = 0
    for idx, ((state, action), reward) in enumerate(reversed(episode)):
        G = DISCOUNT_RATE * G + reward

        value_prediction_conditions = [
            (state, action) not in \
                visited_state_actions[:len(visited_state_actions) - idx - 1],
            state not in TERMINAL_STATES
        ]

        if all(value_prediction_conditions):
            returns[(state, action)].append(G)
            state_action_values[state[0], state[1], action] \
                = np.mean(returns[(state, action)])


# 소프트 탐욕적 정책 생성
def generate_soft_greedy_policy(env, state_action_values, policy):
    new_policy = dict()

    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            actions = []
            action_probs = []
            if (i, j) in TERMINAL_STATES:
                for action in env.ACTIONS:
                    actions.append(action)
                    action_probs.append(0.25)
                new_policy[(i, j)] = (actions, action_probs)
            else:
                max_prob_actions = [action_ for action_, value_
                                    in enumerate(state_action_values[i, j, :]) if
                                    value_ == np.max(state_action_values[i, j, :])]
                for action in env.ACTIONS:
                    actions.append(action)
                    if action in max_prob_actions:
                        action_probs.append(
                            (1 - EPSILON) / len(max_prob_actions) \
                            + EPSILON / env.NUM_ACTIONS
                        )
                    else:
                        action_probs.append(
                            EPSILON / env.NUM_ACTIONS
                        )

                new_policy[(i, j)] = (actions, action_probs)

    error = 0.0
    for i in range(GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            error += np.sum(
                np.absolute(
                    np.array(policy[(i, j)][1]) - np.array(new_policy[(i, j)][1])
                )
            )

    return new_policy, error


def soft_policy_control_main():
    # 그리드 월드 환경 객체 생성
    env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=(0, 1),       # 시작 상태 고정
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )

    # 비어있는 상태-가치 함수를 0으로 초기화하며 생성함
    state_action_values, returns = generate_initial_q_value_and_return(env)

    # 초기 임의 정책 생성
    policy = generate_initial_random_policy(env)

    iter_num = 0

    print("[[[ MC 제어 반복 시작! ]]]")
    while iter_num < MAX_EPISODES:
        iter_num += 1

        episode, visited_state_actions = generate_episode(env, policy)
        print("*** 에피소드 생성 완료 ***")

        first_visit_mc_prediction(
            state_action_values, returns, episode, visited_state_actions
        )
        print("*** MC 예측 수행 완료 ***")

        policy, error = generate_soft_greedy_policy(
            env, state_action_values, policy
        )
        print("*** 정책 개선 [에러 값: {0:9.7f}], 총 반복 수: {1} ***\n".format(
            error, iter_num
        ))

    print("[[[ MC 제어 반복 종료! ]]]\n\n")

    draw_grid_world_optimal_policy_image(
        policy,
        GRID_HEIGHT, GRID_WIDTH,
        env.ACTION_SYMBOLS
    )


if __name__ == "__main__":
    soft_policy_control_main()