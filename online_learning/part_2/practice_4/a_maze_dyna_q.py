import random
import numpy as np
import matplotlib.pyplot as plt

# 최대 에피소드
from online_learning.common.c_maze import Maze

MAX_EPISODES = 50

# 감가율
GAMMA = 0.95

# 탐색(exploration) 확률 파라미터
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.01
LAST_SCHEDULED_EPISODES = 30

# 스텝 사이즈
ALPHA = 0.1

# 총 실험 횟수 (성능에 대한 평균을 구하기 위함)
TOTAL_RUNS = 10


def epsilon_scheduled(current_episode):
    fraction = min(current_episode / LAST_SCHEDULED_EPISODES, 1.0)
    epsilon = min(
        INITIAL_EPSILON + fraction * (FINAL_EPSILON - INITIAL_EPSILON),
        INITIAL_EPSILON
    )
    return epsilon


# Dyna-Q의 계획 과정에서 사용하는 간단한 모델
class EnvModel:
    def __init__(self):
        self.model = dict()

    # 경험 샘플 저장
    def store(self, state, action, reward, next_state):
        if state not in self.model.keys():
            self.model[state] = dict()

        self.model[state][action] = [reward, next_state]

    # 저장해 둔 경험 샘플들에서 임으로 선택하여 반환
    def sample(self):
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))
        reward, next_state = self.model[state][action]
        return state, action, reward, next_state


# 비어있는 행동 가치 테이블을 0~1 사이의 임의의 값으로 초기화하며 생성함
def generate_initial_q_value(env):
    q_table = np.zeros((env.MAZE_HEIGHT, env.MAZE_WIDTH, env.NUM_ACTIONS))

    for i in range(env.MAZE_HEIGHT):
        for j in range(env.MAZE_WIDTH):
            if (i, j) not in env.GOAL_STATES:
                for action in env.ACTIONS:
                    q_table[i, j, action] = random.random()
    return q_table


# 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
# 초기에 각 행동의 선택 확률은 모두 같음
def generate_initial_random_policy(env):
    policy = dict()

    for i in range(env.MAZE_HEIGHT):
        for j in range(env.MAZE_WIDTH):
            if (i, j) not in env.GOAL_STATES:
                actions = []
                prob = []
                for action in env.ACTIONS:
                    actions.append(action)
                    prob.append(0.25)

                policy[(i, j)] = (actions, prob)

    return policy


# epsilon-탐욕적 정책 갱신
def update_epsilon_greedy_policy(env, state, q_value, policy, current_episode):
    max_prob_actions = [action_ for action_, value_
                        in enumerate(q_value[state[0], state[1], :]) if
                        value_ == np.max(q_value[state[0], state[1], :])]

    epsilon = epsilon_scheduled(current_episode)

    actions = []
    action_probs = []
    for action in env.ACTIONS:
        actions.append(action)
        if action in max_prob_actions:
            action_probs.append(
                (1 - epsilon) / len(max_prob_actions) + epsilon / env.NUM_ACTIONS
            )
        else:
            action_probs.append(
                epsilon / env.NUM_ACTIONS
            )

    policy[state] = (actions, action_probs)


# Dyna-Q 알고리즘의 각 에피소드 별 학습
# @q_table: 행동 가치 테이블, dyna_q 함수 수행 후 값이 갱신 됨
def dyna_q(
        q_table, policy, env_model, env, episode, planning_repeat, step_size=ALPHA):
    state = env.reset()
    steps = 0
    rewards = 0.0

    done = False

    while not done:
        # 타임 스텝 기록
        steps += 1

        # 행동 얻어오기
        actions, prob = policy[state]
        action = np.random.choice(actions, size=1, p=prob)[0]

        # 행동 수행후
        next_state, reward, done, _ = env.step(action)

        # Q-러닝 갱신
        target = reward + GAMMA * \
                 np.max(q_table[next_state[0], next_state[1], :])

        q_table[state[0], state[1], action] += step_size * \
                                               (target - q_table[state[0], state[1], action])

        # 경험 샘플을 모델에 저장 (모델 구성)
        env_model.store(state, action, reward, next_state)

        # 모델로 부터 샘플 얻어오면서 Q-계획 반복 수행
        for t in range(planning_repeat):
            state_, action_, reward_, next_state_ = env_model.sample()

            target = reward_ + GAMMA * \
                     np.max(q_table[next_state_[0], next_state_[1], :])

            q_table[state_[0], state_[1], action_] += step_size * \
                                                      (target - q_table[state_[0], state_[1], action_])

        update_epsilon_greedy_policy(env, state, q_table, policy, episode)
        state = next_state
        rewards += reward

    return steps, rewards


def maze_dyna_q(env):
    planning_repeats = [0, 3, 30]
    performance_steps = np.zeros((len(planning_repeats), MAX_EPISODES))

    for run in range(TOTAL_RUNS):
        print("RUN: {0}".format(run))
        for i, planning_repeat in enumerate(planning_repeats):
            # 행동 가치 저장
            q_table = generate_initial_q_value(env)

            # Dyna-Q를 위한 환경 모델 생성
            env_model = EnvModel()

            policy = generate_initial_random_policy(env)

            for episode in range(MAX_EPISODES):
                steps_, _ = dyna_q(
                    q_table, policy, env_model, env, episode, planning_repeat
                )
                performance_steps[i, episode] += steps_

    # 총 수행 횟수에 대한 평균 값 산출
    performance_steps /= TOTAL_RUNS

    linestyles = ['-', '--', ':']
    for i in range(len(planning_repeats)):
        plt.plot(
            performance_steps[i, :],
            linestyle=linestyles[i],
            label='Number of planning steps: {0}'.format(planning_repeats[i])
        )

    plt.xlabel('Episode')
    plt.ylabel('Executed steps per episode')
    plt.legend()

    plt.show()
    plt.close()


def maze_dyna_q_main():
    env = Maze()  # 미로 환경 객체 구성
    env.reset()
    print(env)

    maze_dyna_q(env)


if __name__ == '__main__':
    maze_dyna_q_main()