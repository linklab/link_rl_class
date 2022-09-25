# https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
# pip install gym[all]
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
import random

print("gym.__version__", gym.__version__)

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


ACTION_STRING_LIST = [" LEFT", " DOWN", "RIGHT", "   UP"]
IS_SLIPPERY = False
MAP_NAME = "4x4"
DESC = None

# MAP_NAME = "8x8"
# DESC = [
#     "SFFFFFFF",
#     "FFFHHFFF",
#     "FFFHFFFF",
#     "FFFFFHFF",
#     "FFFHFFFF",
#     "FHHFFFHF",
#     "FHFFHFHF",
#     "FFFHFFFG",
# ]


def greedy_action(action_values):
    max_value = np.max(action_values)
    return np.random.choice(
        [action_ for action_, value_ in enumerate(action_values) if value_ == max_value]
    )


def epsilon_greedy_action(action_values, epsilon):
    if np.random.rand() < epsilon:
        return random.choice(range(len(action_values)))
    else:
        max_value = np.max(action_values)
        return np.random.choice(
            [action_ for action_, value_ in enumerate(action_values) if value_ == max_value]
        )


def q_learning(num_episodes=500, num_test_episodes=7, alpha=0.1, gamma=0.95, epsilon=0.1):
    env = gym.make('FrozenLake-v1', desc=DESC, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)

    # Q-Table 초기화
    q_table = np.zeros(
        [env.observation_space.n, env.action_space.n]
    )

    episode_reward_list = []

    training_time_steps = 0
    last_episode_reward = 0

    is_train_success = False

    for episode in range(1, num_episodes + 1):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()
        print("[EPISODE: {0:>2}]".format(episode, observation), end=" ")
        sList = [observation]

        episode_step = 1
        done = truncated = False

        # The Q-Table 알고리즘
        while not done and not truncated:
            episode_step += 1
            action = epsilon_greedy_action(q_table[observation, :], epsilon)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, truncated, _ = env.step(action)

            episode_reward += reward

            # Q-Learning
            td_error = reward + gamma * np.max(q_table[next_observation, :]) - q_table[observation, action]
            q_table[observation, action] = q_table[observation, action] + alpha * td_error

            training_time_steps += 1  # Q-table 업데이트 횟수
            episode_reward_list.append(last_episode_reward)

            sList.append(next_observation)
            observation = next_observation

        print(
            "Episode Steps: {0:>2}, Visited States: {1}, Episode Reward: {2}".format(episode_step, sList, episode_reward),
            "GOAL" if done and observation == 15 else ""
        )
        last_episode_reward = episode_reward

        if episode % 10 == 0:
            episode_reward_list_test, avg_episode_reward_test = q_learning_testing(
                num_test_episodes=num_test_episodes, q_table=q_table
            )
            print("[TEST RESULTS: {0} Episodes, Episode Reward List: {1}] Episode Reward Mean: {2:.3f}".format(
                num_test_episodes, episode_reward_list_test, avg_episode_reward_test
            ))
            if avg_episode_reward_test == 1.0:
                print("***** TRAINING DONE!!! *****")
                is_train_success = True
                break

    return q_table, training_time_steps, episode_reward_list, is_train_success


def q_learning_testing(num_test_episodes, q_table):
    episode_reward_list = []

    test_env = gym.make('FrozenLake-v1', desc=DESC, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)

    for episode in range(num_test_episodes):
        episode_reward = 0  # cumulative_reward
        episode_step = 0

        observation, _ = test_env.reset()

        done = False
        while not done:
            episode_step += 1
            action = greedy_action(q_table[observation, :])
            next_observation, reward, done, truncated, _ = test_env.step(action)
            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            observation = next_observation

        episode_reward_list.append(episode_reward)

    return episode_reward_list, np.mean(episode_reward_list)


def q_learning_playing(q_table):
    test_env = gym.make('FrozenLake-v1', desc=DESC, map_name=MAP_NAME, is_slippery=IS_SLIPPERY, render_mode="human")
    observation, _ = test_env.reset()
    time.sleep(1)
    done = False
    while not done:
        action = greedy_action(q_table[observation, :])
        next_observation, reward, done, truncated, _ = test_env.step(action)
        observation = next_observation
        time.sleep(1)


def main_q_table_learning():
    NUM_EPISODES = 200
    NUM_TEST_EPISODES = 7
    ALPHA = 0.1
    GAMMA = 0.95
    EPSILON = 0.1

    q_table, training_time_steps, episode_reward_list, is_train_success = q_learning(
        NUM_EPISODES, NUM_TEST_EPISODES, ALPHA, GAMMA, EPSILON
    )
    print("\nFinal Q-Table Values")
    print("    LEFT   DOWN  RIGHT     UP")
    for idx, observation in enumerate(q_table):
        print("{0:2d}".format(idx), end=":")
        for action_state in observation:
            print("{0:5.3f} ".format(action_state), end=" ")
        print()

    plt.plot(range(training_time_steps), episode_reward_list, color="Blue")
    plt.xlabel("training steps")
    plt.ylabel("episode reward")
    plt.show()

    if is_train_success:
        q_learning_playing(q_table=q_table)


if __name__ == "__main__":
    main_q_table_learning()