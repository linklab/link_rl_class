# https://www.gymlibrary.dev/environments/box2d/lunar_lander/
# pip install gym[all]
# pip3 install box2d box2d-kengz box2d-py
import gym
import time
import numpy as np

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

print("gym.__version__:", gym.__version__)

env = gym.make('LunarLander-v2', render_mode="human")

ACTION_STRING_LIST = ["DO NOTHING", " FIRE LEFT", " FIRE MAIN", "FIRE RIGHT"]


def env_info_details():
    #####################
    # observation space #
    #####################
    print("*" * 80)
    print("[observation_space]")
    print(env.observation_space)
    # We should expect to see 15 possible grids from 0 to 15 when
    # we uniformly randomly sample from our observation space
    for i in range(10):
        print(env.observation_space.sample())
    print()

    print("*" * 80)
    ################
    # action space #
    ################
    print("[action_space]")
    print(env.action_space)
    print(env.action_space.n)
    # We should expect to see 4 actions when
    # we uniformly randomly sample:
    #     1. LEFT: 0
    #     2. RIGHT: 1
    for i in range(10):
        print(env.action_space.sample(), end=" ")
    print()

    print("*" * 80)
    # This sets the initial state at S, our starting point
    # We can render the environment to see where we are on the 4x4 frozenlake gridworld
    observation, info = env.reset()

    action = 1  # RIGHT
    next_observation, reward, done, truncated, info = env.step(action)

    # Prob = 1: deterministic policy, if we choose to go right, we'll go right
    print("Obs.: {0}, Action: {1}({2}), Next Obs.: {3}, Reward: {4}, Done: {5}, Truncated: {6}, Info: {7}".format(
        observation, action, ACTION_STRING_LIST[action], next_observation, reward, done, truncated, info
    ))

    observation = next_observation

    time.sleep(3)

    action = 1  # RIGHT
    next_observation, reward, done, truncated, info = env.step(action)

    print("Obs.: {0}, Action: {1}({2}), Next Obs.: {3}, Reward: {4}, Done: {5}, Truncated: {6}, Info: {7}".format(
        observation, action, ACTION_STRING_LIST[action], next_observation, reward, done, truncated, info
    ))

    print("*" * 80)
    time.sleep(3)

if __name__ == "__main__":
    env_info_details()