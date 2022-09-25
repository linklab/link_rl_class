# https://www.gymlibrary.dev/environments/toy_text/taxi/
# pip install gym[all]
import gym
import random
import time
import math
import numpy as np

env = gym.make('Taxi-v3', render_mode="human")

ACTION_STRING_LIST = ["  SOUTH", "  NORTH", "   EAST", "   WEST", " PICKUP", "DROPOFF"]

NEGATIVE_BIG_NUMBER = -1.0 * 1_000_000_000

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


class Dummy_Agent:
    def get_action(self, observation):
        available_action_ids = [0, 1, 2, 3, 4, 5]
        action_id = random.choice(available_action_ids)
        return action_id


def run_env():
    print("START RUN!!!")
    agent = Dummy_Agent()
    observation, info = env.reset()

    done = truncated = False
    episode_step = 1
    while not done and not truncated:
        action = agent.get_action(observation)
        next_observation, reward, done, truncated, info = env.step(action)

        print("[Step: {0:3}] Obs.: {1:>3}, Action: {2}({3}), Next Obs.: {4:>3}, Reward: {5:>3}, Done: {6}, Truncated: {7}, "
              "Info: {8}".format(
            episode_step, observation, action, ACTION_STRING_LIST[action], next_observation, reward, done, truncated, info
        ))
        observation = next_observation
        episode_step += 1
        time.sleep(1)


if __name__ == "__main__":
    run_env()