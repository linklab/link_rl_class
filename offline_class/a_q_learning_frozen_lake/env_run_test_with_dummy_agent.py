# https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
# pip install gym[all]
import gym
import random
import time

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")
ACTION_STRING_LIST = [" LEFT", " DOWN", "RIGHT", "   UP"]

class Dummy_Agent:
    def get_action(self, observation):
        available_action_ids = [0, 1, 2, 3]
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

        print("[Step: {0:3}] Obs.: {1:>2}, Action: {2}({3}), Next Obs.: {4}, Reward: {5}, Done: {6}, Truncated: {7}, "
              "Info: {8}".format(
            episode_step, observation, action, ACTION_STRING_LIST[action], next_observation, reward, done, truncated, info
        ))
        observation = next_observation
        episode_step += 1
        time.sleep(2)


if __name__ == "__main__":
    run_env()

