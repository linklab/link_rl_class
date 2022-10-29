# https://www.gymlibrary.dev/environments/classic_control/cart_pole/
# pip install gym[all]
import gym
import random
import time

from gym.wrappers import AtariPreprocessing, FrameStack

ENV_NAME = 'ALE/Pong-v5'

env = gym.make(
    ENV_NAME, mode=0, difficulty=0,
    obs_type="grayscale",
    frameskip=4,
    repeat_action_probability=0.0,
    full_action_space=False
)

env = FrameStack(AtariPreprocessing(env, frame_skip=1, screen_size=84, scale_obs=True), num_stack=4)

ACTION_STRING_LIST = ['     NOOP', '     FIRE', '    RIGHT', '     LEFT', 'RIGHTFIRE', ' LEFTFIRE']


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

        print("[Step: {0:3}] Obs.: {1:>2}, Action: {2}({3}), Next Obs.: {4}, Reward: {5}, Done: {6}, Truncated: {7}, "
              "Info: {8}".format(
            episode_step,
            str(observation.shape), action, ACTION_STRING_LIST[action], str(next_observation.shape), reward, done, truncated, info
        ))
        observation = next_observation
        episode_step += 1
        time.sleep(0.01)


if __name__ == "__main__":
    run_env()

