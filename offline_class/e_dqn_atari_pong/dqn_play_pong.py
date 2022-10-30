# https://www.gymlibrary.dev/environments/classic_control/cart_pole/
# -*- coding: utf-8 -*-
import sys
import gym
import torch
import os
from gym.wrappers import FrameStack, AtariPreprocessing

from offline_class.e_dqn_atari_pong.qnet_pong import AtariCNNQnet

ENV_NAME = 'ALE/Pong-v5'

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

MODEL_DIR = os.path.join(PROJECT_HOME, "offline_class", "e_dqn_atari_pong", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play(env, q, num_episodes):
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()

        episode_steps = 0

        done = truncated = False

        while not done and not truncated:
            episode_steps += 1
            action = q.get_action(observation, epsilon=0.0)

            # action을 통해서 next_state, reward, done, info를 받아온다
            next_observation, reward, done, truncated, _ = env.step(action)

            episode_reward += reward  # episode_reward 를 산출하는 방법은 감가률 고려하지 않는 이 라인이 더 올바름.
            observation = next_observation

            if done:
                break

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))


def main_q_play(num_episodes):
    env = gym.make(ENV_NAME, render_mode="human")

    env = gym.make(
        ENV_NAME, mode=0, difficulty=0,
        obs_type="grayscale",
        frameskip=4,
        repeat_action_probability=0.0,
        full_action_space=False,
        render_mode="human"
    )
    env = FrameStack(AtariPreprocessing(env, frame_skip=1, screen_size=84, scale_obs=True), num_stack=4)

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    q = AtariCNNQnet(obs_shape=obs_shape, n_actions=n_actions, device=DEVICE).to(DEVICE)
    model_params = torch.load(os.path.join(MODEL_DIR, "dqn_Pong-v5_latest.pth"))
    q.load_state_dict(model_params)

    play(env, q, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 3
    main_q_play(num_episodes=NUM_EPISODES)
