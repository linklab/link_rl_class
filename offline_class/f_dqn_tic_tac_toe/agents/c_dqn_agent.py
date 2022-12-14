import random
import torch
from torch import optim
import numpy as np
import torch.nn.functional as F

from offline_class.f_dqn_tic_tac_toe.common.b_models_and_buffer import QNet, ReplayBuffer, Transition
from offline_class.f_dqn_tic_tac_toe.common.d_utils import AGENT_TYPE


class TTTAgentDqn:
    def __init__(
            self, name, env, gamma=0.99, learning_rate=0.001,
            replay_buffer_size=10_000, batch_size=32, target_sync_step_interval=500,
            min_buffer_size_for_training=100
    ):
        self.name = name
        self.env = env
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_sync_step_interval = target_sync_step_interval
        self.replay_buffer_size = replay_buffer_size
        self.min_buffer_size_for_training = min_buffer_size_for_training
        self.agent_type = AGENT_TYPE.DQN.value

        # network
        self.q_model = QNet(n_features=12, n_actions=12)
        self.target_q = QNet(n_features=12, n_actions=12)
        self.target_q.load_state_dict(self.q_model.state_dict())
        self.optimizer = optim.Adam(self.q_model.parameters(), lr=self.learning_rate)

        # agent
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.episode_reward_lst = []

        self.time_steps = 0
        self.training_time_steps = 0

    def get_action(self, state, epsilon=0.0, mode="TRAIN"):
        available_actions = state.get_available_actions()
        unavailable_actions = list(set(self.env.ALL_ACTIONS) - set(available_actions))
        obs = state.data.flatten()
        action = None
        # dqn 모델에 환경 입력 후 예측 액션 값 도출
        action_prob = self.q_model.forward(obs)

        # TODO

        return action

    def learning(self, state, action, next_state, reward, done):
        loss = 0.0
        self.replay_buffer.append(
            Transition(state.data.flatten(), action, next_state.data.flatten(), reward, done)
        )
        if len(self.replay_buffer) < self.min_buffer_size_for_training:
            return loss

        batch = self.replay_buffer.sample(self.batch_size)

        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32, 1])
        observations, actions, next_observations, rewards, dones = batch

        # TODO

        self.training_time_steps += 1

        return loss.item()

