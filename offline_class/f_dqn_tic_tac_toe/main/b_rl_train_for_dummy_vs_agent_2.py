# 선수 에이전트: Dummy 에이전트, 후수 에이전트: RL 에이전트
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from offline_class.f_dqn_tic_tac_toe.agents.a_dummy_agent import Dummy_Agent
from offline_class.f_dqn_tic_tac_toe.common.c_game_stats import (
    draw_performance, print_game_statistics, epsilon_scheduled, GameStatus
)
from offline_class.f_dqn_tic_tac_toe.common.a_env_tic_tac_toe_333 import TicTacToe333
from offline_class.f_dqn_tic_tac_toe.common.a_env_tic_tac_toe_343 import TicTacToe343
from offline_class.f_dqn_tic_tac_toe.common.d_utils import PLAY_TYPE, EarlyStopModelSaver
import copy

# from offline_class.f_dqn_tic_tac_toe.agents.c_dqn_agent import TTTAgentDqn
from offline_class.f_dqn_tic_tac_toe.agents.c_dqn_agent_solution import TTTAgentDqn

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
LAST_SCHEDULED_PERCENT = 0.75

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 100_000

STEP_VERBOSE = False
BOARD_RENDER = False

# GAME = "333"
GAME = "343"


def learning_for_dummy_vs_agent_2():
    game_status = GameStatus()

    if GAME == "333":
        env = TicTacToe333()
        n_cells = 9
        print("Tic-Tac-Toe-333")
    elif GAME == "343":
        env = TicTacToe343()
        n_cells = 12
        print("Tic-Tac-Toe-343")
    else:
        raise ValueError()

    agent_1 = Dummy_Agent(name="AGENT_1", env=env)
    agent_2 = TTTAgentDqn(
        name="AGENT_2", env=env, n_cells=n_cells, gamma=0.99, learning_rate=0.001,
        replay_buffer_size=100_000, batch_size=32, target_sync_step_interval=1_000,
        min_buffer_size_for_training=1_000
    )

    total_steps = 0

    early_stop_model_saver = EarlyStopModelSaver(target_win_percent=99.0)
    win_percent = 0.0

    last_scheduled_episodes = MAX_EPISODES * LAST_SCHEDULED_PERCENT

    early_stop = False

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        epsilon = epsilon_scheduled(
            episode, last_scheduled_episodes, INITIAL_EPSILON, FINAL_EPSILON
        )

        if BOARD_RENDER:
            env.render()

        done = False
        STATE_2, ACTION_2 = None, None

        agent_2_episode_td_error = 0.0

        while not done:
            total_steps += 1
            if isinstance(agent_2, TTTAgentDqn):
                agent_2.time_steps += 1

            # agent_1 (Dummy Agent) 스텝 수행
            action_1 = agent_1.get_action(state)
            next_state, reward, done, info = env.step(action_1)

            if done:
                # 게임 완료 및 게임 승패 관련 통계 정보 획득
                win_percent = print_game_statistics(
                    info, episode, epsilon, total_steps, game_status, PLAY_TYPE.SECOND
                )

                # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                # reward: agent_1이 착수하여 done=True
                # agent_1이 이기면 1.0, 비기면 0.0
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error = agent_2.learning(
                        STATE_2, ACTION_2, next_state, -1.0 * reward, done
                    )
            else:
                # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error = agent_2.learning(
                        STATE_2, ACTION_2, next_state, reward, done
                    )

                # agent_2 스텝 수행
                state = next_state
                action = agent_2.get_action(state)
                next_state, reward, done, info = env.step(action)

                if done:
                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    win_percent = print_game_statistics(
                        info, episode, epsilon, total_steps, game_status, PLAY_TYPE.SECOND
                    )

                    # reward: agent_2가 착수하여 done=True
                    # agent_2가 이기면 -1.0, 비기면 0.0
                    agent_2_episode_td_error = agent_2.learning(
                        state, action, next_state, -1.0 * reward, done
                    )
                else:
                    # agent_2가 방문한 현재 상태 및 수행한 행동 정보를 저장해 두었다가 추후 활용
                    STATE_2 = copy.deepcopy(state)
                    ACTION_2 = copy.deepcopy(action)

            state = next_state

        game_status.set_agent_2_episode_td_error(agent_2_episode_td_error)

        if episode > 5000:
            early_stop = early_stop_model_saver.check(
                agent_2.agent_type, PLAY_TYPE.SECOND, win_percent, agent_2_episode_td_error, agent_2.q_model
            )
            if early_stop:
                break

    if not early_stop:
        early_stop_model_saver.save_checkpoint(
            agent_type=agent_2.agent_type, play_type=PLAY_TYPE.SECOND, win_percent=win_percent,
            loss=agent_2_episode_td_error,
            q_model=agent_2.q_model
        )

    draw_performance(game_status, episode)


if __name__ == '__main__':
    learning_for_dummy_vs_agent_2()
