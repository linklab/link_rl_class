# 선수 에이전트: Q-Learning 에이전트, 후수 에이전트: Q-Learning 에이전트
from online_learning.part_2.practice_5.a_tic_tac_toe_env import TicTacToe
from online_learning.part_2.practice_6.a_q_learning_agent import Q_Learning_Agent
from online_learning.part_2.practice_6.f_q_learning_vs_dummy import MAX_EPISODES, LAST_SCHEDULED_EPISODES, \
    INITIAL_EPSILON, FINAL_EPSILON, BOARD_RENDER, STEP_VERBOSE
from online_learning.part_2.practice_6.e_tic_tac_toe_utils import GameStatus, epsilon_scheduled, print_step_status, \
    print_game_statistics, draw_performance


def q_learning_for_self_play():
    game_status = GameStatus()

    env = TicTacToe()

    self_agent_1 = Q_Learning_Agent(name="AGENT_1", env=env)
    self_agent_2 = Q_Learning_Agent(name="AGENT_2", env=env)

    self_agent_2.q_table = self_agent_1.q_table
    self_agent_2.policy = self_agent_1.policy

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        epsilon = epsilon_scheduled(
            episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON
        )

        if BOARD_RENDER:
            env.render()

        done = False
        STATE_2, ACTION_2 = None, None

        agent_1_episode_td_error = 0.0
        agent_2_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # self_agent_1 스텝 수행
            action = self_agent_1.get_action(state)
            next_state, reward, done, info = env.step(action)
            print_step_status(
                self_agent_1, state, action, next_state,
                reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            if done:
                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                print_game_statistics(
                    info, episode, epsilon, total_steps,
                    game_status, self_agent_1, self_agent_2
                )

                # reward: self_agent_1가 착수하여 done=True
                # agent_1이 이기면 1.0, 비기면 0.0
                agent_1_episode_td_error += self_agent_1.q_learning(
                    state, action, None, reward, done, epsilon
                )

                # 미루워 두었던 self_agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += self_agent_2.q_learning(
                        STATE_2, ACTION_2, None, -1.0 * reward, done, epsilon
                    )
            else:
                # 미루워 두었던 self_agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += self_agent_2.q_learning(
                        STATE_2, ACTION_2, next_state, reward, done, epsilon
                    )

                # self_agent_1이 방문한 현재 상태 및 수행한
                # 행동 정보를 저장해 두었다가 추후 활용
                STATE_1 = state
                ACTION_1 = action

                # self_agent_2 스텝 수행
                state = next_state
                action = self_agent_2.get_action(state)
                next_state, reward, done, info = env.step(action)
                print_step_status(
                    self_agent_2, state, action, next_state,
                    reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
                )

                if done:
                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    print_game_statistics(
                        info, episode, epsilon, total_steps,
                        game_status, self_agent_1, self_agent_2
                    )

                    # reward: self_agent_2가 착수하여 done=True
                    # self_agent_2가 이기면 -1.0, 비기면 0.0
                    agent_2_episode_td_error += self_agent_2.q_learning(
                        state, action, None, -1.0 * reward, done, epsilon
                    )

                    # 미루워 두었던 self_agent_1의 배치에 transition 정보 추가
                    agent_1_episode_td_error += self_agent_1.q_learning(
                        STATE_1, ACTION_1, None, reward, done, epsilon
                    )
                else:
                    # self_agent_2이 방문한 현재 상태 및 수행한
                    # 행동 정보를 저장해 두었다가 추후 활용
                    STATE_2 = state
                    ACTION_2 = action

                    # 미루워 두었던 self_agent_1의 배치에 transition 정보 추가
                    agent_1_episode_td_error += self_agent_1.q_learning(
                        STATE_1, ACTION_1, next_state, reward, done, epsilon
                    )

            state = next_state

        game_status.set_agent_1_episode_td_error(agent_1_episode_td_error)
        game_status.set_agent_2_episode_td_error(agent_2_episode_td_error)

    game_status.agent_1_count_state_updates = self_agent_1.count_state_updates
    game_status.agent_2_count_state_updates = self_agent_2.count_state_updates
    draw_performance(game_status, MAX_EPISODES)

    # 훈련 종료 직후 완전 탐욕적으로 정책 설정
    # self_agent_1과 self_agent_2는 동일한 에이전트이므로 self_agent_1에 대해서만 정책 설정
    self_agent_1.make_greedy_policy()

    # self_agent_1과 self_agent_2는 동일한 에이전트이므로 self_agent_1만 반환
    return self_agent_1


if __name__ == '__main__':
    trained_self_agent = q_learning_for_self_play()