# 선수 에이전트: Dummy 에이전트, 후수 에이전트: Q-Learning 에이전트
from practice_2.practice_2_6.practice_2_6_dummy_agent import Dummy_Agent
from practice_2.practice_2_6.practice_2_6_env import TicTacToe
from practice_2.practice_2_6.practice_2_6_qlearning_agent import Q_Learning_Agent
from practice_2.practice_2_6.practice_2_6_qlearning_vs_dummy import MAX_EPISODES, LAST_SCHEDULED_EPISODES, \
    INITIAL_EPSILON, FINAL_EPSILON, BOARD_RENDER, STEP_VERBOSE
from practice_2.practice_2_6.practice_2_6_utils import GameStatus, epsilon_scheduled, print_step_status, \
    print_game_statistics, draw_performance


def q_learning_for_dummy_vs_agent_2():
    game_status = GameStatus()

    env = TicTacToe()

    agent_1 = Dummy_Agent(name="AGENT_1", env=env)
    agent_2 = Q_Learning_Agent(name="AGENT_2", env=env)

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        current_agent = agent_1

        epsilon = epsilon_scheduled(
            episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON
        )

        if BOARD_RENDER:
            env.render()

        done = False
        STATE_2, ACTION_2 = None, None

        agent_2_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # agent_1 스텝 수행
            action_1 = agent_1.get_action(state)
            next_state, reward, done, info = env.step(action_1)
            print_step_status(
                current_agent, state, action_1, next_state,
                reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            if done:
                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                print_game_statistics(
                    info, episode, epsilon, total_steps,
                    game_status, agent_1, agent_2
                )

                # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                # reward: agent_1이 착수하여 done=True
                # agent_1이 이기면 1.0, 비기면 0.0
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += agent_2.q_learning(
                        STATE_2, ACTION_2, None, -1.0 * reward, done, epsilon
                    )
            else:
                # 미루워 두었던 agent_2의 배치에 transition 정보 추가
                if STATE_2 is not None and ACTION_2 is not None:
                    agent_2_episode_td_error += agent_2.q_learning(
                        STATE_2, ACTION_2, next_state, reward, done, epsilon
                    )

                # agent_2 스텝 수행
                state = next_state
                action = agent_2.get_action(state)
                next_state, reward, done, info = env.step(action)
                print_step_status(
                    agent_2, state, action, next_state,
                    reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
                )

                if done:
                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    print_game_statistics(
                        info, episode, epsilon, total_steps,
                        game_status, agent_1, agent_2
                    )

                    # reward: agent_2가 착수하여 done=True
                    # agent_2가 이기면 -1.0, 비기면 0.0
                    agent_2_episode_td_error += agent_2.q_learning(
                        state, action, None, -1.0 * reward, done, epsilon
                    )
                else:
                    # agent_2에 방문한 현재 상태 및 수행한 행동 정보를
                    # 저장해 두었다가 추후 활용
                    STATE_2 = state
                    ACTION_2 = action

            state = next_state

        game_status.set_agent_2_episode_td_error(agent_2_episode_td_error)

    game_status.agent_2_count_state_updates = agent_2.count_state_updates
    draw_performance(game_status, MAX_EPISODES)

    # 훈련 종료 직후 완전 탐욕적으로 정책 설정
    agent_2.make_greedy_policy()

    return agent_2


if __name__ == '__main__':
    trained_agent_2 = q_learning_for_dummy_vs_agent_2()