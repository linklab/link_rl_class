from practice_2.practice_2_6.practice_2_6_dummy_agent import Dummy_Agent
from practice_2.practice_2_6.practice_2_6_env import TicTacToe
from practice_2.practice_2_6.practice_2_6_qlearning_agent import Q_Learning_Agent
from practice_2.practice_2_6.practice_2_6_utils import GameStatus, epsilon_scheduled, print_step_status, \
    print_game_statistics, draw_performance

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
LAST_SCHEDULED_EPISODES = 50000

# 최대 반복 에피소드(게임) 횟수
MAX_EPISODES = 100000

STEP_VERBOSE = False
BOARD_RENDER = False


# 선수 에이전트: Q-Learning 에이전트, 후수 에이전트: Dummy 에이전트
def q_learning_for_agent_1_vs_dummy():
    game_status = GameStatus()
    env = TicTacToe()

    agent_1 = Q_Learning_Agent(name="AGENT_1", env=env)
    agent_2 = Dummy_Agent(name="AGENT_2", env=env)

    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        epsilon = epsilon_scheduled(
            episode, LAST_SCHEDULED_EPISODES, INITIAL_EPSILON, FINAL_EPSILON
        )

        if BOARD_RENDER:
            env.render()

        done = False

        agent_1_episode_td_error = 0.0
        while not done:
            total_steps += 1

            # agent_1 스텝 수행
            action = agent_1.get_action(state)
            next_state, reward, done, info = env.step(action)
            print_step_status(
                agent_1, state, action, next_state,
                reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
            )

            if done:
                # reward: agent_1이 착수하여 done=True
                # agent_1이 이기면 1.0, 비기면 0.0
                agent_1_episode_td_error += agent_1.q_learning(
                    state, action, None, reward, done, epsilon
                )

                # 게임 완료 및 게임 승패 관련 통계 정보 출력
                print_game_statistics(
                    info, episode, epsilon, total_steps,
                    game_status, agent_1, agent_2
                )
            else:
                # agent_2 스텝 수행
                action_2 = agent_2.get_action(next_state)
                next_state, reward, done, info = env.step(action_2)
                print_step_status(
                    agent_2, state, action_2, next_state,
                    reward, done, info, env, STEP_VERBOSE, BOARD_RENDER
                )

                if done:
                    # reward: agent_2가 착수하여 done=True
                    # agent_2가 이기면 -1.0, 비기면 0.0
                    agent_1_episode_td_error += agent_1.q_learning(
                        state, action, None, reward, done, epsilon
                    )

                    # 게임 완료 및 게임 승패 관련 통계 정보 출력
                    print_game_statistics(
                        info, episode, epsilon, total_steps,
                        game_status, agent_1, agent_2
                    )
                else:
                    agent_1_episode_td_error += agent_1.q_learning(
                        state, action, next_state, reward, done, epsilon
                    )

            state = next_state

        game_status.set_agent_1_episode_td_error(agent_1_episode_td_error)

    game_status.agent_1_count_state_updates = agent_1.count_state_updates
    draw_performance(game_status, MAX_EPISODES)

    # 훈련 종료 직후 완전 탐욕적으로 정책 설정
    agent_1.make_greedy_policy()

    return agent_1


if __name__ == '__main__':
    trained_agent_1 = q_learning_for_agent_1_vs_dummy()