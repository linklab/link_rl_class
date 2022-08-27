from practice_2.practice_2_6.practice_2_6_env import TicTacToe
from practice_2.practice_2_6.practice_2_6_qlearning_self import q_learning_for_self_play
from practice_2.practice_2_6.practice_2_6_utils import print_game_statistics, GameStatus

MAX_EPISODES = 10000
VERBOSE = False


def self_play(self_agent):
    env = TicTacToe()

    agent_1 = self_agent
    agent_2 = self_agent

    agent_2.q_table = agent_1.q_table
    agent_2.policy = agent_1.policy

    current_agent = agent_1

    game_status = GameStatus()
    total_steps = 0

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()

        if VERBOSE:
            print("[시작 상태]")
            env.render()

        done = False
        while not done:
            total_steps += 1
            action = current_agent.get_action(state)

            next_state, _, done, info = env.step(action)

            if VERBOSE:
                print("[{0}]".format("Q-Learning 에이전트 1" if current_agent == agent_1 else "Q-Learning 에이전트 2"))
                env.render()

            if done:
                if VERBOSE:
                    if info['winner'] == 1:
                        print("Q-Learning 에이전트 1이 이겼습니다.")
                    elif info['winner'] == -1:
                        print("Q-Learning 에이전트 2가 이겼습니다!")
                    else:
                        print("비겼습니다!")

                done = done
                print_game_statistics(info, episode, 0.0, total_steps, game_status)
            else:
                state = next_state

            if current_agent == agent_1:
                current_agent = agent_2
            else:
                current_agent = agent_1


if __name__ == '__main__':
    trained_self_agent = q_learning_for_self_play()
    self_play(trained_self_agent)
