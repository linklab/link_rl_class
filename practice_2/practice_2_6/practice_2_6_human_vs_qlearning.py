from practice_2.practice_2_6.practice_2_6_dummy_vs_qlearning import q_learning_for_dummy_vs_agent_2
from practice_2.practice_2_6.practice_2_6_env import TicTacToe
from practice_2.practice_2_6.practice_2_6_human_agent import Human_Agent


def play_with_agent_2(agent_2):
    env = TicTacToe()
    env.print_board_idx()
    state = env.reset()

    agent_1 = Human_Agent(name="AGENT_1", env=env)
    current_agent = agent_1

    print()

    print("[당신(사람) 차례]")
    env.render()

    done = False
    while not done:
        action = current_agent.get_action(state)
        next_state, _, done, info = env.step(action)
        if current_agent == agent_2:
            print("     State:", state)
            print("   Q-value:", current_agent.get_q_values_for_one_state(state))
            print("    Policy:", current_agent.get_policy_for_one_state(state))
            print("    Action:", action)
            print("Next State:", next_state, end="\n\n")

        print("[{0}]".format(
            "Q-Learning 에이전트 차례" if current_agent == agent_1 \
            else "당신(사람) 차례"
        ))
        env.render()

        if done:
            if info['winner'] == 1:
                print("당신(사람)이 이겼습니다. 놀랍습니다!")
            elif info['winner'] == -1:
                print("Q-Learning 에이전트가 이겼습니다.")
            else:
                print("비겼습니다. 잘했습니다!")
        else:
            state = next_state

        if current_agent == agent_1:
            current_agent = agent_2
        else:
            current_agent = agent_1


if __name__ == '__main__':
    trained_agent_2 = q_learning_for_dummy_vs_agent_2()
    play_with_agent_2(trained_agent_2)