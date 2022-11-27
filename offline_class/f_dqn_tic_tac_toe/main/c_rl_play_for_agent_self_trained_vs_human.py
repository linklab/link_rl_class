# 선수 에이전트: RL 에이전트, 후수 에이전트: RL 에이전트
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from offline_class.f_dqn_tic_tac_toe.agents.b_human_agent import Human_Agent
from offline_class.f_dqn_tic_tac_toe.common.a_env_tic_tac_toe_333 import TicTacToe333
from offline_class.f_dqn_tic_tac_toe.common.a_env_tic_tac_toe_343 import TicTacToe343
from offline_class.f_dqn_tic_tac_toe.common.d_utils import model_load

# from offline_class.f_dqn_tic_tac_toe.agents.c_dqn_agent import TTTAgentDqn
from offline_class.f_dqn_tic_tac_toe.agents.c_dqn_agent_solution import TTTAgentDqn

GAME = "333"
# GAME = "343"


def self_play(env, agent_1):
    env.print_board_idx()
    state = env.reset()

    agent_2 = Human_Agent(name="AGENT_2", env=env)

    current_agent = agent_1

    print()

    print("[RL 에이전트 차례]")
    env.render()

    done = False
    while not done:
        action = current_agent.get_action(state, mode="PLAY")
        next_state, _, done, info = env.step(action)
        if current_agent == agent_2:
            print("     State:", state)
            print("    Action:", action)
            print("Next State:", next_state, end="\n\n")

        print("[{0}]".format(
            "Q-Learning 에이전트 차례" if current_agent == agent_1 \
            else "당신(사람) 차례"
        ))
        env.render()

        if done:
            if info['winner'] == -1:
                print("당신(사람)이 이겼습니다. 놀랍습니다!")
            elif info['winner'] == 1:
                print("Q-Learning 에이전트가 이겼습니다.")
            else:
                print("비겼습니다. 잘했습니다!")
        else:
            state = next_state

        current_agent = agent_2 if current_agent == agent_1 else agent_1


if __name__ == '__main__':
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

    agent = TTTAgentDqn(name="SELF_AGENT", env=env, n_cells=n_cells)

    model_file_name = "DQN_SELF_20.0.pth"
    model_load(agent.model, file_name=model_file_name)
    self_play(env, agent)
