from a_tic_tac_toe_env import TicTacToe
import random
import time


class Dummy_Agent:
    def __init__(self, name, env):
        self.name = name
        self.env = env

    def get_action(self, state):
        available_action_ids = state.get_available_actions()
        action_id = random.choice(available_action_ids)
        return action_id


def main():
    env = TicTacToe()
    state = env.reset()
    env.render()

    agent_1 = Dummy_Agent(name="AGENT_1", env=env)
    agent_2 = Dummy_Agent(name="AGENT_2", env=env)

    current_agent = agent_1

    done = False
    total_steps = 0

    while not done:
        total_steps += 1

        action = current_agent.get_action(state)

        next_state, reward, done, info = env.step(action)

        print("[{0}] action: {1}, reward: {2}, done: {3}, \
                info: {4}, total_steps: {5}".format(
            current_agent.name, action, reward, done, info, total_steps
        ))

        env.render()

        state = next_state
        time.sleep(2)

        if current_agent == agent_1:
            current_agent = agent_2
        else:
            current_agent =agent_1


if __name__ == "__main__":
    main()
