import random
import time

# -------------------------------
# T1 0 1 2 3 4 T2
# -------------------------------


class RandomWalk():
    def __init__(
            self,
            num_internal_states=5,        # 종료 상태를 제외한 내부 상태 개수
            transition_reward=0.0,        # 일반적인 상태 전이 보상
            left_terminal_reward=0.0,     # 왼쪽 종료 상태로 이동하는 행동
                                          # 수행 시 받는 보상
            right_terminal_reward=1.0     # 오른쪽 종료 상태로 이동하는 행동
                                          # 수행 시 받는 보상
    ):
        self.__version__ = "0.0.1"

        self.num_internal_states = num_internal_states

        self.num_states = num_internal_states + 2
        self.STATES = [i for i in range(num_internal_states)]
        self.TERMINAL_STATES = ['T1', 'T2']

        # 모든 가능한 행동
        self.ACTION_LEFT = 0
        self.ACTION_RIGHT = 1
        self.ACTION_SYMBOLS = ["\u2190", "\u2192"]

        # 종료 상태를 제외한 임의의 상태에서 왼쪽 이동 또는 오른쪽 이동
        self.ACTIONS = [
            self.ACTION_LEFT,
            self.ACTION_RIGHT
        ]
        self.NUM_ACTIONS = len(self.ACTIONS)

        # 시작 상태 위치
        self.START_STATE = self.STATES[int(num_internal_states / 2)]

        self.transition_reward = transition_reward

        self.left_terminal_reward = left_terminal_reward

        self.right_terminal_reward = right_terminal_reward

        self.current_state = None

    def reset(self):
        self.current_state = self.START_STATE
        return self.current_state

    def moveto(self, state):
        self.current_state = state

    def get_next_state(self, state, action):
        if state in self.TERMINAL_STATES:
            next_state = state
        else:
            if action == self.ACTION_LEFT:
                if state == 0:
                    next_state = 'T1'
                else:
                    next_state = state - 1
            elif action == self.ACTION_RIGHT:
                if state == self.num_internal_states - 1:
                    next_state = 'T2'
                else:
                    next_state = state + 1
            else:
                raise ValueError()

        return next_state

    def get_reward(self, state, next_state):
        if next_state == 'T1':
            reward = self.left_terminal_reward
        elif next_state == 'T2':
            reward = self.right_terminal_reward
        else:
            reward = self.transition_reward

        return reward

    def get_state_action_probability(self, state, action):
        next_state = self.get_next_state(state, action)

        reward = self.get_reward(state, next_state)
        prob = 1.0

        return next_state, reward, prob

    # take @action in @state
    # @return: (reward, new state)
    def step(self, action):
        next_state = self.get_next_state(
            state=self.current_state, action=action
        )

        reward = self.get_reward(self.current_state, next_state)

        self.current_state = next_state

        if self.current_state in self.TERMINAL_STATES:
            done = True
        else:
            done = False

        return next_state, reward, done, None

    def render(self, mode='human'):
        print(self.__str__(), end="\n\n")

    def get_random_action(self):
        return random.choice(self.ACTIONS)

    def __str__(self):
        randomwalk_str = ""
        randomwalk_str += " T1 " + " ".join(
            ["{0}".format(i) for i in range(self.num_internal_states)]
        ) + " T2\n"

        if self.current_state in self.STATES:
            blank = "    " + "  " * self.current_state
        elif self.current_state == 'T1':
            blank = " "
        elif self.current_state == 'T2':
            blank = "  " + "  " * (self.num_internal_states + 1)
        else:
            raise ValueError()

        randomwalk_str += blank + "*"

        return randomwalk_str