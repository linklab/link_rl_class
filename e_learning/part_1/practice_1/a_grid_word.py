import random

# -------------------------------
# |(0,0)|(0,1)|(0,2)|(0,3)|(0,4)|
# |(1,0)|(1,1)|(1,2)|(1,3)|(1,4)|
# |(2,0)|(2,1)|(2,2)|(2,3)|(2,4)|
# |(3,0)|(3,1)|(3,2)|(3,3)|(3,4)|
# |(4,0)|(4,1)|(4,2)|(4,3)|(4,4)|
# -------------------------------


class GridWorld():
	def __init__(
			self,
			height=5, width=5,  # 격자판의 크기
			start_state=(0, 0),  # 시작 상태
			terminal_states=[(4, 4)],  # 종료 상태
			transition_reward=0.0,  # 일반적인 상태 전이 보상
			terminal_reward=1.0,  # 종료 상태로 이동하는 행동 수행
			# 때 받는 보상
			outward_reward=0.0,  # 미로 바깥으로 이동하는 행동 수행
			# 때 받는 보상
			warm_hole_states=None  # 윔홀 정의
	):
		self.__version__ = "0.0.1"

		# 그리드월드의 세로 길이
		self.HEIGHT = height

		# 그리드월드의 가로 길이
		self.WIDTH = width

		self.STATES = []
		self.num_states = self.WIDTH * self.HEIGHT

		for i in range(self.HEIGHT):
			for j in range(self.WIDTH):
				self.STATES.append((i, j))

		# 터미널 상태를 상태 집합에서 제거
		for state in terminal_states:
			self.STATES.remove(state)

		self.current_state = None

		# 모든 가능한 행동
		self.ACTION_UP = 0
		self.ACTION_DOWN = 1
		self.ACTION_LEFT = 2
		self.ACTION_RIGHT = 3

		self.ACTIONS = [
			self.ACTION_UP,
			self.ACTION_DOWN,
			self.ACTION_LEFT,
			self.ACTION_RIGHT
		]

		# UP, DOWN, LEFT, RIGHT
		self.ACTION_SYMBOLS = ["↑", "↓", "←", "→"]

		self.NUM_ACTIONS = len(self.ACTIONS)

		# 시작 상태 위치
		self.START_STATE = start_state

		# 종료 상태 위치
		self.TERMINAL_STATES = terminal_states

		# 웜홀 상태 위치
		self.WARM_HOLE_STATES = warm_hole_states

		# 일반 상태 전이 수행에 대한 보상 값
		self.transition_reward = transition_reward

		# 종료 상태 전이 수행에 대한 보상 값
		self.terminal_reward = terminal_reward

		# 그리드월드 바깥으로 상태 전이 수행해 대한 보상 값
		self.outward_reward = outward_reward

	# 환경 초기화 작업: 에이전트의 현재 위치를 START_STATE로 설정
	def reset(self):
		self.current_state = self.START_STATE
		return self.current_state

	# 에이전트의 현재 위치를 state로 설정
	def moveto(self, state):
		self.current_state = state

	# 주어진 state 상태가 웜홀 상태이면 True 반환하고, 아니면 False 반환
	def is_warm_hole_state(self, state):
		i, j = state

		if self.WARM_HOLE_STATES is not None \
				and len(self.WARM_HOLE_STATES) > 0:
			for warm_hole_info in self.WARM_HOLE_STATES:
				warm_hole_state = warm_hole_info[0]
				if i == warm_hole_state[0] and j == warm_hole_state[1]:
					return True
		return False

	# 주어진 웜홀 상태 state에 대하여 미리 정해져 있는 다음 상태 반환
	def get_next_state_warm_hole(self, state):
		i, j = state
		next_state = None

		for warm_hole_info in self.WARM_HOLE_STATES:
			warm_hole_state = warm_hole_info[0]
			warm_hole_prime_state = warm_hole_info[1]

			if i == warm_hole_state[0] and j == warm_hole_state[1]:
				next_state = warm_hole_prime_state
				break
		return next_state

	# 주어진 웜홀 상태 state에 대하여 미리 정해져 있는 보상 반환
	def get_reward_warm_hole(self, state):
		i, j = state
		reward = None

		for warm_hole_info in self.WARM_HOLE_STATES:
			warm_hole_state = warm_hole_info[0]
			warm_hole_reward = warm_hole_info[2]

			if i == warm_hole_state[0] and j == warm_hole_state[1]:
				reward = warm_hole_reward
				break

		return reward

	# 주어진 상태 state와 행동 action에 대한 다음 상태 반환
	def get_next_state(self, state, action):
		i, j = state

		# 주어진 상태가 웜홀 상태이면 미리 정해진 다음 상태를 반환
		if self.is_warm_hole_state(state):
			next_state = self.get_next_state_warm_hole(state)
			next_i = next_state[0]
			next_j = next_state[1]
		# 주어진 상태가 종료 상태이면 현재 상태를
		# 다음 상태로 설정하여 반환
		elif (i, j) in self.TERMINAL_STATES:
			next_i = i
			next_j = j
		# 주어진 행동 수행에 따른 다음 상태 반환
		# 주어진 행동에 의하여 그리드월드 밖으로 이동하는 경우
		# 제자리에 멈춤
		else:
			if action == self.ACTION_UP:
				next_i = max(i - 1, 0)
				next_j = j
			elif action == self.ACTION_DOWN:
				next_i = min(i + 1, self.HEIGHT - 1)
				next_j = j
			elif action == self.ACTION_LEFT:
				next_i = i
				next_j = max(j - 1, 0)
			elif action == self.ACTION_RIGHT:
				next_i = i
				next_j = min(j + 1, self.WIDTH - 1)
			else:
				raise ValueError()

		return next_i, next_j

	# 주어진 상태 state에서 다음 상태 next_state로 이동할 경우
	# 얻는 보상 반환
	def get_reward(self, state, next_state):
		i, j = state
		next_i, next_j = next_state

		# 주어진 상태가 웜홀 상태이면 미리 정해진 보상 반환
		if self.is_warm_hole_state(state):
			reward = self.get_reward_warm_hole(state)
		else:
			# 다음 상태가 종료 상태이면 미리 정해진 보상 반환
			if (next_i, next_j) in self.TERMINAL_STATES:
				reward = self.terminal_reward
			else:
				# 주어진 행동에 의하여 그리드월드 밖으로 이동하는 경우
				# 제자리에 멈추면서 미리 정해진 보상 반환
				if i == next_i and j == next_j:
					reward = self.outward_reward
				# 일반적인 상태 전이인 경우 미리 정해진
				# 일반 전이 보상 반환
				else:
					reward = self.transition_reward

		return reward

	# 주어진 상태 state에서 행동 action을 수행할 때 전이되는
	# 다음 상태 및 보상과 이에 대한 전이 확률을 반환
	def get_state_action_probability(self, state, action):
		next_i, next_j = self.get_next_state(state, action)

		reward = self.get_reward(state, (next_i, next_j))
		transition_prob = 1.0

		return (next_i, next_j), reward, transition_prob

	# 에이전트가 행동 action을 선택하여 환경에 적용할 때 호출하는 함수
	# 행동 action이 수행된 이후 전이된 다음 상태, 보상, 종료 유무 반환
	def step(self, action):
		next_i, next_j = self.get_next_state(
			state=self.current_state, action=action
		)

		reward = self.get_reward(self.current_state, (next_i, next_j))

		self.current_state = (next_i, next_j)

		if self.current_state in self.TERMINAL_STATES:
			done = True
		else:
			done = False

		return (next_i, next_j), reward, done, None

	# 그리드월드 환경을 화면에 출력
	def render(self, mode='human'):
		print(self.__str__())

	# 임의의 행동을 선택하여 반환
	def get_random_action(self):
		return random.choice(self.ACTIONS)

	# 그리드월드 환경을 문자열로 변환
	def __str__(self):
		gridworld_str = ""
		for i in range(self.HEIGHT):
			gridworld_str += "-------------------------------\n"

			for j in range(self.WIDTH):
				if self.current_state[0] == i and self.current_state[1] == j:
					gridworld_str += "|  {0}  ".format("*")
				elif (i, j) == self.START_STATE:
					gridworld_str += "|  {0}  ".format("S")
				elif (i, j) in self.TERMINAL_STATES:
					gridworld_str += "|  {0}  ".format("G")
				elif self.WARM_HOLE_STATES and \
						(i, j) in [state[0] for state in self.WARM_HOLE_STATES]:
					gridworld_str += "|  {0}  ".format("W")
				else:
					gridworld_str += "|     "
			gridworld_str += "|\n"

			for j in range(self.WIDTH):
				gridworld_str += "|({0},{1})".format(i, j)

			gridworld_str += "|\n"

		gridworld_str += "-------------------------------\n"
		return gridworld_str
