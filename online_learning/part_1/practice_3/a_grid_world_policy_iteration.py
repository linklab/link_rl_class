import numpy as np

from online_learning.common.a_grid_word import GridWorld
from online_learning.common.e_util import softmax, draw_grid_world_state_values_image, draw_grid_world_action_values_image, \
	draw_grid_world_optimal_policy_image

GRID_HEIGHT = 4
GRID_WIDTH = 4
TERMINAL_STATES = [(0, 0), (GRID_HEIGHT - 1, GRID_WIDTH - 1)]

DISCOUNT_RATE = 0.9

THETA_1 = 0.0001
THETA_2 = 0.0001


# 정책 반복 클래스
class PolicyIteration:
	def __init__(self, env):
		self.env = env
		self.state_values = None
		self.policy = np.empty([GRID_HEIGHT, GRID_WIDTH, self.env.NUM_ACTIONS])

		# 모든 상태에서 수행 가능한 행동에 맞춰 임의의 정책을 생성함
		# 초기에 각 행동의 선택 확률은 모두 같음
		for i in range(GRID_HEIGHT):
			for j in range(GRID_WIDTH):
				for action in self.env.ACTIONS:
					if (i, j) in TERMINAL_STATES:
						self.policy[i][j][action] = 0.00
					else:
						self.policy[i][j][action] = 0.25

	# 정책 평가 함수
	def policy_evaluation(self):
		# 임시 상태-가치 저장 변수 초기화
		state_values = np.zeros((GRID_HEIGHT, GRID_WIDTH))

		# 가치 함수의 값들이 수렴할 때까지 반복
		iter_num = 0
		while True:
			# 현재 상태 가치 저장
			old_state_values = state_values.copy()

			for i in range(GRID_HEIGHT):
				for j in range(GRID_WIDTH):
					if (i, j) in TERMINAL_STATES:
						state_values[i][j] = 0.0
					else:
						values = []
						for action in self.env.ACTIONS:
							(next_i, next_j), reward, transition_prob = self.env.get_state_action_probability(
								state=(i, j), action=action
							)

							# Bellman-Equation, 벨만 방정식 적용
							values.append(
								self.policy[i, j, action] * transition_prob * (reward + DISCOUNT_RATE * old_state_values[next_i, next_j])
							)

						state_values[i][j] = np.sum(values)

			iter_num += 1

			# 갱신되는 값이 THETA_1(=0.0001)을 기준으로 수렴하는지 판정
			delta_value = np.sum(np.absolute(old_state_values - state_values))
			if delta_value < THETA_1:
				break

		self.state_values = state_values

		return iter_num

	# 정책 개선 함수
	def policy_improvement(self):
		new_policy = np.empty([GRID_HEIGHT, GRID_WIDTH, self.env.NUM_ACTIONS])

		is_policy_stable = True

		# 행동-가치 함수 생성
		for i in range(GRID_HEIGHT):
			for j in range(GRID_WIDTH):
				if (i, j) in TERMINAL_STATES:
					for action in self.env.ACTIONS:
						new_policy[i][j][action] = 0.00
				else:
					q_func = []
					for action in self.env.ACTIONS:
						(next_i, next_j), reward, transition_prob = self.env.get_state_action_probability(
							state=(i, j), action=action
						)
						q_func.append(
							transition_prob * (reward + DISCOUNT_RATE * self.state_values[next_i, next_j])
						)

					new_policy[i, j, :] = softmax(q_func)

		error = np.sum(np.absolute(self.policy - new_policy))

		if error > THETA_2:
			is_policy_stable = False

		self.policy = new_policy

		return is_policy_stable, error

	# 정책 반복 함수
	def start_iteration(self):
		iter_num = 0

		# 정책의 안정성 검증
		is_policy_stable = False

		print("[[[ 정책 반복 시작! ]]]")

		while not is_policy_stable:
			iter_num_policy_evaluation = self.policy_evaluation()
			print("*** 정책 평가 [수렴까지 반복 횟수: {0}] ***".format(
				iter_num_policy_evaluation
			))

			is_policy_stable, error = self.policy_improvement()
			print("*** 정책 개선 [에러 값: {0:7.5f}] ***".format(error))

			iter_num += 1
			print("*** 정책의 안정(Stable) 여부: {0}, 반복 횟수: {1} ***\n".format(
				is_policy_stable,
				iter_num
			))
		print("[[[ 정책 반복 종료! ]]]\n\n")

		return self.state_values, self.policy

	def calculate_optimal_policy(self):
		optimal_policy = dict()
		for i in range(GRID_HEIGHT):
			for j in range(GRID_WIDTH):
				indices = [
					idx for idx, value_ in enumerate(self.policy[i, j, :]) if value_ == np.max(self.policy[i, j, :])
				]
				optimal_policy[(i, j)] = indices

		return optimal_policy

	def calculate_grid_world_optimal_action_values(self):
		action_value_function = np.zeros((GRID_HEIGHT, GRID_WIDTH, self.env.NUM_ACTIONS))
		for i in range(GRID_HEIGHT):
			for j in range(GRID_WIDTH):
				# 주어진 상태에서 가능한 모든 행동들의 결과로
				# 다음 상태 및 보상 정보 갱신
				for action in self.env.ACTIONS:
					(next_i, next_j), reward, transition_prob = self.env.get_state_action_probability(
						state=(i, j), action=action
					)

					action_value_function[i, j, action] = transition_prob * (reward + DISCOUNT_RATE * self.state_values[next_i, next_j])

		return action_value_function


def main():
	# 그리드 월드 환경 객체 생성
	env = GridWorld(
        height=GRID_HEIGHT,
        width=GRID_WIDTH,
        start_state=(0, 0),
        terminal_states=TERMINAL_STATES,
        transition_reward=-1.0,
        terminal_reward=-1.0,
        outward_reward=-1.0
    )
	env.reset()

	PI = PolicyIteration(env)
	PI.start_iteration()

	print(PI.state_values)

	draw_grid_world_state_values_image(
		PI.state_values, GRID_HEIGHT, GRID_WIDTH
	)

	draw_grid_world_action_values_image(
		PI.calculate_grid_world_optimal_action_values(),
		GRID_HEIGHT, GRID_WIDTH, env.NUM_ACTIONS, env.ACTION_SYMBOLS
	)

	draw_grid_world_optimal_policy_image(
		PI.calculate_optimal_policy(),
		GRID_HEIGHT, GRID_WIDTH, env.ACTION_SYMBOLS
	)


if __name__ == "__main__":
	main()
