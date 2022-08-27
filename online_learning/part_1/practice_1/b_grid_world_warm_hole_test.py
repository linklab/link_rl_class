# -------------------------------
# |(0,0)|(0,1)|(0,2)|(0,3)|(0,4)|
# |(1,0)|(1,1)|(1,2)|(1,3)|(1,4)|
# |(2,0)|(2,1)|(2,2)|(2,3)|(2,4)|
# |(3,0)|(3,1)|(3,2)|(3,3)|(3,4)|
# |(4,0)|(4,1)|(4,2)|(4,3)|(4,4)|
# -------------------------------
import time

from online_learning.env.a_grid_word import GridWorld


def main_warm_hole():
    A_POSITION = (0, 1)  # 임의로 지정한 특별한 상태 A 좌표
    B_POSITION = (0, 3)  # 임의로 지정한 특별한 상태 B 좌표

    A_PRIME_POSITION = (4, 1)  # 상태 A에서 임의의 행동시 도착할 위치 좌표
    B_PRIME_POSITION = (2, 3)  # 상태 B에서 임의의 행동시 도착할 위치 좌표

    env = GridWorld(
        warm_hole_states=[
            (A_POSITION, A_PRIME_POSITION, 10.0),
            (B_POSITION, B_PRIME_POSITION, 5.0)
        ]
    )

    env.reset()
    print("reset")
    env.render()

    done = False
    total_steps = 0
    while not done:
        total_steps += 1
        action = env.get_random_action()
        next_state, reward, done, _ = env.step(action)
        print("action: {0}, reward: {1}, done: {2}, total_steps: {3}".format(
            env.ACTION_SYMBOLS[action],
            reward, done, total_steps
        ))
        env.render()
        time.sleep(1)


if __name__ == "__main__":
    main_warm_hole()