#
import gym
import matplotlib.pyplot as plt

ENV_NAME = "MsPacman-v4"

env = gym.make(ENV_NAME, render_mode="rgb_array")
env.reset()
plt.imshow(env.render())
plt.show()
