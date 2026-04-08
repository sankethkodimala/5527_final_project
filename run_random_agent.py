import gymnasium as gym
from vizdoom import gymnasium_wrapper 

env = gym.make("VizdoomBasic-v1", render_mode="human")
obs, info = env.reset()
done = False
truncated = False
total_reward = 0

while not (done or truncated):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print("Episode finished.")
print("Total reward:", total_reward)

env.close()