from doom_env import DoomEnv

env = DoomEnv(env_id="VizdoomBasic-v1", render=True, discrete_actions=None)

print(env.env.action_space)  # should be Discrete(...)

obs, info = env.reset()
total_reward = 0

while True:
    action = env.sample_action()   # integer, not a list/array
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print("Env Setup Complete")
print("Total Reward:", total_reward)
env.close()