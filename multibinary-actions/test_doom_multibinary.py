from doom_multibinary_env import DoomEnv

discrete_actions = [
    [0, 0, 0],  # no-op
    [1, 0, 0],  # left
    [0, 1, 0],  # right
    [0, 0, 1],  # shoot
    [1, 0, 1], # left + shoot
    [0, 1, 1]  # right + shoot
]

env = DoomEnv(env_id="VizdoomBasic-MultiBinary-v1", render=True, discrete_actions=discrete_actions)
obs, info = env.reset()
total_reward = 0

while True:
    action = env.sample_action() 
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print("Env Setup Complete")
print("Total Reward:", total_reward)
env.close()