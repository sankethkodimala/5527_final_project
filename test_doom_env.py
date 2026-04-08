from doom_env import DoomEnv

env = DoomEnv(env_id="VizdoomBasic-v1", render=True)

obs, info = env.reset()
done = False
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