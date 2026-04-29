import numpy as np
import matplotlib.pyplot as plt
from corridor.doom_env_cooridor import DoomEnv

def verify():
    # 1. init environment with preprocessing
    # Using 'VizdoomBasic-v1' as a default
    env = DoomEnv(
        env_id="VizdoomBasic-v1",
        render=False,
        preprocess=True,
        resize_shape=(84, 84),
        grayscale=True,
        frame_stack=4
    )

    print("Checking Observation Space:")
    print(f"Shape: {env.observation_space.shape}")
    print(f"Low: {env.observation_space.low.min()}")
    print(f"High: {env.observation_space.high.max()}")

    # 2. Reset and take a few steps
    obs, _ = env.reset()
    
    # Take 5 random steps to get some variation
    for _ in range(5):
        action = env.sample_action()
        obs, _, _, _, _ = env.step(action)
    
    # 3. Verify observation properties
    # obs should be (4, 84, 84) because of FrameStack(4) and Resize(84, 84)
    print("\nVerifying Observation:")
    print(f"Obs shape: {obs.shape}")
    print(f"Obs dtype: {obs.dtype}")
    print(f"Value range: [{np.min(obs):.2f}, {np.max(obs):.2f}]")

    # 4. Save the most recent frame in the stack for visual inspection
    latest_frame = np.array(obs)[-1] 
    plt.imshow(latest_frame, cmap='gray')
    plt.title("Preprocessed Frame (84x84, Grayscale)")
    plt.colorbar()
    plt.savefig("preprocessed_preview.png")
    print("\nPreview saved as 'preprocessed_preview.png'")

    env.close()

if __name__ == "__main__":
    try:
        verify()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nCould not run verification: {e}")
