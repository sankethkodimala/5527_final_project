import os
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# This script reads TensorBoard logs from training runs (like PPO and DQN)
# and plots a comparison of their performance over time. Can be extended to include more algorithms or metrics as needed.

def get_data(log_dir, tag='rollout/ep_rew_mean'): # default tag for ppo reward in stable-baselines3
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    if tag not in event_acc.Tags()['scalars']:
        return None, None

    events = event_acc.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    return steps, values

def plot_comparison(log_base='tensorboard_logs', output_file='performance_comparison.png'):
    plt.figure(figsize=(10, 6))

    found_any = False

    for subdir in sorted(os.listdir(log_base)):
        path = os.path.join(log_base, subdir)
        if not os.path.isdir(path):
            continue

        # Determine algorithm
        label = subdir
        color = None
        if subdir.lower().startswith('ppo'):
            color = 'blue'
        elif subdir.lower().startswith('dqn'):
            color = 'red'

        steps, values = get_data(path)

        if steps:
            plt.plot(steps, values, label=label, color=color, alpha=0.8)
            found_any = True
            print("Loaded", len(steps), "data points from", subdir)
        else:
            print("No reward data found in", subdir)

    if not found_any:
        print("No training logs with reward data found.")
        return

    plt.title('PPO vs DQN Performance Comparison')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Comparison plot saved to {output_file}")

    # Print summary statistics
    print("\n" + "="*50)
    print(f"{'Run':<25} | {'Max Rew':<10} | {'Final Rew':<10}")
    print("-" * 50)

    for subdir in sorted(os.listdir(log_base)):
        path = os.path.join(log_base, subdir)

        if not os.path.isdir(path):
            continue

        steps, values = get_data(path)
        if values:
            max_rew = max(values)
            final_rew = values[-1]
            print(f"{subdir:<25} | {max_rew:<10.2f} | {final_rew:<10.2f}") # '<' aligns fields to 25, 10, and 10 chars

    print("="*50)
    print()

    # plt.show()

if __name__ == "__main__":
    if not os.path.exists('tensorboard_logs'):
        print("Error: tensorboard_logs directory not found. Run training first to generate logs.")
    else:
        plot_comparison()
