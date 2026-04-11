## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Action Space

### Discrete
This takes an input of (n) Which is a single integer [0..2] where:

0 is move left
1 is move right
2 is shoot

Files for this are located at
`doom_env` and `test_doom_env.py`

### MultiBinary
This takes an input vector [x, y, z] where:
[LEFT, RIGHT, SHOOT]
Examples:
```
[0, 0, 0]  # do nothing
[1, 0, 0]  # move left
[0, 1, 0]  # move right
[0, 0, 1]  # shoot
[1, 0, 1]  # move left + shoot
[0, 1, 1]  # move right + shoot
```

Files for this are located at
`multibinary-actions/doom_multibinary.py` and `multibinary-actions/test_doom_multibinary.py`

### Box (don't use this)

If we want to train hard, we can do this:
Box(low, high, shape)

Where it takes contiuous values, really hard to train


## Training
According to ChatGPT for training:
I’d train in two phases:

Get a baseline working with VizdoomBasic-v1 + PPO
Only after that works, switch to a richer action space or harder scenario

That’s the best path because ViZDoom recommends the maintained Gymnasium wrappers, the default BASIC env exists as a standard Gymnasium env, and Stable-Baselines3’s PPO supports Discrete, MultiDiscrete, and MultiBinary action spaces.

Examples:

Training Logic
```
import gymnasium as gym
from vizdoom import gymnasium_wrapper
from stable_baselines3 import PPO

env = gym.make("VizdoomBasic-v1")

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_vizdoom_basic")
env.close()
```

HyperParams
```
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)
```