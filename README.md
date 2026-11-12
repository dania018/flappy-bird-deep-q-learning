# Flappy Bird AI with Deep Q-Learning (DQN)

A **Deep Reinforcement Learning** agent, built in PyTorch, that learns to play Flappy Bird by trial and error. Nobody tells it how to play: it only gets rewards for staying alive and passing pipes.

![Agent playing Flappy Bird](assets/demo.gif)

> **Environment:** [`flappy-bird-gymnasium`](https://github.com/markub3327/flappy-bird-gymnasium) (`FlappyBird-v0`) · **Framework:** PyTorch · **Algorithm:** Deep Q-Network (DQN)

## Table of Contents

- [How Reinforcement Learning Works Here](#how-reinforcement-learning-works-here)
- [From Q-Learning to Deep Q-Networks](#from-q-learning-to-deep-q-networks)
- [The Two Problems with a Naive DQN](#the-two-problems-with-a-naive-dqn)
- [Model Architecture](#model-architecture)
- [Training Loop](#training-loop)
- [Hyperparameters](#hyperparameters)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Results](#results)
- [What I Learned](#what-i-learned)
- [Next Steps](#next-steps)

---

## How Reinforcement Learning Works Here

| RL concept | In Flappy Bird |
|---|---|
| **Agent** | The bird, controlled by the DQN |
| **Environment** | The game world: pipes, gravity, ground |
| **State** | **180 LIDAR readings**: distances measured by rays cast from the bird to pipes and ground (the red lines in the demo). The environment can also give a 12-feature state with `use_lidar=False` |
| **Actions** | `0` = do nothing, `1` = flap |
| **Reward** | +0.1 per frame alive · +1.0 per pipe passed · −1.0 for dying · −0.5 for touching the top of the screen |
| **Episode** | One game, from start until the bird crashes (or hits the step cap) |

The agent's goal is to learn a **policy**, a rule for choosing the action that maximises total future reward in every state.

## From Q-Learning to Deep Q-Networks

### Q-Learning and its limit

Classic Q-learning stores a **Q-table**: one value `Q(s, a)` for every state–action pair, estimating how good it is to take action `a` in state `s`. The table is updated with:

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

This works for small, discrete problems like a 48-state grid world. It breaks down when:

1. **There are millions of states.** The table gets too big to store or fill.
2. **States are continuous.** Flappy Bird's positions and velocities are real numbers, so a table can't list every one.

### The DQN idea

Instead of looking values up in a table, **a neural network approximates the Q-function**:

```
Q-learning:  Q(s, a)        → table lookup
DQN:         Q(s, a; θ)     → neural network with parameters θ
```

The network takes a **state** as input and outputs one **Q-value per action**. The agent picks the action with the highest Q-value, for example `Q(flap) = 0.8 > Q(do nothing) = 0.2` means it flaps.

## The Two Problems with a Naive DQN

Training a neural network directly on the agent's experience is unstable. A DQN adds two fixes.

### 1. Correlated samples → Experience Replay

**Problem:** Neural networks learn best from *independent* samples, but game experiences arrive in *sequential, highly correlated* order (`E1 → E2 → E3 → …`). Learning from them in order makes training unstable.

**Fix: Experience Replay.** Each experience is stored in a memory buffer:

```
experience = (state, action, reward, next_state, terminated)
```

During training, the agent **randomly samples a mini-batch** from this buffer (e.g. `E9 → E121 → E2 → E5`), which breaks the correlation. The buffer is a **FIFO deque**: when it's full, the oldest experiences drop out as new ones arrive.

### 2. Moving target → Target Network

**Problem:** The DQN learns by reducing the gap between a **prediction** and a **target**:

```
prediction  ŷ = Q(s, a; θ)
target      y = r + γ · max_a' Q(s', a'; θ)
loss          = (y − ŷ)²
```

Both sides use the *same* network θ, so every update also moves the target. It's like chasing a moving goalpost, and it makes training unstable.

**Fix: Target Network.** Use two networks with the same architecture:

| Network | Role |
|---|---|
| **Policy network** (online) | Chooses actions and is trained every step |
| **Target network** | An older copy of the policy network, used only to compute the target `y` |

The target network is frozen and **synced with the policy network every `network_sync_rate` steps**, which gives the policy network a stable target to learn towards.

## Model Architecture

```
State (180) → Linear(180, 256) → ReLU → Linear(256, 2) → Q-values [Q(do nothing), Q(flap)]
```

| Layer | Nodes | Activation |
|---|---|---|
| Input | 180 (LIDAR readings) | — |
| Hidden | 256 | ReLU |
| Output | 2 (one Q-value per action) | Linear |

The output layer is **linear** because Q-values are unbounded estimates of future reward, not probabilities.

## Training Loop

1. **Set up the environment:** `FlappyBird-v0` from `flappy-bird-gymnasium`
2. **Define the DQN:** a policy network plus a target network that starts as an exact copy
3. **Create the experience replay** buffer (`deque`, max 100,000 experiences)
4. **Run episodes until you stop it (`Ctrl+C`).** Each episode is capped at `reward_threshold` = 1000 steps. At each step:
   - **Choose an action with an ε-greedy policy:** with probability ε pick a random action (*explore*), otherwise pick the action with the highest Q-value (*exploit*)
   - Take the action, then store `(state, action, next_state, reward, terminated)` in memory
   - **Decay ε every step:** `ε ← max(ε × 0.99995, 0.05)`. The agent starts fully random (ε = 1) and gradually relies on what it has learned
5. **After each episode:**
   - If it's the best reward so far, **save the model** to `runs/flappybirdv0.pt` and write it to `runs/flappybirdv0.log`
   - **Sample a mini-batch of 32** from memory and update the policy network with the MSE loss between predicted and target Q-values (Adam optimizer)
   - **Sync the target network** with the policy network once 1000 steps have passed since the last sync
6. **Test and visualise:** load the saved policy network and render the game to watch the agent play

### Device

Training runs on whatever hardware is available:

| Hardware | PyTorch device |
|---|---|
| Apple Silicon (M1/M2/M3) | `mps` |
| NVIDIA GPU | `cuda` |
| Anything else | `cpu` |

## Hyperparameters

All hyperparameters are kept in `parameters.yaml`, so you can experiment without touching the code:

| Parameter | Value | What it controls |
|---|---|---|
| `alpha` | 0.0005 | Learning rate of the Adam optimizer |
| `gamma` | 0.99 | How much future rewards count compared with immediate ones |
| `epsilon_init` | 1.0 | Starting exploration rate (100% random actions) |
| `epsilon_min` | 0.05 | Lowest exploration rate allowed |
| `epsilon_decay` | 0.99995 | How fast exploration decreases (per step) |
| `replay_memory_size` | 100,000 | Maximum experiences stored |
| `mini_batch_size` | 32 | Experiences sampled per training update |
| `reward_threshold` | 1000 | Maximum steps per episode |
| `network_sync_rate` | 1000 | Steps between target network updates |

## Project Structure

```
├── agent.py               # Environment setup, training loop, ε-greedy policy, testing
├── dqn.py                 # DQN neural network (180 → 256 → 2)
├── experience_replay.py   # ReplayMemory (FIFO deque buffer)
├── game-flappy_bird.py    # Flappy Bird game script
├── parameters.yaml        # Hyperparameters
├── runs/                  # Saved best model (.pt) and training log (.log)
└── README.md
```

## Getting Started

```bash
git clone https://github.com/dania018/flappy-bird-dqn.git
cd flappy-bird-dqn

pip install torch gymnasium flappy-bird-gymnasium pyyaml
```

**Train the agent:**

```bash
python agent.py flappybirdv0 --train
```

**Watch the trained agent play:**

```bash
python agent.py flappybirdv0
```

Training runs until you press `Ctrl+C`. The best model so far is always saved in `runs/`.

## Results

At the start the bird crashes almost immediately, because it's flapping at random. The reward turning from negative to positive shows it learned to stay alive longer and to start getting through pipes, instead of crashing straight away. Most of the progress came while ε was decaying (episodes 1 to ~1,200). After that, gains were slower.

---

**Dania Tariq**: [LinkedIn](#) · [GitHub](https://github.com/dania018)
