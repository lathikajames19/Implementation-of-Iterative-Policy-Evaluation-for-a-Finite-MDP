# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:
```math

$$
\pi(a|s) = \frac{1}{4}
$$

```
This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:
```math
$$
V^\pi(s)
$$
```

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:


$$
V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s,a)
\left[ R(s,a,s') + \gamma V^\pi(s') \right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python

import gymnasium as gym
import numpy as np

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# Access the unwrapped environment to use the transition model
env = env.unwrapped

# Number of states and actions
n_states = env.observation_space.n
n_actions = env.action_space.n

# Parameters
gamma = 0.9
theta = 1e-8

# Random policy: each action has equal probability
policy = np.ones((n_states, n_actions)) / n_actions

# Initialize value function
V = np.zeros(n_states)

# -------------------------------------------------
# Policy Evaluation Function
# -------------------------------------------------

def policy_evaluation(env, policy, gamma=0.9, theta=1e-8):
    """
    Performs iterative policy evaluation using the Bellman expectation equation.

    Parameters:
        env    : Gymnasium FrozenLake environment
        policy : Fixed policy to be evaluated
        gamma  : Discount factor
        theta  : Convergence threshold

    Returns:
        V         : Estimated state-value function
        iteration : Number of iterations used for convergence
    """
    V = np.zeros(env.observation_space.n)
    iteration = 0

    while True:
        delta = 0

        for state in range(env.observation_space.n):
            v = V[state]
            new_value = 0

            for action in range(env.action_space.n):
                action_value = 0

                for probability, next_state, reward, done in env.P[state][action]:
                    action_value += probability * (reward + gamma * V[next_state] * (not done))

                new_value += policy[state][action] * action_value

            V[state] = new_value
            delta = max(delta, abs(v - V[state]))

        iteration += 1

        if delta < theta:
            break

    return V, iteration

# Run policy evaluation
V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name:Lathika K")
print("Register Number: 212224230140")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("Name:Lathika K")
print("Register Number: 212224230140")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()
```

---

## Output

Gamma=0.9

<img width="577" height="347" alt="Screenshot 2026-10-28 142630" src="https://github.com/user-attachments/assets/19d15f16-6958-4aa1-9d44-77c4fe2caadd" />

Gamma=0.99

<img width="708" height="332" alt="Screenshot 2026-10-28 142242" src="https://github.com/user-attachments/assets/46b8e73c-efdd-40ae-89c9-32dd6ee0d6bf" />

---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference
The experiment shows that gamma value affects future rewards. Gamma = 0.99 gives more importance to future rewards than gamma = 0.9. Thus, the gamma value influences the state values in policy evaluation.
