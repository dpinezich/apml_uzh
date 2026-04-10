# Chapter 11 — Basic RL Algorithms

**Session:** 4 | **Chapter:** 2 of 3 | **Duration:** 50 min  
**Format:** Slides + Examples + Exercises

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Implement the Q-Learning algorithm from scratch
- Explain the Bellman equation intuitively
- Use the ε-greedy strategy for exploration
- Train a Q-Learning agent on FrozenLake (Gymnasium)
- Visualize the Q-table and training progress
- Understand the difference between model-free and model-based RL

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | Model-Free vs Model-Based RL | 5 min |
| 2 | Q-Learning: The Algorithm | 15 min |
| 3 | SARSA: On-Policy Alternative | 5 min |
| 4 | Policy Gradient (conceptual) | 5 min |
| 5 | Live Example: FrozenLake | 10 min |
| 6 | **Exercises** | **10 min** |
| **Total** | | **50 min** |

> **Exercise time: 10 minutes**

---

## Content Outline

### Block 1 — Model-Free vs Model-Based RL (5 min)

**Model-Based RL:**
- Agent learns a model of the environment (transition probabilities P, reward R)
- Plans ahead using this model
- Sample efficient, but hard to learn an accurate model
- Example: AlphaGo (learned to model Go positions)

**Model-Free RL:**
- Agent does NOT learn a model of the environment
- Learns directly from experience (trial and error)
- More practical for complex environments
- Two sub-types:
  - Value-based: learn Q(s, a) → derive policy (Q-Learning, DQN)
  - Policy-based: learn π(s) directly (Policy Gradient, PPO)

**This chapter:** Model-free, value-based → Q-Learning

---

### Block 2 — Q-Learning: The Algorithm (15 min)

**Goal:** Learn the optimal Q-table — Q(s, a) for all state-action pairs.

**The Bellman Equation (the heart of Q-Learning):**
```
Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') - Q(s, a)]
```

**Breaking it down:**
- `Q(s, a)`: current estimate of taking action a in state s
- `r`: reward received after taking action a in state s
- `γ · max Q(s', a')`: discounted best future value from new state s'
- `α` (alpha): learning rate — how much we update toward the new estimate
- `[r + γ · max Q(s', a') - Q(s, a)]`: the "TD error" (Temporal Difference error)

**Intuition:** "The estimate of Q should move toward the observed reward plus the best we can do from the next state."

**The Q-Learning Algorithm:**
```
Initialize Q(s, a) = 0 for all s, a

For each episode:
  Reset environment → get initial state s
  
  Repeat:
    Choose action a using ε-greedy:
      if random < ε: a = random action  (explore)
      else:          a = argmax_a Q(s, a)  (exploit)
    
    Take action a → get reward r and next state s'
    
    Update Q:
      Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') - Q(s, a)]
    
    s ← s'
    
  Until episode ends (goal reached or timeout)

Decay ε after each episode
```

**Convergence:** Q-Learning is guaranteed to converge to the optimal Q* (given enough exploration and proper learning rate)

---

### Block 3 — SARSA: On-Policy Alternative (5 min)

**SARSA = State, Action, Reward, State, Action**

Like Q-Learning, but the update uses the *actual* next action taken, not the best possible one:

```
Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') - Q(s, a)]
```

**Key difference:**
- **Q-Learning** is **off-policy**: learns the optimal Q* regardless of the policy being followed (greedy update)
- **SARSA** is **on-policy**: learns Q for the policy being currently followed (including exploration)

**When SARSA is better:** In risky environments — SARSA learns to avoid risky actions even during exploration, while Q-Learning can learn an optimal but dangerous policy.

---

### Block 4 — Policy Gradient (Conceptual, 5 min)

**A completely different approach:** Instead of learning Q-values, directly learn the policy π(a|s).

**Idea:** Parametrize the policy (e.g., with a neural network). Adjust parameters to make better actions more likely.

**REINFORCE algorithm (vanilla policy gradient):**
```
Collect full episode trajectory
Compute cumulative reward G at each timestep
Update parameters: θ ← θ + α · G · ∇log π(a|s; θ)
```

**Modern algorithms:** PPO (Proximal Policy Optimization), A3C, SAC — all extensions of policy gradients combined with deep networks.

**When to use:** Continuous action spaces (robot joints), where Q-tables are impossible.

---

### Block 5 — Live Example: FrozenLake (10 min)

→ See `02-examples/ch11_rl_algorithms_examples.ipynb`

FrozenLake environment:
- 4×4 grid, frozen lake
- States: 16 cells (0–15)
- Actions: 4 (left, down, right, up)
- Goal: reach cell 15 without falling into holes (cells 5, 7, 11, 12)
- Reward: +1 at goal, 0 otherwise

Demonstrate:
1. Environment setup with Gymnasium
2. Q-table initialization
3. Training loop with ε-greedy
4. Visualizing Q-table evolution
5. Success rate over episodes

---

### Block 6 — Exercises (10 min)

→ See `03-exercises/ch11_rl_algorithms_exercises.ipynb`

Students implement Q-Learning step by step from scratch on FrozenLake.

---

## Instructor Notes

- The Bellman equation: go through it slowly — each term has a clear meaning
- TD error analogy: "prediction error correction" — like adjusting a GPS estimate as you drive
- Q-table on FrozenLake: show it as a heatmap — students can see which actions are valued
- SARSA vs Q-Learning: this difference matters in practice — use the cliff example
- Policy gradient: keep it conceptual — this is just to show the landscape of RL

---

## Materials

- Slides: `01-slides/ch11_slides.md`
- Examples: `02-examples/ch11_rl_algorithms_examples.ipynb`
- Exercises: `03-exercises/ch11_rl_algorithms_exercises.ipynb`
- Solutions: `04-solutions/ch11_rl_algorithms_solutions.ipynb`
