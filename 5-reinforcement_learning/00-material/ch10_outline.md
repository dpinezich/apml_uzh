# Chapter 10 — Introduction to Reinforcement Learning

**Session:** 4 | **Chapter:** 1 of 3 | **Duration:** 50 min  
**Format:** Slides + Examples (no formal exercises — conceptual chapter)

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Explain the RL framework: agent, environment, state, action, reward
- Describe the RL loop and how learning happens
- Understand Markov Decision Processes (MDPs) at an intuitive level
- Explain the exploration vs exploitation dilemma
- Distinguish RL from supervised and unsupervised learning

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | What Is Reinforcement Learning? | 10 min |
| 2 | Core Components: Agent, Environment, MDP | 12 min |
| 3 | The Exploration-Exploitation Dilemma | 8 min |
| 4 | Value Functions | 5 min |
| 5 | Applications Gallery | 5 min |
| 6 | Live Example: Grid World | 10 min |
| **Total** | | **50 min** |

> **No formal exercises** — this is a conceptual and demonstration chapter.

---

## Content Outline

### Block 1 — What Is Reinforcement Learning? (10 min)

**The third ML paradigm:**
- Supervised: learn from labeled examples
- Unsupervised: discover structure in unlabeled data
- **Reinforcement Learning:** learn by trial and error, guided by rewards

**The core idea:** An agent takes actions in an environment, receives feedback (rewards), and learns to take better actions over time.

**The ultimate goal:** Maximize total cumulative reward over time.

**Relatable analogies:**
1. **Training a dog:** Give treats for good behavior, withhold for bad. The dog learns which actions lead to treats.
2. **Learning to ride a bike:** Falls are negative rewards, staying balanced is positive. You adapt through experience.
3. **Video games:** Score = reward. The game is the environment. Controls = actions.

**What makes RL special:**
- No supervisor telling you the right action — only reward/penalty feedback
- Actions affect future states (temporal dependency)
- Delayed rewards: sometimes a bad action now leads to a big reward later

---

### Block 2 — Core Components: Agent, Environment, MDP (12 min)

**The RL Loop:**
```
        Action aₜ
Agent ──────────────→ Environment
  ↑                        ↓
  ←── State sₜ₊₁ + Reward rₜ₊₁ ──
```

**Key components:**

**State (s):** Complete description of the environment at a given time
- Example (chess): the positions of all pieces on the board
- Example (robot): sensor readings, joint angles, velocities

**Action (a):** What the agent can do in a given state
- Discrete: move left/right/up/down (grid world), play card (card game)
- Continuous: steer by 3.2°, apply 50N force

**Reward (r):** Scalar feedback signal from the environment
- Positive: reached goal, scored a point, made a profitable trade
- Negative: hit a wall, lost a piece, went bankrupt
- Zero: nothing happened

**Policy (π):** The agent's strategy — maps states to actions
- π(s) = a: given state s, take action a
- The goal of learning IS to find a good policy

**Markov Decision Process (MDP):**
The formal framework for RL. Defined by tuple (S, A, P, R, γ):
- S: set of all possible states
- A: set of all possible actions
- P(s'|s, a): transition probability (what state do we end up in?)
- R(s, a): reward function (what reward do we get?)
- γ (gamma): discount factor — how much we value future rewards (0–1)

**The Markov Property:** "The future depends only on the current state, not on the history."
This simplifies the problem enormously — you only need to know where you are, not how you got there.

**Discount factor γ:**
- γ = 0: greedy, only care about immediate reward
- γ = 1: care equally about all future rewards
- γ = 0.9: typical value — future rewards are worth 90% of immediate ones

---

### Block 3 — The Exploration-Exploitation Dilemma (8 min)

**The fundamental tension in RL:**
- **Exploitation:** Do what you know works best (maximize expected reward)
- **Exploration:** Try new things to discover potentially better strategies

**Analogy: Restaurant choice**
- Exploitation: go to your favorite restaurant
- Exploration: try a new place that might be better (or worse)

**If you only exploit:** You might be stuck with a suboptimal strategy
**If you only explore:** You never actually use what you've learned

**The ε-greedy strategy (most common solution):**
```python
if random() < epsilon:
    action = random_action()     # explore
else:
    action = best_known_action() # exploit
```

**Epsilon decay:** Start with high ε (lots of exploration), slowly reduce it as learning progresses.

**Other strategies:** UCB (Upper Confidence Bound), Thompson Sampling (for bandit problems)

---

### Block 4 — Value Functions (5 min)

Instead of directly learning the policy, RL often learns value functions first.

**State Value V(s):**
- "How good is it to be in state s?"
- Expected total discounted reward starting from state s, following policy π

**Action Value Q(s, a):**
- "How good is it to take action a in state s?"
- Expected total discounted reward after taking action a in state s

**Why Q-values?** If you know Q(s, a) for all (s, a), the optimal policy is trivial:
```python
optimal_action(s) = argmax_a Q(s, a)
```

This is what Q-Learning (Chapter 11) learns!

---

### Block 5 — Applications Gallery (5 min)

| Domain | State | Actions | Reward |
|--------|-------|---------|--------|
| Games | Board position | Game moves | Win/lose |
| Robotics | Sensor readings | Motor commands | Task completion |
| Finance | Market prices | Buy/sell/hold | Profit/loss |
| Recommendation | User history | Show item | Click/engage |
| Healthcare | Patient state | Treatment | Health improvement |
| Traffic | Grid state | Light timings | Reduced wait time |

**Famous milestones:**
- 2013: DeepMind plays Atari games from pixels, superhuman level
- 2016: AlphaGo defeats world champion Lee Sedol at Go
- 2019: AlphaStar defeats top StarCraft II players
- 2022: AlphaCode writes competitive programming solutions

---

### Block 6 — Live Example: Grid World (10 min)

→ See `02-examples/ch10_rl_intro_examples.ipynb`

Demonstrate:
1. A simple grid world environment (states = cells, actions = NSEW)
2. Manual policy walkthrough
3. Reward signals and the path to goal
4. Visual animation of random vs optimal agent

---

## Instructor Notes

- The "training a dog" analogy resonates universally — use it liberally
- The Markov Property: don't go deep into the math — keep it intuitive
- Exploration vs exploitation: the restaurant analogy is always a hit
- The application gallery: ask students which applications surprise them most
- Grid world: make it visual and interactive — students love watching the agent learn

---

## Materials

- Slides: `01-slides/ch10_slides.md`
- Examples: `02-examples/ch10_rl_intro_examples.ipynb`
