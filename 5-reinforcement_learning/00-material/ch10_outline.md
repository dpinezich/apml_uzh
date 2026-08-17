# Chapter 10 — Introduction to Reinforcement Learning

**Session:** 4 | **Chapter:** 1 of 3 | **Duration:** ~45 min (blocks sum to ~39 min + buffer)  
**Format:** Slides + live demo (GridWorld). No exercise phase — the exercise notebook is a **bonus** (reward shaping with a learning agent; do it after Ch11 or at home).

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Explain the RL loop: agent, environment, state, action, reward, episode, return, policy
- Distinguish RL from supervised and unsupervised learning ("the agent creates its own data")
- Describe an MDP (S, A, P, R, γ) intuitively, incl. deterministic vs stochastic transitions
- Explain the exploration–exploitation dilemma and ε-greedy with decay
- Say what Q(s, a) means and why knowing Q makes the policy trivial (`argmax`)
- Recognise the credit-assignment problem and reward-design pitfalls (reward hacking)

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | Third paradigm + RL loop + analogies (credit assignment introduced) | 7 min |
| 2 | Vocabulary on the GridWorld (state, action, reward, episode, return, policy) | 5 min |
| 3 | MDP · deterministic vs slippery · discount factor γ | 6 min |
| 4 | Exploration vs exploitation · ε-greedy · ε-decay | 5 min |
| 5 | Q(s, a) and the Q-table · Quick check (quiz) | 6 min |
| 6 | Applications + milestones | 2 min |
| 7 | Live demo: GridWorld notebook | 8 min |
| **Total** | | **39 min** (+ buffer) |

---

## Content Outline

### Block 1 — The third paradigm (7 min)
- Supervised: (X, y) · Unsupervised: X · **RL: no data set — only rewards for actions the agent tries**
- The loop: state → action → reward + next state; goal = maximise the *total* reward of an episode
- Analogies table (dog, bike, video game, chess). Chess row → **credit assignment**: nobody says which move was the mistake

### Block 2 — Vocabulary on the GridWorld (5 min)
- 4×4 grid = FrozenLake map; 16 states, 4 actions (←0 ↓1 →2 ↑3); rewards +1 goal, −1 hole, −0.01 per step
- Episode, return (shortest path: 0.95), policy = table state → action
- Figure `random_vs_optimal_paths.png`: same world, different policy

### Block 3 — MDP, stochasticity, γ (6 min)
- (S, A, P, R, γ); Markov property (GPS analogy)
- `deterministic_vs_slippery.png`: our GridWorld is deterministic; FrozenLake slips with p = 1/3 sideways — P(s'|s,a) captures this
- γ: impatience **and** the mechanism by which value flows backwards; we use γ = 0.99 in Ch11

### Block 4 — Exploration vs exploitation (5 min)
- Restaurant analogy; ε-greedy code; ε-decay curve (`epsilon_decay.png`, ε 1 → 0.01, ×0.999 per episode — the exact schedule used in Ch11)

### Block 5 — Q-values + quick check (6 min)
- Q(s, a) = expected discounted return; Q-table 16×4; policy = argmax per row
- Quiz slide (robot vacuum MDP; ε = 0 forever; γ = 0) with `<v-click>` answers

### Block 6 — Applications (2 min)
- Games, robotics, recommenders, data-centre cooling, RLHF for LLMs; milestones 2013 → 2022

### Block 7 — Live demo (8 min)
→ `02-examples/ch10_rl_intro_examples.ipynb`
1. GridWorld from scratch (`env_reset()`, `env_step(state, action)` — the same API as Ch11 and the animation)
2. Random agent over 1000 episodes (~1–2 % success) + return histogram
3. Hand-crafted policy: 100 %, return 0.95; sanity check of every table entry
4. Teaser GIF `agent_paths_improving.gif` (Q-learning agent after 1 / 50 / 3000 episodes)

---

## Bonus exercise — Reward shaping (~10 min, after Ch11 or at home)

→ `03-exercises/ch10_rl_intro_exercises.ipynb` · solutions in `04-solutions/`

A Q-learning agent is provided as a black box (`train_q_learning`). Students change the rewards and watch the *learned* policy:
- default → shortest safe path
- step reward **+0.05** → the agent farms steps and never finishes (reward hacking)
- step −0.5, hole −0.1 → the agent jumps into the nearest hole (cheapest exit)
- design your own. Figure `reward_shaping.png` shows all three policies.

---

## Instructor Notes

- Dog analogy resonates; the chess row is the bridge to credit assignment (picked up in Ch11 with the TD update).
- Do the quiz — Q3 (γ = 0) makes the role of γ concrete.
- Keep the MDP formal part light; only P (slippery) and γ are needed later.
- Everything in the demo (rewards, API, hyperparameters) is identical to Ch11 and to `0-animations/04_rl_agent_learning.ipynb`.

## Materials

- Slides: `01-slides/ch10_slides.md` (images: `rl_loop.png`, `gridworld_env.png`, `random_vs_optimal_paths.png`, `deterministic_vs_slippery.png`, `discount_factor.png`, `epsilon_decay.png`, `agent_paths_improving.gif`)
- Examples: `02-examples/ch10_rl_intro_examples.ipynb`
- Bonus exercise/solutions: `03-exercises/ch10_rl_intro_exercises.ipynb`, `04-solutions/ch10_rl_intro_solutions.ipynb`
- Image generator: `imagegen/ch10.py`
