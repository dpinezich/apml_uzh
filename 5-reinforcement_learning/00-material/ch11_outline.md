# Chapter 11 — Q-Learning

**Session:** 4 | **Chapter:** 2 of 3 | **Duration:** ~45 min (blocks ~34 min + 10 min exercise)  
**Format:** Slides + live demo (FrozenLake, Gymnasium) + exercise (GridWorld)

---

## Learning Objectives

By the end of this chapter, students will be able to:
- Implement tabular Q-learning from scratch: Q-table, ε-greedy, TD update, training loop
- Explain the TD update term by term (target, TD error, α) and how value flows backwards from the reward (credit assignment)
- Explain the roles of α, γ, ε and their typical values; know why a large α fails in stochastic environments
- Train and evaluate an agent on FrozenLake (deterministic and slippery) and read a Q-table / policy plot
- Distinguish training success rate from the success rate of the greedy policy
- (Further reading) name SARSA, DQN, policy gradient, model-based RL

---

## Timing Breakdown

| Block | Content | Time |
|-------|---------|------|
| 1 | From Ch10 to Ch11 + Q-table GIF + the Q-table | 5 min |
| 2 | TD update (idea) + worked step | 8 min |
| 3 | Full algorithm + hyperparameters | 5 min |
| 4 | FrozenLake: deterministic vs slippery · why RL is hard | 5 min |
| 5 | Quick check (quiz) | 2 min |
| 6 | Live demo: FrozenLake notebook | 8 min |
| 7 | **Exercise** | **10 min** |
| 8 | Further reading + takeaways | 1 min |
| **Total** | | **44 min** |

---

## Content Outline

### Block 1 — Motivation and the Q-table (5 min)
- Ch10: know Q → argmax. Ch11: learn Q from experience.
- `qtable_learning.gif`: max-Q heatmap + greedy arrows filling in from the goal backwards, success curve alongside
- Q = zeros((16, 4)); ties in argmax → action 0 → why ε starts at 1

### Block 2 — The TD update (8 min)
- Naming: a **temporal-difference update derived from the Bellman optimality equation** (the equation is the fixed point; the update is how we get there)
- `target = r + γ · max_a' Q(s', a')` (no future if done) · `Q(s,a) += α · (target − Q(s,a))`
- Worked step with our numbers (α = 0.1, γ = 0.99): Q(14,→) = 0.10, then Q(13,→) = 0.0089 → value flows backwards = credit assignment
- `td_error_flow.png`

### Block 3 — Algorithm + hyperparameters (5 min)
- Full loop (our `env_step(state, action)` API; Gymnasium 5-tuple `obs, reward, terminated, truncated, info`)
- Table: α = 0.1, γ = 0.99, ε 1 → 0.01 (×0.999), 3000–5000 episodes — **identical in slides, demo, exercise, animation**
- `alpha_effect_slippery.png` (real numbers): α ≤ 0.5 fine, α ≥ 0.9 collapses on slippery ice

### Block 4 — FrozenLake and why RL is hard (5 min)
- Same map as the GridWorld; reward only at the goal; `is_slippery=True` → p = 1/3 intended direction
- `frozenlake_learning_curves.png` (real): deterministic ~100 %, slippery ~70 %; optimal policy on slippery ice ≈ 75 % → **the environment sets the ceiling**
- Sparse rewards · stochasticity · credit assignment; convergence guarantee (visit everything, α decreasing)

### Block 5 — Quiz (2 min)
- α = 1 deterministic vs slippery · γ = 0 in FrozenLake · training success ≠ greedy-policy success

### Block 6 — Live demo (8 min)
→ `02-examples/ch11_rl_algorithms_examples.ipynb`
1. Gymnasium wrapper (`make_env(slippery)`), fallback GridWorld if gymnasium is missing
2. `q_learning(...)` ~25 lines; train deterministic (~99 %) and slippery (~65–70 %); `evaluate_greedy`
3. Learning curves + ε curve; Q heatmaps per action + greedy policy arrows (slippery policy: "into the wall" next to holes is the safe move)

### Block 7 — Exercise (10 min)
→ `03-exercises/ch11_rl_algorithms_exercises.ipynb`
- Task 1 Q-table (1 min) · Task 2 ε-greedy (3 min) · Task 3 TD update with two numeric checks (3 min) · Task 4 fill three lines of the given loop + learning curve/policy plot (3 min)
- Bonus A: run your loop on slippery FrozenLake · Bonus B: SARSA (one changed line), comparison text derived from the numbers

### Block 8 — Further reading (1 min)
- SARSA (on-policy), DQN, policy gradient / PPO (RLHF), model-based, off- vs on-policy — words only

---

## Instructor Notes

- Let the Q-table GIF loop once in silence; ask "where does knowledge appear first?"
- Go slowly through the worked TD step; the exercise's check values (0.1, 0.0089) are exactly these
- "Training success rate" includes 5 % random actions — always evaluate the greedy policy (train ≠ test, again)
- Typical exercise bugs: missing `0 if done`, updating the wrong state, `max` over the wrong axis
- Fast students: Bonus A/B or the Ch10 reward-shaping bonus notebook

## Materials

- Slides: `01-slides/ch11_slides.md` (images: `qtable_learning.gif`, `td_error_flow.png`, `alpha_effect_slippery.png`, `frozenlake_learning_curves.png`; also `gridworld_env.png`, `epsilon_decay.png`, `deterministic_vs_slippery.png`, `agent_paths_improving.gif` shared with Ch10)
- Examples: `02-examples/ch11_rl_algorithms_examples.ipynb` (requires `gymnasium>=1.0`, falls back to GridWorld)
- Exercises / Solutions: `03-exercises/ch11_rl_algorithms_exercises.ipynb`, `04-solutions/ch11_rl_algorithms_solutions.ipynb`
- Animation: `0-animations/04_rl_agent_learning.ipynb`
- Image generator: `imagegen/ch11.py`
