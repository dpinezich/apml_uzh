"""Ch10 — Introduction to RL: grid world figures, ε-decay curve, deterministic vs slippery.

Also exposes the shared GridWorld helpers used by imagegen/ch11.py.
"""
from imagegen.common import *

# ── Shared GridWorld (identical to the notebooks in 5-reinforcement_learning) ─
GRID = np.array([[3, 0, 0, 0],
                 [0, 1, 0, 1],
                 [0, 0, 0, 1],
                 [1, 0, 0, 2]])          # 0=Free 1=Hole 2=Goal 3=Start (= FrozenLake 4x4 map)
ROWS, COLS = GRID.shape
N_STATES, N_ACTIONS = ROWS * COLS, 4
ACTIONS = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
MOVES   = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
REWARDS = {0: -0.01, 1: -1.0, 2: 1.0, 3: -0.01}
ARROWS  = {0: "←", 1: "↓", 2: "→", 3: "↑"}
CELL_COLORS = {0: "#eef3f5", 1: RED, 2: GREEN, 3: BLUE}
CELL_LABELS = {0: "", 1: "H", 2: "G", 3: "S"}


def env_reset():
    return 0


def env_step(state, action):
    r, c = divmod(state, COLS)
    dr, dc = MOVES[action]
    nr, nc = min(ROWS - 1, max(0, r + dr)), min(COLS - 1, max(0, c + dc))
    nxt = nr * COLS + nc
    cell = GRID[nr, nc]
    return nxt, REWARDS[cell], cell in (1, 2)


def draw_grid(ax, show_state_ids=True, title=None):
    for r in range(ROWS):
        for c in range(COLS):
            ct = GRID[r, c]
            ax.add_patch(mpatches.FancyBboxPatch(
                (c + 0.04, ROWS - 1 - r + 0.04), 0.92, 0.92,
                boxstyle="round,pad=0.02", facecolor=CELL_COLORS[ct],
                edgecolor="white", lw=2, zorder=1))
            if CELL_LABELS[ct]:
                ax.text(c + 0.5, ROWS - 1 - r + 0.5, CELL_LABELS[ct], ha="center",
                        va="center", fontsize=18, fontweight="bold",
                        color="white" if ct else DARK, zorder=2)
            if show_state_ids:
                ax.text(c + 0.1, ROWS - 1 - r + 0.86, str(r * COLS + c), fontsize=8,
                        color=MUTED, ha="left", va="top", zorder=2)
    ax.set_xlim(0, COLS); ax.set_ylim(0, ROWS)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")


def train_q(n_episodes=3000, alpha=0.1, gamma=0.99, eps_decay=0.999,
            eps_min=0.01, max_steps=100, seed=42, snapshots=()):
    """Tabular Q-learning on the GridWorld. Returns Q, success list, snapshot dict."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    eps, success, snaps = 1.0, [], {}
    for ep in range(n_episodes):
        if ep in snapshots:
            snaps[ep] = (Q.copy(), eps)
        s = env_reset()
        for _ in range(max_steps):
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = env_step(s, a)
            Q[s, a] += alpha * (r + (0 if done else gamma * Q[s2].max()) - Q[s, a])
            s = s2
            if done:
                break
        success.append(float(done and r > 0))
        eps = max(eps * eps_decay, eps_min)
    snaps[n_episodes] = (Q.copy(), eps)
    return Q, success, snaps


def run_policy(policy_fn, rng, max_steps=30):
    s = env_reset(); traj = [s]; total = 0.0
    for _ in range(max_steps):
        s, r, done = env_step(s, policy_fn(s))
        traj.append(s); total += r
        if done:
            break
    return traj, total


def draw_traj(ax, traj, color=DARK):
    xs = [s % COLS + 0.5 for s in traj]
    ys = [ROWS - s // COLS - 0.5 for s in traj]
    ax.plot(xs, ys, "-", color=color, lw=2.5, alpha=0.55, zorder=3)
    ax.plot(xs[0] + 0.28, ys[0] - 0.28, "o", color=color, ms=7, zorder=4)
    ax.plot(xs[-1], ys[-1], "*", color=ORANGE, ms=18, markeredgecolor=DARK, zorder=5)


# ── Figures ──────────────────────────────────────────────────────────────────

def make_gridworld_env():
    fig, ax = plt.subplots(figsize=(5, 5))
    draw_grid(ax, title="GridWorld: 16 states, 4 actions")
    ax.text(2, -0.25, "S start · H hole (−1, episode ends) · G goal (+1) · every step −0.01",
            ha="center", fontsize=9.5, color=MUTED)
    ax.set_ylim(-0.5, ROWS)
    save(fig, "gridworld_env.png", ["ch10", "ch11"])


def make_random_vs_optimal():
    rng = np.random.default_rng(3)
    OPT = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 1, 10: 1, 13: 2, 14: 2}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5))
    # find a random trajectory that ends in a hole for illustration
    for _ in range(50):
        traj, tot = run_policy(lambda s: int(rng.integers(4)), rng, max_steps=12)
        if traj[-1] in (5, 7, 11, 12) and len(traj) > 4:
            break
    draw_grid(axes[0], title=f"Random policy\n(this episode: {len(traj)-1} steps, return {tot:.2f})")
    draw_traj(axes[0], traj, color=PURPLE)
    traj_o, tot_o = run_policy(lambda s: OPT[s], rng)
    draw_grid(axes[1], title=f"A good policy\n({len(traj_o)-1} steps, return {tot_o:.2f})")
    draw_traj(axes[1], traj_o, color=TEAL_DARK)
    fig.suptitle("Same environment — the policy makes the difference", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "random_vs_optimal_paths.png", ["ch10"])


def make_epsilon_decay():
    n = 5000
    eps = np.maximum(1.0 * 0.999 ** np.arange(n), 0.01)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(eps, color=TEAL_DARK, lw=2.5)
    ax.fill_between(range(n), 0, eps, color=TEAL, alpha=0.12)
    for x0, x1, lbl, col in [(0, 700, "mostly EXPLORE\n(ε > 0.5)", PURPLE),
                             (700, 2300, "balance", MUTED),
                             (2300, n, "mostly EXPLOIT\n(ε ≈ 0.1 → 0.01)", ORANGE)]:
        ax.axvspan(x0, x1, color=col, alpha=0.05)
        ax.text((x0 + x1) / 2, 0.75, lbl, ha="center", fontsize=10, color=col, fontweight="bold")
    ax.axhline(0.01, color=MUTED, ls=":", lw=1); ax.text(n, 0.03, "ε_min = 0.01", ha="right", fontsize=9, color=MUTED)
    ax.set_xlabel("Episode"); ax.set_ylabel("ε  (probability of a random action)")
    ax.set_title("ε-decay schedule:  ε ← max(ε · 0.999, 0.01),  start ε = 1.0")
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, "epsilon_decay.png", ["ch10", "ch11"])


def make_deterministic_vs_slippery():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
    for ax, title, probs in [
        (axes[0], "Deterministic (our GridWorld)\naction = Right", {2: 1.0}),
        (axes[1], "Slippery (FrozenLake, is_slippery=True)\naction = Right", {2: 1/3, 1: 1/3, 3: 1/3}),
    ]:
        draw_grid(ax, show_state_ids=False, title=title)
        r, c = 2, 1  # state 9
        x0, y0 = c + 0.5, ROWS - 1 - r + 0.5
        ax.plot(x0, y0, "o", color=DARK, ms=12, zorder=5)
        for a, p in probs.items():
            dr, dc = MOVES[a]
            ax.annotate("", xy=(x0 + dc * 0.85, y0 - dr * 0.85), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=TEAL_DARK if a == 2 else ORANGE,
                                        lw=2 + 4 * p, mutation_scale=18), zorder=6)
            ax.text(x0 + dc * 1.0 + (0.34 if dr != 0 else 0), y0 - dr * 1.0 + (0.28 if dr == 0 else 0),
                    f"{p:.0%}" if p == 1 else "1/3",
                    ha="center", fontsize=11, fontweight="bold",
                    color=TEAL_DARK if a == 2 else ORANGE, zorder=7)
    fig.suptitle("Where do I end up? Same state, same action — different outcome distributions",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "deterministic_vs_slippery.png", ["ch10", "ch11"])


def make_reward_shaping():
    """Three learned policies under three reward designs — reward shaping in one picture."""
    global REWARDS
    base = dict(REWARDS)
    configs = [
        ("Default rewards\nstep −0.01 · hole −1 · goal +1", dict(base)),
        ("Step reward +0.05 (!)\n→ agent farms steps, avoids the goal", {**base, 0: 0.05, 3: 0.05}),
        ("Step −0.5, hole −0.1\n→ cheapest exit is a hole", {**base, 0: -0.5, 3: -0.5, 1: -0.1}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (title, rew) in zip(axes, configs):
        REWARDS.clear(); REWARDS.update(rew)
        Q, succ, _ = train_q(n_episodes=3000)
        rng = np.random.default_rng(0)
        greedy = lambda s: int(np.argmax(Q[s]))
        wins = np.mean([run_policy(greedy, rng, max_steps=100)[0][-1] == 15 for _ in range(200)])
        draw_grid(ax, show_state_ids=False, title=title + f"\ngreedy policy reaches goal: {wins:.0%}")
        for s in range(N_STATES):
            if GRID[s // COLS, s % COLS] in (1, 2):
                continue
            ax.text(s % COLS + 0.5, ROWS - s // COLS - 0.5 - (0.18 if s == 0 else 0), ARROWS[int(np.argmax(Q[s]))],
                    ha="center", va="center", fontsize=20, color=DARK, fontweight="bold", zorder=6)
    REWARDS.clear(); REWARDS.update(base)
    fig.suptitle("Reward shaping: what you reward is what you get", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "reward_shaping.png", ["ch10"])


def generate():
    print("── Ch10 ──")
    make_gridworld_env()
    make_random_vs_optimal()
    make_epsilon_decay()
    make_deterministic_vs_slippery()
    make_reward_shaping()


if __name__ == "__main__":
    generate()
