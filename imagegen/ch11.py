"""Ch11 — Q-Learning: Q-table filling in (GIF), agent path improving (GIF),
FrozenLake deterministic vs slippery learning curves, effect of α on slippery ice."""
from imagegen.common import *
from imagegen.ch10 import (GRID, ROWS, COLS, N_STATES, N_ACTIONS, ARROWS, MOVES,
                           env_reset, env_step, draw_grid, train_q, run_policy, draw_traj)

SNAPS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]


def _policy_arrows(ax, Q, texts=None):
    out = []
    for s in range(N_STATES):
        r, c = divmod(s, COLS)
        if GRID[r, c] in (1, 2):
            continue
        visited = np.any(Q[s] != 0)
        t = ax.text(c + 0.5, ROWS - r - 0.5 - (0.2 if s == 0 else 0),
                    ARROWS[int(np.argmax(Q[s]))] if visited else "·",
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color=DARK if visited else MUTED, zorder=6)
        out.append(t)
    return out


def make_qtable_gif():
    Q, succ, snaps = train_q(n_episodes=3000, snapshots=SNAPS)
    succ = np.array(succ)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    axL, axR = axes
    state = {"artists": []}

    def update(i):
        ep = SNAPS[i]
        Qs, eps = snaps[ep]
        for a in state["artists"]:
            a.remove()
        state["artists"] = []
        axL.clear(); axR.clear()
        # left: max-Q heatmap with greedy arrows
        maxq = Qs.max(axis=1).reshape(ROWS, COLS)
        im = axL.imshow(maxq, cmap="YlGn", vmin=-0.2, vmax=1.0, extent=(0, COLS, 0, ROWS), zorder=0)
        for s in range(N_STATES):
            r, c = divmod(s, COLS)
            ct = GRID[r, c]
            if ct in (1, 2):
                axL.add_patch(mpatches.Rectangle((c, ROWS - 1 - r), 1, 1, facecolor=RED if ct == 1 else GREEN,
                                                 edgecolor="white", lw=2, zorder=1))
                axL.text(c + 0.5, ROWS - r - 0.5, "H" if ct == 1 else "G", ha="center", va="center",
                         color="white", fontsize=16, fontweight="bold", zorder=2)
            else:
                axL.add_patch(mpatches.Rectangle((c, ROWS - 1 - r), 1, 1, fill=False, edgecolor="white", lw=2, zorder=1))
                axL.text(c + 0.5, ROWS - r - 0.15, f"{maxq[r, c]:.2f}", ha="center", va="center",
                         fontsize=8.5, color=MUTED, zorder=2)
        axL.text(0.5, ROWS - 0.5 + 0.2, "S", ha="center", va="center", color=BLUE, fontsize=14, fontweight="bold", zorder=3)
        _policy_arrows(axL, Qs)
        axL.set_xlim(0, COLS); axL.set_ylim(0, ROWS); axL.set_aspect("equal"); axL.axis("off")
        rate = succ[max(0, ep - 100):ep].mean() if ep > 0 else 0.0
        axL.set_title(f"Episode {ep}   ε = {eps:.2f}\nmax Q per state · arrow = greedy action", fontsize=11)
        # right: rolling success curve up to ep
        w = 100
        cs = np.cumsum(np.insert(succ, 0, 0.0))
        idx = np.arange(1, len(succ) + 1)
        roll = (cs[idx] - cs[np.maximum(idx - w, 0)]) / np.minimum(idx, w)
        roll = np.insert(roll, 0, 0.0)              # roll[e] = success rate over episodes < e
        axR.plot(np.arange(len(roll)), roll, color=BORDER, lw=1.5)
        axR.plot(np.arange(ep + 1), roll[:ep + 1], color=TEAL_DARK, lw=2.5)
        axR.plot(ep, roll[ep], "o", color=ORANGE, ms=9)
        axR.set_xlim(0, 3000); axR.set_ylim(0, 1.05)
        axR.set_xlabel("Episode"); axR.set_ylabel("success rate (rolling 100)")
        axR.set_title(f"Success rate so far: {rate:.0%}", fontsize=11)
        axR.grid(True, alpha=0.3)
        for sp in ("top", "right"):
            axR.spines[sp].set_visible(False)

    fig.suptitle("Q-Learning fills in the Q-table — trial, error, update, repeat", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_gif(fig, update, len(SNAPS), "qtable_learning.gif", ["ch11"], fps=2, hold_last=4)


def make_agent_paths_gif():
    episodes = [1, 50, 3000]
    Q, succ, snaps = train_q(n_episodes=3000, snapshots=episodes)
    rng = np.random.default_rng(7)
    trajs = {}
    for ep in episodes:
        Qs, eps = snaps[ep]
        # episode 1: nothing learned yet -> random walk; later: what the agent has learned (greedy)
        eps_eval = 1.0 if ep == 1 else 0.0
        pol = lambda s, Qs=Qs, e=eps_eval: (int(rng.integers(N_ACTIONS)) if rng.random() < e else int(np.argmax(Qs[s])))
        traj, tot = run_policy(pol, rng, max_steps=14)
        trajs[ep] = (traj, tot, eps_eval)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    frames = []
    for k, ep in enumerate(episodes):
        for step in range(len(trajs[ep][0])):
            frames.append((k, step))
    frames = frames[:40]

    def update(i):
        k, step = frames[i]
        for j, ep in enumerate(episodes):
            ax = axes[j]; ax.clear()
            traj, tot, eps = trajs[ep]
            done_here = j < k or (j == k and step == len(traj) - 1)
            n = len(traj) - 1 if j < k else (step if j == k else 0)
            end = traj[n]
            outcome = "GOAL ✓" if end == 15 else ("hole ✗" if GRID[end // COLS, end % COLS] == 1
                                                   else ("stuck — bumps the wall (Q still 0 here)" if done_here else "…"))
            draw_grid(ax, show_state_ids=False,
                      title=f"After {ep} episode{'s' if ep > 1 else ''}  ({'random walk' if eps == 1 else 'greedy policy'})\n"
                            + (f"step {n}: {outcome}" if n > 0 or done_here else "step 0"))
            if n > 0:
                draw_traj(ax, traj[:n + 1], color=[PURPLE, BLUE, TEAL_DARK][j])
    fig.suptitle("The same agent, earlier and later in training", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_gif(fig, update, len(frames), "agent_paths_improving.gif", ["ch10", "ch11"], fps=3, hold_last=6)


def _train_frozenlake(slippery, n_episodes, alpha, gamma=0.99, decay=0.999, seed=42):
    import gymnasium as gym
    rng = np.random.default_rng(seed)
    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    env.reset(seed=seed)
    Q = np.zeros((16, 4)); eps = 1.0; succ = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        for _ in range(100):
            a = int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, term, trunc, _ = env.step(a)
            Q[s, a] += alpha * (r + (0 if term else gamma * Q[s2].max()) - Q[s, a])
            s = s2
            if term or trunc:
                break
        succ.append(float(r > 0)); eps = max(eps * decay, 0.01)
    # greedy evaluation
    wins = 0
    for _ in range(1000):
        s, _ = env.reset()
        for _ in range(100):
            s, r, term, trunc, _ = env.step(int(np.argmax(Q[s])))
            if term or trunc:
                break
        wins += r > 0
    return np.array(succ), wins / 1000


def make_frozenlake_curves():
    try:
        import gymnasium  # noqa
    except ImportError:
        print("  (gymnasium missing — skipping frozenlake figures)")
        return
    w = 200
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for slippery, col, lbl in [(False, TEAL_DARK, "deterministic (is_slippery=False)"),
                               (True, ORANGE, "slippery (is_slippery=True)")]:
        succ, greedy = _train_frozenlake(slippery, 5000, alpha=0.1)
        roll = np.convolve(succ, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(w - 1, 5000), roll, color=col, lw=2.5,
                label=f"{lbl} — greedy policy wins {greedy:.0%}")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Episode"); ax.set_ylabel(f"success rate (rolling {w})")
    ax.set_title("FrozenLake 4×4, same Q-learning code (α=0.1, γ=0.99, ε 1→0.01)")
    ax.legend(loc="lower right", frameon=False); ax.grid(True, alpha=0.3)
    ax.text(150, 0.97, "on slippery ice even the BEST policy\nonly succeeds ≈ 75% of the time",
            ha="left", va="top", fontsize=9, color=ORANGE, style="italic")
    fig.tight_layout()
    save(fig, "frozenlake_learning_curves.png", ["ch11"])

    # α effect on slippery ice
    fig, ax = plt.subplots(figsize=(9, 4.2))
    alphas = [0.05, 0.1, 0.3, 0.5, 0.9, 1.0]
    res = [_train_frozenlake(True, 5000, alpha=a)[1] for a in alphas]
    bars = ax.bar([str(a) for a in alphas], res, color=[TEAL if a == 0.1 else MUTED for a in alphas], alpha=0.85)
    for b, v in zip(bars, res):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1); ax.set_xlabel("learning rate α"); ax.set_ylabel("greedy-policy success (1000 test episodes)")
    good = [a for a, v in zip(alphas, res) if v >= max(res) - 0.05]
    ax.set_title(f"Slippery FrozenLake: α in [{min(good)}, {max(good)}] works; larger α over-reacts to lucky/unlucky slips")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "alpha_effect_slippery.png", ["ch11"])


def generate():
    print("── Ch11 ──")
    make_qtable_gif()
    make_agent_paths_gif()
    make_frozenlake_curves()


if __name__ == "__main__":
    generate()
