"""Schematic figures for the GSP talk (synthetic particles, NOT simulation results).

Run:  python figures/make_figures.py   (needs numpy + matplotlib)
Writes vector PDFs next to this script.

Figures
  pic_vs_pif.pdf               same particles: 8x8 grid deposit (PIC) vs 8x8 Fourier modes (PIF)
  decomposition_balance.pdf    particle vs domain decomposition (+ rebalanced), particles per rank
  bug_rescale.pdf              per-rank local min/max rescaling, particle vs domain decomposition
  bug_reconstruct.pdf          density rebuilt from the all-reduced Fourier modes in both cases
  seed_noise.pdf               same RNG seed on every rank vs seed + 100*rank, mode noise vs W
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent
rgb = lambda r, g, b: (r / 255, g / 255, b / 255)
FZJ = dict(
    blue=rgb(2, 61, 107), lightblue=rgb(173, 189, 227), red=rgb(235, 95, 115),
    green=rgb(185, 210, 95), yellow=rgb(250, 235, 90), violet=rgb(175, 130, 185),
    orange=rgb(250, 180, 90), gray=rgb(156, 156, 156),
)
# rank colours: avoid orange (= ML side) and red (= bugs), see CLAUDE.md colour code
RANK = [FZJ["blue"], FZJ["green"], FZJ["violet"], FZJ["lightblue"]]
CMAP = LinearSegmentedColormap.from_list("fzj", ["white", FZJ["lightblue"], FZJ["blue"]])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 9,
    "axes.edgecolor": FZJ["gray"], "axes.linewidth": 0.6,
    "xtick.color": "0.35", "ytick.color": "0.35", "pdf.fonttype": 42,
})
rng = np.random.default_rng(7)


def sample_rejection(n, density, dim):
    """Draw n points in [0,1)^dim from an (unnormalised, max <= 1) density."""
    pts = []
    while sum(len(p) for p in pts) < n:
        x = rng.random((4 * n, dim))
        pts.append(x[rng.random(4 * n) < density(x)])
    return np.concatenate(pts)[:n]


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(FZJ["gray"])


# --------------------------------------------------------------------- PIC vs PIF
def pic_vs_pif():
    dens = lambda x: (1 + 0.8 * np.cos(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1])) / 1.8
    x = sample_rejection(20000, dens, 2)
    fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.15), constrained_layout=True)

    axs[0].scatter(x[:8000, 0], x[:8000, 1], s=1.0, color=FZJ["blue"], lw=0)
    axs[0].set_title("particles")

    m = 8
    h, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=m, range=[[0, 1], [0, 1]])
    axs[1].imshow(h.T, origin="lower", extent=[0, 1, 0, 1], cmap=CMAP, interpolation="nearest")
    for g in np.linspace(0, 1, m + 1):
        axs[1].axhline(g, color="white", lw=0.4); axs[1].axvline(g, color="white", lw=0.4)
    axs[1].set_title(f"PIC: deposit to {m}x{m} grid")

    # PIF: scatter directly to modes k in [-m/2, m/2), evaluate the truncated series
    k = np.arange(-m // 2, m // 2)
    ex = np.exp(-2j * np.pi * np.outer(k, x[:, 0]))
    ey = np.exp(-2j * np.pi * np.outer(k, x[:, 1]))
    rho_hat = ex @ ey.T / len(x)                       # rho_hat[kx, ky]
    g = np.linspace(0, 1, 200, endpoint=False)
    bx = np.exp(2j * np.pi * np.outer(g, k)); by = np.exp(2j * np.pi * np.outer(g, k))
    rho = np.real(bx @ rho_hat @ by.T)
    axs[2].imshow(rho.T, origin="lower", extent=[0, 1, 0, 1], cmap=CMAP)
    axs[2].set_title(f"PIF: {m}x{m} Fourier modes, no grid")
    for ax in axs:
        clean(ax); ax.set_aspect("equal")
    fig.savefig(OUT / "pic_vs_pif.pdf")
    plt.close(fig)


# ---------------------------------------------------------- decomposition / balance
def decomposition_balance():
    # beam-like cluster on a uniform background
    n = 4000
    nb = int(0.65 * n)
    beam = np.clip(rng.normal([0.32, 0.62], 0.09, (nb, 2)), 0, 0.999)
    x = np.concatenate([beam, rng.random((n - nb, 2))])

    part = rng.integers(0, 4, n)                                     # particle decomposition
    dom = (x[:, 0] >= 0.5).astype(int) * 2 + (x[:, 1] >= 0.5)        # 2x2 equal boxes
    xs = np.median(x[:, 0])                                          # ORB-style rebalanced boxes
    left = x[:, 0] < xs
    ys_l, ys_r = np.median(x[left, 1]), np.median(x[~left, 1])
    bal = np.where(left, (x[:, 1] >= ys_l).astype(int), 2 + (x[:, 1] >= ys_r))

    fig = plt.figure(figsize=(5.6, 2.75))
    gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1], hspace=0.12, wspace=0.12,
                          left=0.02, right=0.98, top=0.9, bottom=0.03)
    cases = [
        ("particle decomposition", part, []),
        ("domain decomposition", dom, [((0.5, 0), (0.5, 1)), ((0, 0.5), (1, 0.5))]),
        ("domain decomp., rebalanced", bal,
         [((xs, 0), (xs, 1)), ((0, ys_l), (xs, ys_l)), ((xs, ys_r), (1, ys_r))]),
    ]
    for j, (title, rank, cuts) in enumerate(cases):
        ax = fig.add_subplot(gs[0, j])
        ax.scatter(x[:, 0], x[:, 1], c=[RANK[r] for r in rank], s=1.2, lw=0)
        for (a, b) in cuts:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); clean(ax)
        ax.set_title(title)

        bx = fig.add_subplot(gs[1, j])
        frac = np.bincount(rank, minlength=4) / n
        bx.bar(range(4), frac, color=RANK, width=0.7)
        bx.axhline(0.25, color="black", lw=0.6, ls="--")
        bx.set_ylim(0, 0.6); bx.set_xticks([]); bx.set_yticks([])
        for s in ("top", "right", "left"):
            bx.spines[s].set_visible(False)
        bx.text(3.6, 0.27, "ideal", fontsize=6, ha="right", va="bottom")
        bx.text(-0.5, 0.58, "particles per rank", fontsize=7, va="top", color="0.3")
    fig.savefig(OUT / "decomposition_balance.pdf")
    plt.close(fig)


# ------------------------------------------------------------------- the bounds bug
def bug_data(n=4000, ranks=4):
    """1D density 1 + 0.6 cos(2 pi x) on [0,1); return positions + rank ids per case."""
    x = sample_rejection(n, lambda p: (1 + 0.6 * np.cos(2 * np.pi * p[:, 0])) / 1.6, 1)[:, 0]
    return x, {"particle": rng.integers(0, ranks, n), "domain": np.floor(x * ranks).astype(int)}


def local_rescale(x, rank):
    """What each rank does with LOCAL bounds: map [min_r, max_r] -> [0, 2 pi)."""
    xt = np.empty_like(x)
    for r in np.unique(rank):
        s = rank == r
        lo, hi = x[s].min(), x[s].max()
        xt[s] = 2 * np.pi * (x[s] - lo) / (hi - lo)
    return xt


def bug_rescale(x, ranks):
    fig, axs = plt.subplots(2, 2, figsize=(5.0, 3.1), sharey=True,
                            gridspec_kw=dict(hspace=0.6, wspace=0.1, left=0.13, right=0.97,
                                             top=0.8, bottom=0.12))
    sub = rng.choice(len(x), 500, replace=False)
    for i, (name, rank) in enumerate(ranks.items()):
        xt = local_rescale(x, rank)
        for j, (pos, xmax, lab) in enumerate([(x, 1, "physical position $x$"),
                                              (xt, 2 * np.pi, "after local rescaling")]):
            ax = axs[i, j]
            for r in range(4):
                s = sub[rank[sub] == r]
                ax.scatter(pos[s], r + rng.uniform(-0.25, 0.25, len(s)), color=RANK[r], s=3, lw=0)
                lo, hi = pos[rank == r].min(), pos[rank == r].max()
                ax.plot([lo, hi], [r + 0.38] * 2, color="black", lw=0.8)
                ax.plot([lo] * 2, [r + 0.3, r + 0.46], color="black", lw=0.8)
                ax.plot([hi] * 2, [r + 0.3, r + 0.46], color="black", lw=0.8)
            ax.set_xlim(-0.03 * xmax, 1.03 * xmax)
            ax.set_yticks(range(4)); ax.set_yticklabels([f"rank {r}" for r in range(4)], fontsize=7)
            ax.set_xticks([0, xmax]); ax.set_xticklabels(["0", "$L$" if j == 0 else r"$2\pi$"])
            ax.tick_params(length=2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                fig.text(0.5 * (ax.get_position().x0 + ax.get_position().x1), 0.95, lab,
                         ha="center", va="top")
        axs[i, 0].text(-0.24, 1.1, f"{name} decomposition", transform=axs[i, 0].transAxes,
                       fontweight="bold", color=FZJ["blue"])
    fig.savefig(OUT / "bug_rescale.pdf")
    plt.close(fig)


def bug_reconstruct(x, ranks, m=8):
    """Each rank computes partial modes sum_p exp(-i k x~_p); all_reduce sums them."""
    k = np.arange(-m // 2, m // 2 + 1)
    g = np.linspace(0, 2 * np.pi, 400)
    true = 1 + 0.6 * np.cos(g)
    fig, ax = plt.subplots(figsize=(5.0, 2.6), constrained_layout=True)
    ax.plot(g, true, color="black", lw=2.2, label="true density")
    style = {"particle": dict(color=FZJ["blue"], ls="--", lw=1.6),
             "domain": dict(color=FZJ["red"], ls="-", lw=1.6)}
    for name, rank in ranks.items():
        xt = local_rescale(x, rank)
        rho_hat = np.exp(-1j * np.outer(k, xt)).sum(axis=1) / len(x)   # = all_reduce over ranks
        rho = np.real(np.exp(1j * np.outer(g, k)) @ rho_hat)
        ax.plot(g, rho, label=f"{name} decomposition", **style[name])
    ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 1.9)
    ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([]); ax.set_xlabel(r"$\tilde x$")
    ax.set_ylabel("density from summed modes")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15))
    fig.savefig(OUT / "bug_reconstruct.pdf")
    plt.close(fig)


# ------------------------------------------------------------- same seed on every rank
def draw(seed, n):
    """Mirror the inference code: one legacy-seeded generator per rank (cp.random.seed)."""
    return np.random.RandomState(seed).random(n)


def seed_noise(n_vis=1600, n_mc=2**16, ks=np.arange(1, 9), trials=20):
    fig = plt.figure(figsize=(4.6, 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], wspace=0.25, hspace=0.55,
                          left=0.16, right=0.97, top=0.92, bottom=0.12)
    w = 4
    for j, (title, seed_of) in enumerate([("seed(152) on every rank", lambda r: 152),
                                          ("seed(152 + 100*rank)", lambda r: 152 + 100 * r)]):
        ax = fig.add_subplot(gs[0, j])
        for r in range(w):
            p = draw(seed_of(r), 2 * n_vis // w).reshape(-1, 2)
            ax.scatter(p[:, 0], p[:, 1], s=2.5, lw=0, color=RANK[r])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); clean(ax)
        ax.set_title(title, fontsize=8, family="monospace")
        distinct = n_vis // w if j == 0 else n_vis
        ax.text(0.5, -0.1, f"{distinct} distinct particles", transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="0.3")

    # RMS of the low Fourier modes of the particle density, fixed total N, W ranks
    ws = 2 ** np.arange(0, 7)
    ax = fig.add_subplot(gs[1, :])
    for label, same, style in [("same seed", True, dict(color=FZJ["red"], marker="o")),
                               ("seed + 100*rank", False, dict(color=FZJ["blue"], marker="s"))]:
        rms = []
        for wi in ws:
            vals = []
            for t in range(trials):
                base = 152 + 7919 * t
                x = np.concatenate([draw(base if same else base + 100 * r, n_mc // wi)
                                    for r in range(wi)])
                rho = np.exp(-2j * np.pi * np.outer(ks, x)).mean(axis=1)
                vals.append(np.mean(np.abs(rho) ** 2))
            rms.append(np.sqrt(np.mean(vals)))
        ax.plot(ws, rms, ms=3.5, lw=1.4, label=label, **style)
    ax.plot(ws, rms[0] * np.sqrt(ws), color="black", ls="--", lw=0.8, label=r"$\propto\sqrt{W}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ws); ax.set_xticklabels([str(v) for v in ws])
    ax.set_xlabel("GPUs $W$ (fixed total $N$)", fontsize=8)
    ax.set_ylabel(r"noise in $|\hat\rho_k|$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.minorticks_off()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(OUT / "seed_noise.pdf")
    plt.close(fig)


if __name__ == "__main__":
    pic_vs_pif()
    decomposition_balance()
    x, ranks = bug_data()
    bug_rescale(x, ranks)
    bug_reconstruct(x, ranks)
    seed_noise()
    print("figures written to", OUT)
