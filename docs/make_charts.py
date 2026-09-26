"""
Regenerate the README result charts (light + dark) from downloaded run data.

Expects, relative to the repo root (see README / setup-guide for the download commands):
  tb_logs/tensorboard_logs/<run>/events.*   baseline train.py TensorBoard log
  runs_v2_fold5/metrics.json                v2 run, fold 5
  cv5/fold{1..5}/metrics.json               v2 5-fold CV run

Usage (repo root, venv active; needs matplotlib):  python docs/make_charts.py
"""
import glob, json, os, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

U = "."
OUT = os.path.join("docs", "images"); os.makedirs(OUT, exist_ok=True)
base_run = sorted(glob.glob(os.path.join(U, "tb_logs", "tensorboard_logs", "run_*")))[-1]
ea = EventAccumulator(base_run); ea.Reload()
base = [(e.step + 1, e.value) for e in ea.Scalars("Accuracy/Validation")]
v2 = json.load(open(f"{U}/runs_v2_fold5/metrics.json"))["history"]
v2_ema = [(h["epoch"], h["ema_val_acc"]) for h in v2]
folds = [json.load(open(f"{U}/cv5/fold{i}/metrics.json"))["final_ema_acc"] for i in range(1, 6)]
mean, sd = st.mean(folds), st.stdev(folds)

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", band="#cde2fb"),
    "dark":  dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926", band="#1c3553"),
}

def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0, pad=6)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8); ax.set_axisbelow(True)

def title(fig, t, head, sub):
    fig.text(0.06, 0.94, head, color=t["text"], fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.875, sub, color=t["text2"], fontsize=9.5, ha="left", va="top")

for mode, t in THEMES.items():
    # ---- Chart 1: validation accuracy curves ----------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160)
    fig.patch.set_facecolor(t["surface"]); style(ax, t)
    fig.subplots_adjust(left=0.06, right=0.80, top=0.78, bottom=0.12)
    ax.axhline(81.3, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)))
    ax.text(1, 82.6, "human 81.3%", color=t["muted"], fontsize=8.5)
    for data, c, lab in ((base, t["s2"], "Baseline — from scratch (train.py)"),
                         (v2_ema, t["s1"], "v2 — ImageNet ResNet-34 + EMA (train_v2.py)")):
        xs, ys = zip(*data)
        ax.plot(xs, ys, color=c, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=lab)
        ax.scatter([xs[-1]], [ys[-1]], s=46, color=c, edgecolors=t["surface"], linewidths=2, zorder=5)
    ax.annotate(f"v2  {v2_ema[-1][1]:.2f}%", (v2_ema[-1][0], v2_ema[-1][1]), xytext=(8, 4),
                textcoords="offset points", color=t["text"], fontsize=9.5, fontweight="bold", va="center")
    ax.annotate(f"Baseline  {base[-1][1]:.2f}%\n(best epoch {max(v for _, v in base):.1f}%)", (base[-1][0], base[-1][1]),
                xytext=(8, -2), textcoords="offset points", color=t["text"], fontsize=9.5, va="center", annotation_clip=False)
    ax.set_xlim(0, 101); ax.set_ylim(0, 100)
    ax.set_xlabel("Epoch", color=t["muted"], fontsize=9); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.04), frameon=False, fontsize=9, labelcolor=t["text2"])
    title(fig, t, "Validation accuracy on held-out fold 5",
          "v2 curve = EMA weights (the saved model); it trails the raw weights for the first ~15 epochs by design")
    fig.savefig(os.path.join(OUT, f"accuracy_curves_{mode}.png"), facecolor=t["surface"]); plt.close(fig)

    # ---- Chart 2: 5-fold CV dot plot -----------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=160)
    fig.patch.set_facecolor(t["surface"]); style(ax, t)
    fig.subplots_adjust(left=0.06, right=0.80, top=0.74, bottom=0.12)
    x = list(range(1, 6))
    ax.axhspan(mean - sd, mean + sd, color=t["band"], alpha=0.55 if mode == "light" else 0.8, linewidth=0)
    ax.axhline(mean, color=t["s1"], linewidth=2)
    ax.text(5.45, mean, f"mean {mean:.2f}%\n± {sd:.2f} (1 std)", color=t["text"], fontsize=9.5,
            fontweight="bold", va="center")
    ax.scatter(x, folds, s=80, color=t["s1"], edgecolors=t["surface"], linewidths=2, zorder=5)
    for xi, f in zip(x, folds):
        above = f >= mean
        ax.annotate(f"{f:.2f}%", (xi, f), xytext=(0, 11 if above else -12), textcoords="offset points",
                    ha="center", va="bottom" if above else "top", color=t["text2"], fontsize=9)
    ax.set_xticks(x, [f"Fold {i}" for i in x]); ax.set_xlim(0.5, 5.4)
    ax.set_ylim(88, 95); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    title(fig, t, "5-fold cross-validation — v2 recipe",
          "Final-epoch EMA model on each held-out fold (trained on the other four); y-axis starts at 88%")
    fig.savefig(os.path.join(OUT, f"cv5_folds_{mode}.png"), facecolor=t["surface"]); plt.close(fig)
print("ok", folds, round(mean, 2), round(sd, 2))
