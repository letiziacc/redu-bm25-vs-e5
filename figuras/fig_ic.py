import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})
AZUL, LAR, TXT, MUT, GR = "#0aa0f3", "#f6790b", "#1a1a1a", "#5b6573", "#b9bfc7"
lo, est, hi = -0.02344, -0.00819, 0.00726
fig, ax = plt.subplots(figsize=(3.3, 1.05))
ax.set_xlim(-0.05, 0.05); ax.set_ylim(-1.15, 1.25)
ax.axhline(0, color=GR, lw=0.8, zorder=1)
for t, l in [(-0.05, "−0,05"), (-0.025, "−0,025"), (0, "0"), (0.025, "+0,025"), (0.05, "+0,05")]:
    ax.plot([t, t], [-0.1, 0.1], color=GR, lw=0.8, zorder=1)
    ax.text(t, -0.3, l, ha="center", va="top", color=MUT, fontsize=6.5)
ax.plot([0, 0], [-0.22, 0.95], color=TXT, lw=0.8, ls=(0, (3, 2)), zorder=2)
ax.text(0, 1.0, "sem diferença", ha="center", va="bottom", color=TXT, fontsize=6.5)
ax.plot([lo, hi], [0, 0], color=AZUL, lw=6, solid_capstyle="butt", zorder=3)
ax.scatter([est], [0], s=34, color=LAR, edgecolor="white", linewidth=0.8, zorder=4)
ax.text(est - 0.0022, 0.3, "−0,008", ha="center", va="bottom", color=TXT, fontweight="bold")
ax.text(lo, -0.72, "−0,023", ha="center", va="top", color=TXT)
ax.text(hi, -0.72, "+0,007", ha="center", va="top", color=TXT)
ax.text(-0.05, 1.0, "← E5 pior", ha="left", va="bottom", color=MUT, fontsize=6.5)
ax.text(0.05, 1.0, "E5 melhor →", ha="right", va="bottom", color=MUT, fontsize=6.5)
ax.axis("off")
fig.savefig("fig_ic_mrr.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("fig_ic_mrr.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
