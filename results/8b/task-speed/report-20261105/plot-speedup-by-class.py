import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = ["1", "8", "32", "128"]
rows = [  # deployed pod, geometric mean over slices
 ("In distribution (math, code, OPB, multi-turn)", "#2a78d6", [2.59, 2.45, 2.12, 1.40]),
 ("Near-ID (JSON, tools, multi-lang code, Arena-Hard)", "#eb6834", [2.18, 1.84, 1.61, 1.20]),
 ("OOD task (chat, summarization, long ctx, translation)", "#1baf7a", [1.57, 1.44, 1.07, 1.02]),
 ("OOD language (12 non-English languages)", "#eda100", [1.32, 1.28, 1.22, 0.87]),
]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
x = range(len(C))
ax.axhline(1.0, color="#52514e", lw=1, ls=(0, (4, 3)))
ax.text(-0.15, 0.965, "no speedup (1.0x)", va="top", color="#52514e", fontsize=9)
for label, col, ys in rows:
    ax.plot(x, ys, color=col, lw=2, marker="o", ms=7, mec="#fcfcfb", mew=2, label=label)
    for xi, y in zip(x, ys):
        ax.annotate(f"{y:.2f}x", (xi, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color="#0b0b0b")
ax.set_xticks(list(x), [f"C={c}" for c in C])
ax.set_xlabel("Concurrent requests")
ax.set_ylabel("Speedup vs plain (output tokens/s)")
ax.set_ylim(0.7, 2.85); ax.set_xlim(-0.2, 3.55)
ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
ax.tick_params(colors="#52514e")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("DSpark e10 k=3 on the deployed Apertus 1.5 8B pod: speedup by task class",
             loc="left", fontsize=12, color="#0b0b0b")
fig.text(0.01, 0.01, "Geometric mean over slices (28 slices, 32 prompts each); live pod via gateway, 2 plain/DSpark window pairs. "
         "Controlled same-GPU run agrees within ~±7%.", fontsize=7.5, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{__import__('sys').argv[1]}", facecolor=fig.get_facecolor())
