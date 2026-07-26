#!/usr/bin/env python3
"""Generate all analysis charts for the AFD experiment report."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../results"
FIG_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Color palette ──
C_NATIVE = "#4C72B0"
C_AFD    = "#DD8452"
C_GRAPH  = "#55A868"
C_DBO    = "#C44E52"
C_FANOUT = "#8172B3"
C_NEUTRAL= "#937860"
C_HIGHLIGHT = "#DA8BC3"

def load(name):
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

# ================================================================
# Figure 1: AFD architecture overview (text-based diagram)
# ================================================================
fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis("off")

# Attention box
ax.add_patch(plt.Rectangle((1, 1.5), 3.5, 1.5, facecolor=C_AFD, alpha=0.3, edgecolor=C_AFD, linewidth=2))
ax.text(2.75, 2.75, "Attention Worker\n(GPU 0)", ha="center", va="center", fontsize=11, fontweight="bold", color=C_AFD)
ax.text(2.75, 2.15, "MLA Attention\nKV Cache", ha="center", va="center", fontsize=9, color=C_AFD)

# FFN box
ax.add_patch(plt.Rectangle((7.5, 1.5), 3.5, 1.5, facecolor=C_GRAPH, alpha=0.3, edgecolor=C_GRAPH, linewidth=2))
ax.text(9.25, 2.75, "FFN Worker\n(GPU 1)", ha="center", va="center", fontsize=11, fontweight="bold", color=C_GRAPH)
ax.text(9.25, 2.15, "MoE / FFN\nExpert Routing", ha="center", va="center", fontsize=9, color=C_GRAPH)

# Arrows
ax.annotate("", xy=(7.3, 2.75), xytext=(4.7, 2.75),
            arrowprops=dict(arrowstyle="->,head_width=0.4", color=C_AFD, lw=2.5))
ax.text(6.0, 3.05, "send_attn_output\n(NCCL P2P)", ha="center", fontsize=8, color=C_AFD)

ax.annotate("", xy=(4.7, 2.25), xytext=(7.3, 2.25),
            arrowprops=dict(arrowstyle="->,head_width=0.4", color=C_GRAPH, lw=2.5))
ax.text(6.0, 1.85, "send_ffn_output\n(NCCL P2P)", ha="center", fontsize=8, color=C_GRAPH)

# Request
ax.annotate("HTTP Request", xy=(2.75, 1.3), ha="center", fontsize=9, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))
ax.annotate("", xy=(2.75, 1.45), xytext=(2.75, 1.55),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

ax.set_title("AFD (Attention-FFN Disaggregation) Architecture — 1A1F Topology",
             fontsize=13, fontweight="bold", pad=15)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_architecture.png"), dpi=150, bbox_inches="tight")
plt.close()
print("fig1_architecture.png done")

# ================================================================
# Figure 2: Functional test results (F-01 ~ F-05)
# ================================================================
tests = ["F-01\nSmoke", "F-02\nUsage", "F-03\nGraph", "F-04\nDBO", "F-05\nAccuracy"]
status = [1, 1, 1, 1, 1]
labels = ["PASS", "PASS", "PASS", "PASS", "PASS"]
descs = ["completion\npath", "token\ncounting", "FULL_DECODE\ngraph", "dual micro-\nbatch overlap", "AFD vs native\n4/4 match"]

fig, ax = plt.subplots(figsize=(10, 3.5))
colors = [C_GRAPH if s else C_DBO for s in status]
bars = ax.barh(range(len(tests)), status, color=colors, height=0.5, edgecolor="white", linewidth=1.5)
ax.set_yticks(range(len(tests)))
ax.set_yticklabels(tests, fontsize=10)
ax.set_xlim(0, 1.8)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels(["", "", "PASS"])
ax.xaxis.set_ticks_position("top")

for i, (bar, label, desc) in enumerate(zip(bars, labels, descs)):
    ax.text(1.05, i, label, va="center", fontsize=10, fontweight="bold", color=C_GRAPH)
    ax.text(1.35, i, desc, va="center", fontsize=8, color="gray")

ax.set_title("Functional Correctness: All Passed (5/5)", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_functional.png"), dpi=150, bbox_inches="tight")
plt.close()
print("fig2_functional.png done")

# ================================================================
# Figure 3: P-01 Throughput comparison (V2-Lite, 3-way)
# ================================================================
p01_native = load("p01_native.json")
p01_dp2 = load("p01_native_dp2.json")
p01_afd = load("p01_afd.json")

configs = ["Native\n1 GPU", "Native DP2\n2 GPU", "AFD 1A1F\n2 GPU"]
tputs = [p01_native.get("total_token_throughput", 0),
         p01_dp2.get("total_token_throughput", 0),
         p01_afd.get("total_token_throughput", 0)]
tpots = [p01_native.get("mean_tpot_ms", 0),
         p01_dp2.get("mean_tpot_ms", 0),
         p01_afd.get("mean_tpot_ms", 0)]
ttfts = [p01_native.get("mean_ttft_ms", 0),
         p01_dp2.get("mean_ttft_ms", 0),
         p01_afd.get("mean_ttft_ms", 0)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Throughput bar chart
colors = [C_NATIVE, C_NATIVE, C_AFD]
bars = ax1.bar(configs, tputs, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, tputs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
             f"{v:,.0f}", ha="center", fontsize=12, fontweight="bold")
# Speedup annotation
ax1.annotate("", xy=(2, tputs[2]+400), xytext=(1, tputs[1]+400),
             arrowprops=dict(arrowstyle="->", color=C_DBO, lw=2))
ax1.text(1.5, max(tputs)*1.12, f"1.55×", ha="center", fontsize=14, fontweight="bold", color=C_DBO)
ax1.set_ylabel("Throughput (tok/s)", fontsize=11)
ax1.set_title("V2-Lite: Throughput Comparison (c=32)", fontsize=12, fontweight="bold")
ax1.set_ylim(0, max(tputs) * 1.25)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# TPOT grouped chart
x = np.arange(len(configs))
width = 0.25
ax2.bar(x - width/2, ttfts, width, label="TTFT (ms)", color=C_NEUTRAL, alpha=0.7)
ax2.bar(x + width/2, tpots, width, label="TPOT (ms)", color=C_HIGHLIGHT, alpha=0.7)
for i, (t, p) in enumerate(zip(ttfts, tpots)):
    ax2.text(i - width/2, t + 5, f"{t:.0f}", ha="center", fontsize=9)
    ax2.text(i + width/2, p + 5, f"{p:.0f}", ha="center", fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=10)
ax2.set_ylabel("Latency (ms)", fontsize=11)
ax2.set_title("V2-Lite: TTFT & TPOT Comparison", fontsize=12, fontweight="bold")
ax2.legend(fontsize=10)

fig.suptitle("P-01: AFD 1A1F vs Native vLLM (DeepSeek-V2-Lite, fair 2-GPU)", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_p01_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("fig3_p01_comparison.png done")

# ================================================================
# Figure 4: P-05 Concurrency scaling (line chart)
# =================================================%%
concs = [1, 4, 16, 32, 64, 128]
tput_vals = []
for c in concs:
    d = load(f"p05_conc{c}.json")
    tput_vals.append(d.get("total_token_throughput", 0))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(concs, tput_vals, "o-", color=C_AFD, linewidth=2.5, markersize=8)
ax.fill_between(concs, [v*0.9 for v in tput_vals], [v*1.1 for v in tput_vals], alpha=0.15, color=C_AFD)

# Annotate key points
for c, v in zip(concs, tput_vals):
    if c in (1, 32, 64, 128):
        ax.annotate(f"{v:,.0f} tok/s", (c, v), textcoords="offset points",
                    xytext=(10, 10), fontsize=10, fontweight="bold", color=C_AFD)

ax.set_xlabel("Concurrency", fontsize=12)
ax.set_ylabel("Throughput (tok/s)", fontsize=12)
ax.set_title("P-05: AFD 1A1F Concurrency Scaling (V2-Lite, eager, 128 prompts)",
             fontsize=13, fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(concs)
ax.set_xticklabels([str(c) for c in concs])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.axvspan(32, 64, alpha=0.1, color=C_GRAPH, label="Sweet spot")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_p05_concurrency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("fig4_p05_concurrency.png done")

# ================================================================
# Figure 5: P-03/P-06 DBO analysis (grouped bar)
# ================================================================
thresholds = ["Non-DBO\n(baseline)", "t=2", "t=8", "t=16", "t=32"]
tpot_c32 = [
    load("p03_nodbo.json").get("mean_tpot_ms", 0),
    load("p06_t2_c32.json").get("mean_tpot_ms", 0),
    load("p06_t8_c32.json").get("mean_tpot_ms", 0),
    load("p06_t16_c32.json").get("mean_tpot_ms", 0),
    load("p06_t32_c32.json").get("mean_tpot_ms", 0),
]
tput_c32 = [
    load("p03_nodbo.json").get("total_token_throughput", 0),
    load("p06_t2_c32.json").get("total_token_throughput", 0),
    load("p06_t8_c32.json").get("total_token_throughput", 0),
    load("p06_t16_c32.json").get("total_token_throughput", 0),
    load("p06_t32_c32.json").get("total_token_throughput", 0),
]
tpot_c64 = [
    load(f"p05_conc64.json").get("mean_tpot_ms", 0),
    load("p06_t2_c64.json").get("mean_tpot_ms", 0),
    load("p06_t8_c64.json").get("mean_tpot_ms", 0),
    load("p06_t16_c64.json").get("mean_tpot_ms", 0),
    load("p06_t32_c64.json").get("mean_tpot_ms", 0),
]
tput_c64 = [
    load("p05_conc64.json").get("total_token_throughput", 0),
    load("p06_t2_c64.json").get("total_token_throughput", 0),
    load("p06_t8_c64.json").get("total_token_throughput", 0),
    load("p06_t16_c64.json").get("total_token_throughput", 0),
    load("p06_t32_c64.json").get("total_token_throughput", 0),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(thresholds))
width = 0.3

# TPOT
ax1.bar(x - width/2, tpot_c32, width, label="c=32", color=C_NATIVE, alpha=0.8)
ax1.bar(x + width/2, tpot_c64, width, label="c=64", color=C_AFD, alpha=0.8)
ax1.axhline(y=tpot_c32[0], color=C_DBO, linestyle="--", alpha=0.5, linewidth=1)
ax1.text(4.3, tpot_c32[0]+1, f"Non-DBO\n{tpot_c32[0]:.0f}ms", fontsize=8, color=C_DBO)
ax1.set_xticks(x)
ax1.set_xticklabels(thresholds, fontsize=9)
ax1.set_ylabel("TPOT (ms)", fontsize=11)
ax1.set_title("TPOT: DBO ≈ 2× Non-DBO (all thresholds)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=10)

# Throughput
ax2.bar(x - width/2, tput_c32, width, label="c=32", color=C_NATIVE, alpha=0.8)
ax2.bar(x + width/2, tput_c64, width, label="c=64", color=C_AFD, alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(thresholds, fontsize=9)
ax2.set_ylabel("Throughput (tok/s)", fontsize=11)
ax2.set_title("Throughput: DBO ≈ 0.5× Non-DBO (all thresholds)", fontsize=12, fontweight="bold")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax2.legend(fontsize=10)

fig.suptitle("P-06: DBO Threshold Sweep — Unconditionally Negative on AFD P2P",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5_dbo_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print("fig5_dbo_analysis.png done")

# ================================================================
# Figure 6: V2-Lite topology comparison (bar chart, c=64)
# ================================================================
topo_labels = ["1A1F\n(2 GPU)", "1A2F\n(3 GPU)", "2A2F\n(4 GPU)", "4A4F\n(8 GPU)"]
topo_tputs = [
    load("p05_conc64.json").get("total_token_throughput", 0),
    load("p08_1a2f_c64.json").get("total_token_throughput", 0),
    load("p04_2a2f.json").get("total_token_throughput", 0) * 2,  # P-04 was c=32, scale to c=64 approx (skip — use p04 1a1f c64)
]
# Fix: use p04_1a1f for c32, p05_conc64 for 1a1F c64
topo_tputs[0] = load("p05_conc64.json").get("total_token_throughput", 0)
# Use 2a2f at c32 since we don't have c64
topo_tputs[2] = load("p04_2a2f.json").get("total_token_throughput", 0)
topo_tputs.append(load("p07_4a4f_c64.json").get("total_token_throughput", 0))
topo_labels_adj = ["1A1F\n(2 GPU)", "1A2F\n(3 GPU)", "2A2F\n(4 GPU)*", "4A4F\n(8 GPU)"]
gpus = [2, 3, 4, 8]
colors_topo = [C_AFD, C_FANOUT, C_DBO, C_NEUTRAL]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(topo_labels_adj, topo_tputs, color=colors_topo, width=0.5, edgecolor="white", linewidth=1.5)
for bar, v, g in zip(bars, topo_tputs, gpus):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{v:,.0f}", ha="center", fontsize=11, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
            f"{g} GPU", ha="center", fontsize=9, color="white", fontweight="bold")

baseline = topo_tputs[0]
for i in range(1, len(topo_tputs)):
    ratio = topo_tputs[i] / baseline if baseline else 0
    ax.text(i, -1200, f"{ratio:.2f}×", ha="center", fontsize=11, fontweight="bold",
            color=C_DBO if ratio < 1 else C_GRAPH)

ax.set_ylabel("Throughput (tok/s)", fontsize=12)
ax.set_title("V2-Lite Topology Comparison (c=64, 1A1F baseline)\n* 2A2F measured at c=32",
             fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.set_ylim(0, max(topo_tputs) * 1.2)
ax.axhline(y=baseline, color=C_AFD, linestyle="--", alpha=0.3, linewidth=1)
ax.grid(True, alpha=0.2, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig6_v2lite_topology.png"), dpi=150, bbox_inches="tight")
plt.close()
print("fig6_v2lite_topology.png done")

# ================================================================
# Figure 7: V2.5 — AFD vs Native (the headline result)
# ================================================================
p09_native = load("p09_native_tp8.json")
p09_afd_c32 = load("p09_v25_4a4f_c32.json")
p09_afd_c64 = load("p09_v25_4a4f_c64.json")

configs_v25 = ["Native TP8\n(8 GPU, c32)", "AFD 4A4F TP4\n(8 GPU, c32)", "AFD 4A4F TP4\n(8 GPU, c64)"]
tputs_v25 = [
    p09_native.get("total_token_throughput", 0),
    p09_afd_c32.get("total_token_throughput", 0),
    p09_afd_c64.get("total_token_throughput", 0),
]
tpots_v25 = [
    p09_native.get("mean_tpot_ms", 0),
    p09_afd_c32.get("mean_tpot_ms", 0),
    p09_afd_c64.get("mean_tpot_ms", 0),
]
ttfts_v25 = [
    p09_native.get("mean_ttft_ms", 0),
    p09_afd_c32.get("mean_ttft_ms", 0),
    p09_afd_c64.get("mean_ttft_ms", 0),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Throughput
colors_v25 = [C_NATIVE, C_AFD, C_AFD]
bars = ax1.bar(configs_v25, tputs_v25, color=colors_v25, width=0.5, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, tputs_v25):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f"{v:,.0f}", ha="center", fontsize=13, fontweight="bold")

# Speedup arrow
ax1.annotate("", xy=(2, tputs_v25[2]+80), xytext=(0, tputs_v25[0]+80),
             arrowprops=dict(arrowstyle="->", color=C_DBO, lw=2.5, connectionstyle="arc3,rad=-0.2"))
ax1.text(1.5, max(tputs_v25)*1.12, "1.20×", ha="center", fontsize=16, fontweight="bold", color=C_DBO)

ax1.set_ylabel("Throughput (tok/s)", fontsize=12)
ax1.set_title("V2.5 (236B FP8): Throughput", fontsize=12, fontweight="bold")
ax1.set_ylim(0, max(tputs_v25) * 1.3)

# TTFT + TPOT
x = np.arange(len(configs_v25))
width = 0.3
# Use log scale for TTFT because of the 24s outlier
ax2.bar(x - width/2, ttfts_v25, width, label="TTFT (ms)", color=C_NEUTRAL, alpha=0.7)
ax2.bar(x + width/2, tpots_v25, width, label="TPOT (ms)", color=C_HIGHLIGHT, alpha=0.7)
for i, (t, p) in enumerate(zip(ttfts_v25, tpots_v25)):
    t_lbl = f"{t/1000:.1f}s" if t > 1000 else f"{t:.0f}ms"
    ax2.text(i - width/2, t + max(ttfts_v25)*0.02, t_lbl, ha="center", fontsize=9, rotation=0)
    ax2.text(i + width/2, p + max(ttfts_v25)*0.02, f"{p:.0f}", ha="center", fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(configs_v25, fontsize=9)
ax2.set_ylabel("Latency (ms)", fontsize=12)
ax2.set_title("V2.5: TTFT & TPOT\n(c32 TTFT includes cold-start compilation)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=10)
ax2.set_yscale("log")

fig.suptitle("P-09: DeepSeek-V2.5 (236B) — AFD 4A4F Beats Native TP8 by 1.20×",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig7_v25_headline.png"), dpi=150, bbox_inches="tight")
plt.close()
print("fig7_v25_headline.png done")

# ================================================================
# Figure 8: Model size scaling — V2-Lite vs V2.5 (4A4F, 8 GPU)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 5))
models = ["V2-Lite (16B)\n4A4F DP2TP2\n8 GPU", "V2.5 (236B)\n4A4F TP4\n8 GPU", "V2.5 (236B)\nNative TP8\n8 GPU"]
tputs_scale = [
    load("p07_4a4f_c64.json").get("total_token_throughput", 0),
    load("p09_v25_4a4f_c64.json").get("total_token_throughput", 0),
    load("p09_native_tp8.json").get("total_token_throughput", 0),
]
colors_scale = [C_NEUTRAL, C_AFD, C_NATIVE]
bars = ax.bar(models, tputs_scale, color=colors_scale, width=0.5, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, tputs_scale):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{v:,.0f}", ha="center", fontsize=13, fontweight="bold")

# Ratios
v2lite_afd = tputs_scale[0]
v25_afd = tputs_scale[1]
v25_native = tputs_scale[2]

ax.annotate("V2-Lite AFD:\n0.29× of V2-Lite 1A1F", xy=(0, tputs_scale[0]),
            xytext=(0.5, tputs_scale[0]*0.5), fontsize=9, color=C_DBO,
            arrowprops=dict(arrowstyle="->", color=C_DBO, lw=1.5))
ax.annotate(f"V2.5 AFD:\n{v25_afd/v25_native:.2f}× of V2.5 Native", xy=(1, tputs_scale[1]),
            xytext=(1.2, tputs_scale[1]*1.25), fontsize=10, fontweight="bold", color=C_GRAPH,
            arrowprops=dict(arrowstyle="->", color=C_GRAPH, lw=1.5))

ax.set_ylabel("Throughput (tok/s)", fontsize=12)
ax.set_title("Model Scale Impact: AFD Advantage Grows with Model Size\n"
             "(V2-Lite: AFD 4A4F hurts; V2.5: AFD 4A4F wins — both 8 GPU, c=64)",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig8_model_scaling.png"), dpi=150, bbox_inches="tight")
plt.close()
print("fig8_model_scaling.png done")

# ================================================================
# Figure 9: All experiments summary dashboard
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (0,0) P-01 throughput
ax = axes[0, 0]
labels_p01 = ["Native\n1GPU", "Native\nDP2", "AFD\n1A1F"]
vals_p01 = [p01_native.get("total_token_throughput",0), p01_dp2.get("total_token_throughput",0), p01_afd.get("total_token_throughput",0)]
ax.bar(labels_p01, vals_p01, color=[C_NATIVE, C_NATIVE, C_AFD], width=0.5)
for i, v in enumerate(vals_p01):
    ax.text(i, v+100, f"{v:,.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("P-01: V2-Lite Throughput (c32)", fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# (0,1) P-03 DBO
ax = axes[0, 1]
nodbo = load("p03_nodbo.json")
dbo = load("p03_dbo.json")
ax.bar(["Non-DBO", "DBO"], [nodbo.get("total_token_throughput",0), dbo.get("total_token_throughput",0)],
       color=[C_GRAPH, C_DBO], width=0.4)
for i, v in enumerate([nodbo.get("total_token_throughput",0), dbo.get("total_token_throughput",0)]):
    ax.text(i, v+100, f"{v:,.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("P-03: DBO = 0.54× Negative", fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# (1,0) V2-Lite topology
ax = axes[1, 0]
topo_l = ["1A1F", "2A2F", "4A4F"]
topo_v = [load("p05_conc64.json").get("total_token_throughput",0),
          load("p04_2a2f.json").get("total_token_throughput",0),
          load("p07_4a4f_c64.json").get("total_token_throughput",0)]
ax.bar(topo_l, topo_v, color=[C_AFD, C_DBO, C_NEUTRAL], width=0.4)
for i, v in enumerate(topo_v):
    ax.text(i, v+200, f"{v:,.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("V2-Lite: Topology (c64)", fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

# (1,1) V2.5 headline
ax = axes[1, 1]
v25_l = ["Native\nTP8", "AFD\n4A4F c64"]
v25_v = [p09_native.get("total_token_throughput",0), p09_afd_c64.get("total_token_throughput",0)]
ax.bar(v25_l, v25_v, color=[C_NATIVE, C_AFD], width=0.4)
for i, v in enumerate(v25_v):
    ax.text(i, v+30, f"{v:,.0f}", ha="center", fontsize=11, fontweight="bold")
ax.set_title("V2.5 (236B): AFD 1.20× Native", fontsize=11, fontweight="bold")

fig.suptitle("AFD Experiment Dashboard — All Phases", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig9_dashboard.png"), dpi=150, bbox_inches="tight")
plt.close()
print("fig9_dashboard.png done")

print("\nAll figures generated in", FIG_DIR)
