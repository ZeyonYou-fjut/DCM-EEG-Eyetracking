"""
DCM-DNN Report Figure Generation Script
Generates 10 standalone PNG figures to results/report_figures/
"""

import os
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# ─── Font Settings ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 19,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.unicode_minus': False
})
print("[INFO] Using serif font (Times New Roman / DejaVu Serif)")


def setup_font():
    """Restore rcParams after plt.style.use() resets them."""
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 19,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.unicode_minus': False
    })

# ─── Color Constants ────────────────────────────────────────────────────────────
CLR_DCM   = '#C44E52'   # DCM-DNN highlight (red)
CLR_BLUE  = '#4C72B0'   # Primary blue
CLR_BLUES = ['#4C72B0', '#5B8DB8', '#6AAEC0', '#7ACFC8', '#8AEFD0']
CLR_GRAY  = '#8C8C8C'

# ─── Path Settings ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
FIG_DIR     = os.path.join(RESULTS_DIR, 'report_figures')
PAPER_FIG_DIR = os.path.join(ROOT_DIR, 'paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# ─── Load Data ──────────────────────────────────────────────────────────────────
def load_json(fname):
    with open(os.path.join(RESULTS_DIR, fname), 'r', encoding='utf-8') as f:
        return json.load(f)

print("Loading experiment data...")
ablation_data  = load_json('ablation.json')
baseline_data  = load_json('baseline_compare.json')
robust_data    = load_json('robustness.json')
tinyml_data    = load_json('tinyml.json')
print("Data loaded.")

DPI = 300


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Model Architecture Overview
# ══════════════════════════════════════════════════════════════════════════════
def fig01_architecture():
    fig, ax = plt.subplots(figsize=(12, 11), dpi=DPI)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 12)
    ax.axis('off')

    def draw_box(ax, x, y, w, h, text, color='#4C72B0', fontsize=18, textcolor='white'):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='white',
                                        linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=textcolor, fontweight='bold', zorder=4)

    def draw_arrow(ax, x1, y1, x2, y2):
        """Simple straight arrow (for vertical connections)"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                   lw=2.2, connectionstyle='arc3,rad=0'))

    def draw_step_arrow(ax, x1, y1, x2, y2):
        """Step-shaped arrow: go vertically to midpoint, then horizontally, then vertically to endpoint"""
        mid_y = (y1 + y2) / 2
        ax.plot([x1, x1], [y1, mid_y], color='#555555', lw=2.2, solid_capstyle='round', zorder=2)
        ax.plot([x1, x2], [mid_y, mid_y], color='#555555', lw=2.2, solid_capstyle='round', zorder=2)
        ax.annotate('', xy=(x2, y2), xytext=(x2, mid_y),
                    arrowprops=dict(arrowstyle='->', color='#555555',
                                   lw=2.2, connectionstyle='arc3,rad=0'))

    # ── Layer coordinates (center_x=5.5, ylim=[0,12], layer spacing ≈1.8) ──
    #
    # Input       y=10.8  h=0.90  top=11.25  bottom=10.35
    # Branches    y= 9.0  h=0.85  top= 9.425 bottom= 8.575
    # GatedFusion y= 7.1  h=0.95  top= 7.575 bottom= 6.625
    # SSI+Fusion  y= 5.2  h=0.90  top= 5.65  bottom= 4.75
    # Utility     y= 3.4  h=0.90  top= 3.85  bottom= 2.95
    # Classif     y= 1.6  h=0.85  top= 2.025 bottom= 1.175

    # ── Input Layer ──
    # center=10.8, h=0.90 → top=11.25, bottom=10.35
    draw_box(ax, 5.5, 10.8, 4.5, 0.90,
             'Input Layer  67 Features\nET(14) + EEG(46) + BEH(7)',
             color='#2D6A4F', fontsize=20)

    # ── Three Branches ──
    # center=9.0, h=0.85 → top=9.425, bottom=8.575
    draw_box(ax, 2.2, 9.0, 3.0, 0.85, 'ET Branch\n14->32  (576 params)',   color='#1D6FA4', fontsize=18)
    draw_box(ax, 5.5, 9.0, 3.0, 0.85, 'EEG Branch\n46->32  (1896 params)', color='#1D6FA4', fontsize=18)
    draw_box(ax, 8.8, 9.0, 3.0, 0.85, 'BEH Branch\n7->16  (128 params)',   color='#1D6FA4', fontsize=18)

    # Input bottom(10.35) → Branches top(9.425)
    draw_step_arrow(ax, 4.3, 10.35, 2.2, 9.425)   # left (step)
    draw_arrow(ax, 5.5, 10.35, 5.5, 9.425)          # center (straight)
    draw_step_arrow(ax, 6.7, 10.35, 8.8, 9.425)    # right (step)

    # ── Gated Fusion ──
    # center=7.1, h=0.95 → top=7.575, bottom=6.625
    draw_box(ax, 5.5, 7.1, 6.2, 0.95,
             'Gated Fusion\nGating Network  (3200 params)\n[Mixed Logit Random Coeff. -- DCM]',
             color='#9B2335', fontsize=18)
    # Branches bottom(8.575) → Gated Fusion top(7.575)
    draw_step_arrow(ax, 2.2, 8.575, 3.8, 7.575)   # left (step)
    draw_arrow(ax, 5.5, 8.575, 5.5, 7.575)          # center (straight)
    draw_step_arrow(ax, 8.8, 8.575, 7.2, 7.575)    # right (step)

    # ── SSI + Fusion ──
    # center=5.2, h=0.90 → top=5.65, bottom=4.75
    draw_box(ax, 5.5, 5.2, 5.8, 0.90,
             'SSI + Fusion Layer  (4320 params)\n[Subject-Specific Intercept -- DCM SSI]',
             color='#805AD5', fontsize=18)
    # Gated Fusion bottom(6.625) → SSI top(5.65)
    draw_arrow(ax, 5.5, 6.625, 5.5, 5.65)

    # ── Utility Function ──
    # center=3.4, h=0.90 → top=3.85, bottom=2.95
    draw_box(ax, 5.5, 3.4, 4.5, 0.90,
             'Utility Function  V(b, x)\nU = Vb*x + SSI + eps',
             color='#C44E52', fontsize=19, textcolor='white')
    # ASC bottom(4.75) → Utility top(3.85)
    draw_arrow(ax, 5.5, 4.75, 5.5, 3.85)

    # ── Classification ──
    # center=1.6, h=0.85 → top=2.025, bottom=1.175
    draw_box(ax, 5.5, 1.6, 3.8, 0.85,
             'Classification\nSoftmax -> Buy / Not Buy',
             color='#2D6A4F', fontsize=19)
    # Utility bottom(2.95) → Classification top(2.025)
    draw_arrow(ax, 5.5, 2.95, 5.5, 2.025)

    # ── Right-side annotations (aligned with Gated/ASC/Utility) ──
    ax.text(10.2, 7.1, 'DCM\nRandom\nCoeff.', ha='center', va='center', fontsize=17,
            color='#9B2335', style='italic')
    ax.text(10.2, 5.2, 'DCM\nSSI', ha='center', va='center', fontsize=17,
            color='#805AD5', style='italic')
    ax.text(10.2, 3.4, 'DCM\nUtility\nV(b,x)', ha='center', va='center', fontsize=17,
            color='#C44E52', style='italic')

    ax.set_title('DCM-DNN Model Architecture\n(Grounded in Discrete Choice Model Theory)',
                 fontsize=24, fontweight='bold', pad=15, loc='center')
    out = os.path.join(FIG_DIR, 'fig01_architecture.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  [OK] Fig.1 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Parameter Distribution Pie Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig02_param_distribution():
    labels  = ['ET Branch', 'EEG Branch', 'BEH Branch', 'Gating Network',
               'Fusion Layer', 'Classifier Head', 'SSI', 'Other']
    sizes   = [576, 1896, 128, 3200, 4144, 50, 176, 3418]
    colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
               '#937860', '#DA8BC3', '#8C8C8C']
    explode = [0, 0, 0, 0.04, 0.04, 0, 0, 0]

    total = sum(sizes)

    # Only show percentage label for slices >= 3% to avoid overlap
    def autopct_filter(pct):
        return f'{pct:.1f}%' if pct >= 3.0 else ''

    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.22)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, explode=explode,
        autopct=autopct_filter, startangle=140,
        pctdistance=0.80, shadow=False,
        wedgeprops=dict(linewidth=1.2, edgecolor='white')
    )
    for at in autotexts:
        at.set_fontsize(14)
        at.set_fontweight('bold')

    # Legend includes ALL modules with exact numbers (including small ones)
    legend_labels = [f'{l}  ({s:,} params, {s/total*100:.1f}%)'
                     for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='lower center',
              bbox_to_anchor=(0.5, -0.38), ncol=2, fontsize=14,
              framealpha=0.9, edgecolor='#cccccc', fancybox=True,
              markerscale=1.0, borderpad=0.8, labelspacing=0.6)
    ax.set_title(f'Parameter Distribution by Module\nTotal Parameters: {total:,}',
                 fontsize=22, fontweight='bold', pad=15, loc='center')
    out = os.path.join(FIG_DIR, 'fig02_param_distribution.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  [OK] Fig.2 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Baseline Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig03_baseline_comparison():
    keys   = ['B0', 'B1', 'B3', 'B5', 'B6']
    labels = ['DCM-DNN', 'Logistic\nRegression', 'SVM\n(RBF)', 'Simple\nMLP', 'Matched\nMLP']
    means  = [baseline_data[k]['acc_mean'] * 100 for k in keys]
    stds   = [baseline_data[k]['acc_std']  * 100 for k in keys]
    colors = [CLR_DCM, CLR_BLUES[0], CLR_BLUES[1], CLR_BLUES[2], CLR_BLUES[3]]

    try:
        style = 'seaborn-v0_8-whitegrid'
        plt.style.use(style)
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass
    setup_font()

    fig, ax = plt.subplots(figsize=(10, 9), dpi=DPI)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  width=0.55, edgecolor='white', linewidth=1.2,
                  error_kw=dict(elinewidth=1.5, ecolor='#333333'))

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + s + 0.5,
                f'{m:.2f}%', ha='center', va='bottom', fontsize=15, fontweight='bold',
                color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=19)
    ax.set_ylim(65, 98)
    ax.set_title('Accuracy Comparison with Single-Model Baselines',
                 fontsize=22, fontweight='bold', pad=15, loc='center')

    # Best line
    ax.axhline(y=means[0], color=CLR_DCM, linestyle='--', alpha=0.5, linewidth=1.2)
    ax.text(len(labels) - 0.4, means[0] + 0.3, f'Best: {means[0]:.2f}%',
            color=CLR_DCM, fontsize=14, va='bottom', fontweight='bold')

    legend_patch = mpatches.Patch(color=CLR_DCM, label='DCM-DNN (Proposed)')
    ax.legend(handles=[legend_patch], loc='upper right', fontsize=14,
              framealpha=0.9, edgecolor='#cccccc', fancybox=True,
              ncol=1, markerscale=1.0, borderpad=0.8, labelspacing=0.6)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='both', width=2.0, length=6)
    # X-axis arrow
    ax.annotate('', xy=(1.02, 0), xycoords='axes fraction',
                xytext=(-0.01, 0), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    # Y-axis arrow
    ax.annotate('', xy=(0, 1.02), xycoords='axes fraction',
                xytext=(0, -0.01), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    out = os.path.join(FIG_DIR, 'fig03_baseline_comparison.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  [OK] Fig.3 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Parameter Efficiency Scatter Plot
# ══════════════════════════════════════════════════════════════════════════════
def fig04_param_efficiency():
    fig, ax = plt.subplots(figsize=(10, 9), dpi=DPI)

    # B0 DCM-DNN  ── offset upper-right
    ax.scatter([13588], [baseline_data['B0']['acc_mean']*100],
               marker='*', s=500, color=CLR_DCM, zorder=5, label='DCM-DNN (Proposed)')
    ax.annotate(f"DCM-DNN\n{baseline_data['B0']['acc_mean']*100:.2f}%",
                xy=(13588, baseline_data['B0']['acc_mean']*100),
                xytext=(22000, baseline_data['B0']['acc_mean']*100 + 1.2),
                fontsize=14, color=CLR_DCM, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CLR_DCM, lw=1.2,
                                connectionstyle='arc3,rad=-0.15'))

    # B1 LR  ── offset upper-left
    lr_params = 67 + 1
    ax.scatter([lr_params], [baseline_data['B1']['acc_mean']*100],
               marker='o', s=120, color=CLR_BLUES[0], zorder=5, label='Logistic Regression')
    ax.annotate(f"LR\n{baseline_data['B1']['acc_mean']*100:.2f}%",
                xy=(lr_params, baseline_data['B1']['acc_mean']*100),
                xytext=(lr_params*3, baseline_data['B1']['acc_mean']*100 + 1.2),
                fontsize=14, color=CLR_BLUES[0], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CLR_BLUES[0], lw=1.0,
                                connectionstyle='arc3,rad=-0.2'))

    # B5 Simple MLP  ── offset lower-left (away from Matched MLP)
    ax.scatter([6498], [baseline_data['B5']['acc_mean']*100],
               marker='s', s=120, color=CLR_BLUES[1], zorder=5, label='Simple MLP')
    ax.annotate(f"Simple MLP\n{baseline_data['B5']['acc_mean']*100:.2f}%",
                xy=(6498, baseline_data['B5']['acc_mean']*100),
                xytext=(900, baseline_data['B5']['acc_mean']*100 - 2.5),
                fontsize=14, color=CLR_BLUES[1], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CLR_BLUES[1], lw=1.0,
                                connectionstyle='arc3,rad=0.25'))

    # B6 Matched MLP  ── offset lower-right (separated from Simple MLP)
    ax.scatter([19362], [baseline_data['B6']['acc_mean']*100],
               marker='^', s=120, color=CLR_BLUES[2], zorder=5, label='Matched MLP')
    ax.annotate(f"Matched MLP\n{baseline_data['B6']['acc_mean']*100:.2f}%",
                xy=(19362, baseline_data['B6']['acc_mean']*100),
                xytext=(28000, baseline_data['B6']['acc_mean']*100 - 2.2),
                fontsize=14, color=CLR_BLUES[2], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=CLR_BLUES[2], lw=1.0,
                                connectionstyle='arc3,rad=0.2'))

    # SVM: no parameter count, shown as horizontal line
    ax.axhline(y=baseline_data['B3']['acc_mean']*100,
               color=CLR_BLUES[3], linestyle=':', linewidth=1.5, alpha=0.8)
    ax.text(80, baseline_data['B3']['acc_mean']*100 + 0.3,
            f"SVM(RBF): {baseline_data['B3']['acc_mean']*100:.2f}% (no comparable param count)",
            color=CLR_BLUES[3], fontsize=13, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Parameters (log scale)', fontsize=19)
    ax.set_ylabel('Accuracy (%)', fontsize=19)
    ax.set_ylim(74, 91)
    ax.set_title('Accuracy vs. Model Complexity',
                 fontsize=22, fontweight='bold', pad=15, loc='center')
    ax.legend(loc='upper right', fontsize=14,
              framealpha=0.9, edgecolor='#cccccc', fancybox=True,
              ncol=2, markerscale=1.0, borderpad=0.8, labelspacing=0.6)
    ax.grid(True, alpha=0.3, which='both')
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='both', width=2.0, length=6)
    # X-axis arrow
    ax.annotate('', xy=(1.02, 0), xycoords='axes fraction',
                xytext=(-0.01, 0), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    # Y-axis arrow
    ax.annotate('', xy=(0, 1.02), xycoords='axes fraction',
                xytext=(0, -0.01), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig04_param_efficiency.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  [OK] Fig.4 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Modality Ablation Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig05_modality_ablation():
    keys   = ['A0', 'A4', 'A5', 'A6']
    labels = ['Full Model\n(ET+EEG+BEH)', 'w/o BEH\n(ET+EEG)', 'w/o EEG\n(ET+BEH)', 'w/o ET\n(EEG+BEH)']
    means  = [ablation_data[k]['acc_mean'] * 100 for k in keys]
    stds   = [ablation_data[k]['acc_std']  * 100 for k in keys]
    colors = [CLR_DCM, '#5B8DB8', '#6AAEC0', '#7ACFC8']

    # Use fixed figsize + explicit subplots_adjust to avoid whitespace issues
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.18)

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  width=0.55, edgecolor='white', linewidth=1.2,
                  error_kw=dict(elinewidth=1.5, ecolor='#333333'))

    baseline = means[0]
    for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
        # Value label
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + s + 0.3,
                f'{m:.2f}%', ha='center', va='bottom', fontsize=15, fontweight='bold',
                color='#333333')
        # Delta label
        if i > 0:
            delta = m - baseline
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                    f'{delta:+.2f}%', ha='center', va='center',
                    fontsize=14, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=19)
    ax.set_ylim(74, 92)
    ax.set_title('Modality Ablation (Leave-One-Out)',
                 fontsize=22, fontweight='bold', pad=15, loc='center')
    ax.axhline(y=baseline, color=CLR_DCM, linestyle='--', alpha=0.5, linewidth=1.2)

    legend_patch = mpatches.Patch(color=CLR_DCM, label='Full Multimodal Baseline')
    ax.legend(handles=[legend_patch], loc='upper right', fontsize=14,
              framealpha=0.9, edgecolor='#cccccc', fancybox=True,
              ncol=1, markerscale=1.0, borderpad=0.8, labelspacing=0.6)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='both', width=2.0, length=6)
    # X-axis arrow
    ax.annotate('', xy=(1.02, 0), xycoords='axes fraction',
                xytext=(-0.01, 0), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    # Y-axis arrow
    ax.annotate('', xy=(0, 1.02), xycoords='axes fraction',
                xytext=(0, -0.01), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    out = os.path.join(FIG_DIR, 'fig05_modality_ablation.png')
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"  [OK] Fig.5 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6: Component Ablation Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig06_component_ablation():
    keys   = ['C0', 'C1', 'C2', 'C3', 'C4']
    labels = ['Full\n(C0)', 'No Gate\n(C1)', 'No SSI\n(C2)',
              'No EEG-Sub\n(C3)', 'No Gate+SSI\n(C4)']
    means  = [ablation_data[k]['acc_mean'] * 100 for k in keys]
    stds   = [ablation_data[k]['acc_std']  * 100 for k in keys]

    contributions = {
        'C1': 'Gate: +1.23%',
        'C2': 'SSI: +1.53%',
        'C3': 'EEG-Sub: +1.49%',
        'C4': 'Gate+SSI: +2.75%',
    }
    # Staggered y-offset for contribution labels: odd annotations raised, even slightly lower, to avoid overlap between adjacent labels
    # key order: C1(Gate), C2(SSI), C3(EEG-Sub), C4(Gate+SSI)
    contrib_y_offset = {
        'C1': 3.0,
        'C2': 6.5,
        'C3': 3.0,
        'C4': 6.5,
    }
    colors = [CLR_DCM, '#5B8DB8', '#6AAEC0', '#7ACFC8', '#8AEFD0']

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.18)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  width=0.6, edgecolor='white', linewidth=1.2,
                  error_kw=dict(elinewidth=1.5, ecolor='#333333'))

    baseline = means[0]
    ylim_lo, ylim_hi = 74, 99
    ax.set_ylim(ylim_lo, ylim_hi)

    for i, (bar, m, s, k) in enumerate(zip(bars, means, stds, keys)):
        bar_cx  = bar.get_x() + bar.get_width() / 2.
        bar_h   = bar.get_height()
        top     = bar_h + s   # top of error bar

        # ── Accuracy value: placed at bar center (white bold)
        ax.text(bar_cx, bar_h / 2.,
                f'{m:.2f}%', ha='center', va='center',
                fontsize=13, fontweight='bold', color='white', zorder=5)

        # ── Contribution label: staggered y-offset to avoid adjacent label overlap (orange italic) + guide line
        # Use axes-transform to ensure annotation stays within axes bounds
        if k in contributions:
            y_offset = contrib_y_offset[k]
            annot_y  = top + y_offset
            # Clamp annot_y within ylim to avoid exceeding axes
            annot_y  = min(annot_y, ylim_hi - 0.5)
            ax.annotate(contributions[k],
                        xy=(bar_cx, top + 0.3),           # arrow points to top of error bar
                        xytext=(bar_cx, annot_y),          # annotation position (staggered height)
                        ha='center', va='bottom',
                        fontsize=12, color='#D07020',
                        fontstyle='italic', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF5E0',
                                  edgecolor='#D07020', alpha=0.85, linewidth=0.8),
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.8),
                        zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=19)
    ax.set_title('Component Ablation Study\n(Quantifiable Contribution of Each DCM Component)',
                 fontsize=22, fontweight='bold', pad=15, loc='center')
    ax.axhline(y=baseline, color=CLR_DCM, linestyle='--', alpha=0.5, linewidth=1.2)
    # Use axes-transform so the label stays inside the plot regardless of x-axis range
    y_norm = (baseline - ylim_lo) / (ylim_hi - ylim_lo)   # normalized y within ylim
    ax.text(0.98, y_norm + 0.015, f'Baseline: {baseline:.2f}%',
            transform=ax.transAxes,
            color=CLR_DCM, fontsize=14, fontweight='bold',
            ha='right', va='bottom')

    legend_patch = mpatches.Patch(color=CLR_DCM, label='Full Model Baseline (C0)')
    ax.legend(handles=[legend_patch], loc='upper right', fontsize=14,
              framealpha=0.9, edgecolor='#cccccc', fancybox=True,
              ncol=1, markerscale=1.0, borderpad=0.8, labelspacing=0.6)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='both', width=2.0, length=6)
    # X-axis arrow
    ax.annotate('', xy=(1.02, 0), xycoords='axes fraction',
                xytext=(-0.01, 0), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    # Y-axis arrow
    ax.annotate('', xy=(0, 1.02), xycoords='axes fraction',
                xytext=(0, -0.01), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    # subplots_adjust already set; skip tight_layout to avoid margin conflict
    out = os.path.join(FIG_DIR, 'fig06_component_ablation.png')
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"  [OK] Fig.6 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7: Bootstrap 95% CI Visualization
# ══════════════════════════════════════════════════════════════════════════════
def fig07_bootstrap_ci():
    s2 = robust_data['stages']['S2']
    mean_val  = s2['mean']
    boot_mean = s2['bootstrap_mean']
    ci_lower  = s2['ci_lower']
    ci_upper  = s2['ci_upper']
    n_resamples = s2['n_resamples']

    s1_fold_accs = []
    for rep in robust_data['stages']['S1']['repeats']:
        if rep['status'] == 'completed':
            s1_fold_accs.extend(rep['fold_accs'])

    rng = np.random.default_rng(42)
    boot_samples = [np.mean(rng.choice(s1_fold_accs, size=len(s1_fold_accs), replace=True))
                    for _ in range(min(n_resamples, 5000))]
    boot_samples = np.array(boot_samples)

    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=DPI)
    n_bins = 50
    n, bins, patches = ax.hist(boot_samples * 100, bins=n_bins,
                                color=CLR_BLUE, alpha=0.75, edgecolor='white', linewidth=0.5)

    # Shade non-CI region
    for patch, left in zip(patches, bins[:-1]):
        if left/100 < ci_lower or left/100 > ci_upper:
            patch.set_facecolor('#CCCCCC')
            patch.set_alpha(0.5)

    ax.axvline(x=mean_val * 100, color=CLR_DCM, linestyle='-', linewidth=2.2,
               label=f'Mean: {mean_val*100:.2f}%')
    ax.axvline(x=ci_lower * 100, color='#E88C2A', linestyle='--', linewidth=2.0,
               label=f'CI Lower: {ci_lower*100:.2f}%')
    ax.axvline(x=ci_upper * 100, color='#E88C2A', linestyle='--', linewidth=2.0,
               label=f'CI Upper: {ci_upper*100:.2f}%')

    # Annotate CI width
    ax.annotate('', xy=(ci_upper*100, max(n)*0.6),
                xytext=(ci_lower*100, max(n)*0.6),
                arrowprops=dict(arrowstyle='<|-|>', color='#E88C2A', lw=2,
                                mutation_scale=15))
    ax.text((ci_lower+ci_upper)/2*100, max(n)*0.65,
            f'95% CI Width: {(ci_upper-ci_lower)*100:.2f}%',
            ha='center', fontsize=15, color='#E88C2A', fontweight='bold')

    ax.set_xlabel('Bootstrap Resampled Mean Accuracy (%)', fontsize=19)
    ax.set_ylabel('Frequency', fontsize=19)
    ax.set_title(f'Bootstrap 95% Confidence Interval\n({n_resamples:,} resamples, 5 seeds × 10 folds)',
                 fontsize=22, fontweight='bold', pad=15, loc='center')
    ax.legend(loc='upper right', fontsize=14,
              framealpha=0.9, edgecolor='#cccccc', fancybox=True,
              ncol=1, markerscale=1.0, borderpad=0.8, labelspacing=0.6)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='both', width=2.0, length=6)
    # X-axis arrow
    ax.annotate('', xy=(1.02, 0), xycoords='axes fraction',
                xytext=(-0.01, 0), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    # Y-axis arrow
    ax.annotate('', xy=(0, 1.02), xycoords='axes fraction',
                xytext=(0, -0.01), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig07_bootstrap_ci.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  [OK] Fig.7 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8: 10-Fold Accuracy Distribution Boxplot
# ══════════════════════════════════════════════════════════════════════════════
def fig08_fold_distribution():
    fold_accs = np.array(ablation_data['A0']['fold_accs']) * 100
    mean_val  = np.mean(fold_accs)
    med_val   = np.median(fold_accs)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=DPI)
    bp = ax.boxplot(fold_accs, vert=True, patch_artist=True,
                    widths=0.45, showfliers=False,
                    medianprops=dict(color='white', linewidth=2.5),
                    boxprops=dict(facecolor=CLR_BLUE, alpha=0.7),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    # Jitter scatter overlay
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, size=len(fold_accs))
    ax.scatter(np.ones(len(fold_accs)) + jitter, fold_accs,
               color=CLR_DCM, s=70, zorder=5, alpha=0.85, label='Per-Fold Accuracy')

    # Mark mean and median
    ax.axhline(mean_val, color=CLR_DCM, linestyle='--', linewidth=1.8,
               label=f'Mean: {mean_val:.2f}%')
    ax.axhline(med_val, color='#E88C2A', linestyle=':', linewidth=1.8,
               label=f'Median: {med_val:.2f}%')

    # ── Fold annotations: sorted by accuracy, alternating left/right to avoid overlap
    fold_data = [(f'F{i+1}', acc, jitter[i]) for i, acc in enumerate(fold_accs)]
    fold_data_sorted = sorted(fold_data, key=lambda x: x[1])

    # Assign label positions: alternating right (+) and left (-) offsets
    label_positions = []
    MIN_V_GAP = 2.5   # minimum vertical gap between labels (%)
    last_y_right = -999.0
    last_y_left  = -999.0

    for idx, (fname, acc, jit) in enumerate(fold_data_sorted):
        side = idx % 2  # 0 = right, 1 = left
        x_offset = 0.30 if side == 0 else -0.30

        if side == 0:
            y_label = max(acc, last_y_right + MIN_V_GAP)
            last_y_right = y_label
        else:
            y_label = max(acc, last_y_left + MIN_V_GAP)
            last_y_left = y_label

        ha = 'left' if side == 0 else 'right'
        label_positions.append((fname, acc, jit, x_offset, y_label, ha))

    for fname, acc, jit, x_offset, y_label, ha in label_positions:
        ax.annotate(
            f'{fname}: {acc:.1f}%',
            xy=(1 + jit * 0.5, acc),          # data point (with jitter)
            xytext=(1 + x_offset, y_label),    # label position
            fontsize=16, fontweight='bold', color='#333333', ha=ha, va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='gray', alpha=0.8),
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.8),
        )

    ax.set_xticks([1])
    ax.set_xticklabels(['DCM-DNN\n(seed=42)'])
    ax.set_ylabel('Accuracy', fontsize=19)
    ax.set_ylim(65, 100)
    ax.set_title('10-Fold Cross-Validation Accuracy Distribution\n(A0 Full Baseline, Stratified KFold, seed=42)',
                 fontsize=22, fontweight='bold', pad=15, loc='center')
    ax.legend(loc='upper right', fontsize=14,
              framealpha=0.9, edgecolor='#cccccc', fancybox=True,
              ncol=1, markerscale=1.0, borderpad=0.8, labelspacing=0.6)
    ax.spines['bottom'].set_linewidth(3.0)
    ax.spines['left'].set_linewidth(3.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(axis='both', width=2.0, length=6)
    # X-axis arrow
    ax.annotate('', xy=(1.02, 0), xycoords='axes fraction',
                xytext=(-0.01, 0), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    # Y-axis arrow
    ax.annotate('', xy=(0, 1.02), xycoords='axes fraction',
                xytext=(0, -0.01), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.10)
    out = os.path.join(FIG_DIR, 'fig08_fold_distribution.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"  [OK] Fig.8 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9: TinyML Quantization Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig09_tinyml_quantization():
    fp32  = tinyml_data['baseline_fp32']
    fp16  = tinyml_data['quantized_fp16']
    int8d = tinyml_data['quantized_int8_dynamic']

    quant_labels = ['FP32\n(Original)', 'FP16\n(Half-Precision)', 'INT8\n(Dynamic Quant.)']
    accs  = [fp32['acc_mean']*100, fp16['acc_mean']*100, int8d['acc_mean']*100]
    sizes = [fp32['model_size_kb'], fp16['model_size_kb'], int8d['model_size_kb']]
    colors_acc  = [CLR_DCM, '#4C72B0', '#55A868']
    colors_size = ['#F0A070', '#A0C4E8', '#90D490']

    x = np.arange(len(quant_labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6.5), dpi=DPI)
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, accs, width, color=colors_acc,
                    edgecolor='white', linewidth=1.2, label='Accuracy (%)', alpha=0.9)
    bars2 = ax2.bar(x + width/2, sizes, width, color=colors_size,
                    edgecolor='white', linewidth=1.2, label='Model Size (KB)', alpha=0.85)

    # Left Y-axis: Accuracy, range 75~93, bar height ~83~84%, bars fill the figure
    ax1.set_ylim(75, 93)
    # Right Y-axis: Model Size, range 0~100KB, bar height ~43~71KB
    ax2.set_ylim(0, 100)

    # Accuracy data labels (above bar top)
    for bar, acc in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold',
                 color='#333333')
        # Show delta relative to FP32 inside bar
        delta = acc - accs[0]
        if abs(delta) > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2., 79,
                     f'{delta:+.2f}%', ha='center', va='center',
                     fontsize=11, color='white', fontweight='bold')

    # Model Size data labels (above bar top)
    for bar, size in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
                 f'{size:.1f} KB', ha='center', va='bottom', fontsize=13,
                 fontweight='bold', color='#444444')

    ax1.set_xticks(x)
    ax1.set_xticklabels(quant_labels, fontsize=13)
    ax1.set_ylabel('Accuracy (%)', fontsize=16, color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=13)
    ax2.set_ylabel('Model Size (KB)', fontsize=16, color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=13)

    ax1.set_title('Quantization: Accuracy vs. Model Size\n(ESP32-S3 Edge Deployment Validation)',
                  fontsize=22, fontweight='bold', pad=15, loc='center')

    # ESP32 deployment threshold line (y=43.3KB on ax2), label placed at upper-right to avoid data labels
    ax2.axhline(y=43.3, color='#E88C2A', linestyle='--', linewidth=2.0, alpha=0.9,
                zorder=5)
    # Label text placed at upper-right corner (axes coordinates), away from all bar data labels
    ax1.text(0.985, 0.97, 'ESP32-S3 Limit (43.3 KB)',
             transform=ax1.transAxes,
             ha='right', va='top', fontsize=11, color='#E88C2A', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#E88C2A', alpha=0.90))

    # Legend: simplified to 3 items (color gradient naturally distinguishes FP32/FP16/INT8, no need to enumerate individually)
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    legend_handles = [
        # ── Color gradient represents Accuracy (left bar group)
        mpatches.Patch(
            facecolor='none', edgecolor='none',
            label='Accuracy  ▐██▒▒░░  FP32→INT8',
        ),
        mpatches.Patch(color=CLR_DCM,   alpha=0.9,  label='  Accuracy (color gradient: FP32→FP16→INT8)'),
        # ── Color gradient represents Model Size (right bar group)
        mpatches.Patch(color='#F0A070', alpha=0.85, label='  Model Size  (color gradient: FP32→FP16→INT8)'),
        # ── ESP32 threshold line
        Line2D([0], [0], color='#E88C2A', linestyle='--', linewidth=2, label='  ESP32-S3 Limit (43.3 KB)'),
    ]
    # Only keep the last 3 items (remove the first pure description item)
    legend_handles = [
        mpatches.Patch(color=CLR_DCM,   alpha=0.9,  label='Accuracy  (FP32\u2192FP16\u2192INT8 gradient)'),
        mpatches.Patch(color='#F0A070', alpha=0.85, label='Model Size  (FP32\u2192FP16\u2192INT8 gradient)'),
        Line2D([0], [0], color='#E88C2A', linestyle='--', linewidth=2, label='ESP32-S3 Limit (43.3 KB)'),
    ]
    ax1.legend(handles=legend_handles, loc='upper left', fontsize=14,
               ncol=1, framealpha=0.9, edgecolor='#cccccc', fancybox=True,
               markerscale=1.0, borderpad=0.8, labelspacing=0.6,
               handlelength=1.4)

    ax1.spines['bottom'].set_linewidth(3.0)
    ax1.spines['left'].set_linewidth(3.0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(True)
    ax1.tick_params(axis='both', width=2.0, length=6)
    ax2.spines['right'].set_linewidth(1.0)
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='y', width=1.0, length=4)
    # X-axis arrow (ax1)
    ax1.annotate('', xy=(1.02, 0), xycoords='axes fraction',
                 xytext=(-0.01, 0), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))
    # Left Y-axis arrow (ax1)
    ax1.annotate('', xy=(0, 1.02), xycoords='axes fraction',
                 xytext=(0, -0.01), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', lw=3.0, color='black'))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig09_tinyml_quantization.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"  [OK] Fig.9 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10: Multi-Dimensional Radar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig10_radar_evaluation():
    categories = ['Accuracy', 'Interpretability', 'Parameter\nEfficiency', 'Deployability', 'Theoretical\nValue']
    N = len(categories)

    dcm_scores = [84, 90, 85, 95, 95]
    mlp_scores = [82, 30, 70, 60, 20]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    dcm_vals = dcm_scores + dcm_scores[:1]
    mlp_vals = mlp_scores + mlp_scores[:1]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI, subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.08, right=0.82, top=0.85, bottom=0.05)

    ax.plot(angles, dcm_vals, 'o-', linewidth=2.5, color=CLR_DCM,
            markersize=7, label='DCM-DNN (Proposed)')
    ax.fill(angles, dcm_vals, alpha=0.25, color=CLR_DCM)

    ax.plot(angles, mlp_vals, 's--', linewidth=2.0, color=CLR_BLUE,
            markersize=7, label='Traditional MLP')
    ax.fill(angles, mlp_vals, alpha=0.12, color=CLR_BLUE)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10, color='#666666')
    ax.grid(color='#CCCCCC', linestyle='--', linewidth=0.8)

    # Value annotations
    # Per-dimension offset tuning to avoid overlapping with axis labels
    # dimensions: [Accuracy(0°), Interpretability, Parameter\nEfficiency, Deployability, Theoretical\nValue]
    # angles[0]=0 (right), angles[1]=72°(upper-right), angles[2]=144°(upper-left),
    #          angles[3]=216°(lower-left), angles[4]=288°(lower-right / Theoretical Value)
    dcm_r_offsets  = [14,  8,  8,  8,  10]   # radial offset for DCM label (outward)
    mlp_r_offsets  = [-14, -12, -12, -12, -14]  # radial offset for MLP label (inward)
    dcm_ha = ['left', 'center', 'center', 'center', 'center']
    mlp_ha = ['left', 'center', 'center', 'center', 'center']

    for i, (angle, dv, mv, cat) in enumerate(zip(angles[:-1], dcm_scores, mlp_scores, categories)):
        ax.text(angle, dv + dcm_r_offsets[i], str(dv),
                ha=dcm_ha[i], va='center',
                fontsize=14, color=CLR_DCM, fontweight='bold')
        ax.text(angle, max(mv + mlp_r_offsets[i], 5), str(mv),
                ha=mlp_ha[i], va='center',
                fontsize=14, color=CLR_BLUE, fontweight='bold')

    ax.set_title('Multi-Dimensional Evaluation\nDCM-DNN vs. Traditional MLP',
                 fontsize=22, fontweight='bold', pad=25, loc='center')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=14,
              framealpha=0.9, edgecolor='#cccccc', fancybox=True,
              ncol=1, markerscale=1.0, borderpad=0.8, labelspacing=0.6)
    out = os.path.join(FIG_DIR, 'fig10_radar_evaluation.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"  [OK] Fig.10 saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DCM-DNN Report Figure Generator")
    print("="*60)
    print(f"Output directory: {FIG_DIR}\n")

    fig01_architecture()
    fig02_param_distribution()
    fig03_baseline_comparison()
    fig04_param_efficiency()
    fig05_modality_ablation()
    fig06_component_ablation()
    fig07_bootstrap_ci()
    # fig08_fold_distribution()  # Removed: replaced by text description in paper
    fig09_tinyml_quantization()
    # fig10_radar_evaluation()   # Removed: replaced by quantitative comparison table

    print("\n" + "="*60)
    print("  Active figures generated successfully!")
    print(f"  Saved to: {FIG_DIR}")

    # Copy to paper/figures/ for PDF generation
    # fig09_tinyml is renamed to fig08_tinyml to reflect updated numbering
    copy_map = {
        'fig01_architecture.png':      'fig01_architecture.png',
        'fig02_param_distribution.png':'fig02_param_distribution.png',
        'fig03_baseline_comparison.png':'fig03_baseline_comparison.png',
        'fig04_param_efficiency.png':  'fig04_param_efficiency.png',
        'fig05_modality_ablation.png': 'fig05_modality_ablation.png',
        'fig06_component_ablation.png':'fig06_component_ablation.png',
        'fig07_bootstrap_ci.png':      'fig07_bootstrap_ci.png',
        'fig09_tinyml_quantization.png':'fig08_tinyml_quantization.png',  # renumbered
    }
    print(f"\n  Copying figures to paper/figures/ ...")
    for src_name, dst_name in copy_map.items():
        src = os.path.join(FIG_DIR, src_name)
        dst = os.path.join(PAPER_FIG_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"    Copied: {src_name} -> {dst_name}")
    print(f"  Paper figures updated: {PAPER_FIG_DIR}")
    print("="*60)
