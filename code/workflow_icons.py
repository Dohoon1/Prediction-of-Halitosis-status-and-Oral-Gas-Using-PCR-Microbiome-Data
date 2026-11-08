"""Vector workflow illustration. Miniature plots are schematic, not study results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Arc

INK = "#203541"
GREEN = "#00B83F"
ORANGE = "#FF7900"


def icon(fig, rect, kind, color=INK):
    a = fig.add_axes(rect)
    a.set(xlim=(0, 1), ylim=(0, 1))
    a.set_aspect("equal")
    a.axis("off")

    def line(x, y, **kw):
        a.plot(x, y, color=color, lw=1.35, solid_capstyle="round", **kw)

    if kind == "people":
        for x, y in [(0.25, 0.68), (0.65, 0.68), (0.45, 0.30)]:
            a.add_patch(Circle((x, y + 0.14), 0.075, fill=False, edgecolor=color, lw=1.3))
            line(
                [x - 0.13, x - 0.13, x + 0.13, x + 0.13], [y - 0.13, y + 0.015, y + 0.015, y - 0.13]
            )
            line([x, x], [y + 0.015, y - 0.2])
    elif kind == "tubes":
        for x, h in [(0.16, 0.8), (0.44, 0.8), (0.72, 0.8)]:
            a.add_patch(
                FancyBboxPatch(
                    (x, 0.15),
                    0.17,
                    0.63,
                    boxstyle="round,pad=0,rounding_size=.07",
                    facecolor="#C9E0EA",
                    edgecolor=color,
                    lw=1.3,
                )
            )
            line([x - 0.02, x + 0.19], [h, h])
            line([x + 0.03, x + 0.14], [0.40, 0.40])
        line([0.08, 0.94], [0.09, 0.09])
    elif kind == "gas":
        a.add_patch(
            Polygon(
                [
                    [0.36, 0.86],
                    [0.64, 0.86],
                    [0.64, 0.64],
                    [0.83, 0.38],
                    [0.69, 0.12],
                    [0.31, 0.12],
                    [0.17, 0.38],
                    [0.36, 0.64],
                ],
                closed=True,
                facecolor="#D7E9E9",
                edgecolor=color,
                lw=1.35,
            )
        )
        line([0.26, 0.43, 0.58, 0.74], [0.38, 0.43, 0.37, 0.41])
        for x, y, r in [(0.42, 0.55, 0.04), (0.63, 0.44, 0.025), (0.45, 0.94, 0.035)]:
            a.add_patch(Circle((x, y), r, fill=False, edgecolor=color, lw=1))
    elif kind in ["grid", "groups"]:
        for j in range(3):
            for k in range(3):
                a.add_patch(
                    Rectangle(
                        (0.1 + j * 0.28, 0.12 + k * 0.26),
                        0.22,
                        0.2,
                        facecolor="#D3E5EA" if (j + k) % 2 else "#FFFFFF",
                        edgecolor=color,
                        lw=1,
                    )
                )
        if kind == "groups":
            line([0.1, 0.85], [0.91, 0.91])
    elif kind == "sliders":
        for x, y in [(0.2, 0.3), (0.5, 0.7), (0.8, 0.48)]:
            line([x, x], [0.12, 0.90])
            a.add_patch(Circle((x, y), 0.075, facecolor="white", edgecolor=color, lw=1.3))
    elif kind == "bars":
        line([0.1, 0.1, 0.95], [0.93, 0.1, 0.1])
        for x, h in [(0.22, 0.43), (0.43, 0.75), (0.64, 0.57), (0.83, 0.28)]:
            a.add_patch(Rectangle((x, 0.12), 0.12, h, facecolor="#C9DDD2", edgecolor=color, lw=1))
    elif kind in ["network", "neural"]:
        pts = [
            (0.13, 0.25),
            (0.13, 0.75),
            (0.5, 0.18),
            (0.5, 0.50),
            (0.5, 0.82),
            (0.88, 0.3),
            (0.88, 0.7),
        ]
        for i in [0, 1]:
            for j in [2, 3, 4]:
                line([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], alpha=0.7)
        for i in [2, 3, 4]:
            for j in [5, 6]:
                line([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], alpha=0.7)
        for x, y in pts:
            a.add_patch(Circle((x, y), 0.055, facecolor="white", edgecolor=color, lw=1.3))
    elif kind == "adjust":
        line([0.5, 0.5], [0.12, 0.9])
        line([0.14, 0.86], [0.75, 0.75])
        line([0.32, 0.68], [0.1, 0.1])
        for x in [0.22, 0.78]:
            line([x - 0.13, x, x + 0.13], [0.33, 0.75, 0.33])
            a.add_patch(Arc((x, 0.33), 0.29, 0.16, theta1=180, theta2=360, color=color, lw=1.2))
    elif kind in ["linear", "scatter", "roc"]:
        line([0.12, 0.12, 0.95], [0.94, 0.12, 0.12])
        if kind == "roc":
            line([0.12, 0.2, 0.31, 0.52, 0.74, 0.95], [0.12, 0.46, 0.69, 0.85, 0.92, 0.95])
            line([0.12, 0.95], [0.12, 0.95], ls=":", alpha=0.4)
        else:
            line([0.18, 0.91], [0.20, 0.85])
            for x, y in [
                (0.22, 0.27),
                (0.33, 0.30),
                (0.46, 0.56),
                (0.6, 0.55),
                (0.74, 0.76),
                (0.86, 0.82),
            ]:
                a.scatter(x, y, s=7, color=color)
    elif kind == "folds":
        for j in range(5):
            for k in range(3):
                a.add_patch(
                    Rectangle(
                        (0.02 + j * 0.19, 0.16 + k * 0.24),
                        0.16,
                        0.2,
                        facecolor="#EF7E72" if j == k else "#C8E6D3",
                        edgecolor="white",
                        lw=0.8,
                    )
                )
        a.add_patch(
            FancyArrowPatch(
                (0.1, 0.95), (0.89, 0.95), arrowstyle="-|>", mutation_scale=8, color=color, lw=1
            )
        )
    elif kind == "distribution":
        x = np.linspace(0.05, 0.95, 100)
        v = np.exp(-(((x - 0.5) / 0.18) ** 2))
        a.fill_between(x, 0.12, 0.12 + 0.7 * v, color="#D8E7EC")
        line(x, 0.12 + 0.7 * v)
        line([0.06, 0.95], [0.12, 0.12])
    elif kind == "seeds":
        for x, y in [(0.25, 0.4), (0.5, 0.63), (0.75, 0.48)]:
            a.scatter(x, y, s=30, color=ORANGE, marker="s")
    return a


def draw_workflow():
    fig = plt.figure(figsize=(7.7, 7.4))
    a = fig.add_axes([0, 0, 1, 1])
    a.set(xlim=(0, 1), ylim=(0, 1))
    a.axis("off")

    def card(x, y, w, h, title, fill):
        a.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.004,rounding_size=.012",
                facecolor=fill,
                edgecolor="#91A6B1",
                lw=0.85,
            )
        )
        a.text(
            x + w / 2,
            y + h - 0.023,
            title,
            ha="center",
            va="top",
            fontsize=9.3,
            fontweight="bold",
            color="black",
        )
        a.plot([x + 0.012, x + w - 0.012], [y + h - 0.062] * 2, color="#B3C2CA", lw=0.7)

    def text(x, y, s, size=8.4, ha="left", weight="normal"):
        a.text(
            x,
            y,
            s,
            fontsize=size,
            ha=ha,
            va="top",
            linespacing=1.35,
            color="black",
            fontweight=weight,
        )

    def arrow(start, end):
        a.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=12, lw=1.4, color="#526E7E"
            )
        )

    card(0.024, 0.55, 0.303, 0.42, "DATA ACQUISITION", "#EDF2F7")
    card(0.358, 0.55, 0.303, 0.42, "INPUTS AND GROUPS", "#ECF5F4")
    card(0.692, 0.55, 0.284, 0.42, "ASSOCIATION ANALYSES", "#F0F4EA")
    icon(fig, [0.043, 0.825, 0.075, 0.080], "people")
    text(0.13, 0.879, "102 participants\nComplete records")
    icon(fig, [0.043, 0.700, 0.075, 0.085], "tubes")
    text(0.13, 0.768, "Unstimulated saliva\n16 targeted qPCR assays")
    icon(fig, [0.043, 0.581, 0.075, 0.085], "gas")
    text(0.13, 0.648, "Mouth air (ppb)\nH$_2$S and CH$_3$SH")
    icon(fig, [0.373, 0.825, 0.064, 0.077], "groups")
    text(0.445, 0.878, "Complaint + VSC criterion\n30 positive | 72 comparison", 8.1)
    icon(fig, [0.373, 0.694, 0.064, 0.080], "grid")
    text(0.445, 0.767, "16 PCR species\nAge, sex, smoking\nand oral hygiene", 8.2)
    icon(fig, [0.373, 0.575, 0.064, 0.075], "sliders")
    text(0.445, 0.641, "Preprocessing fitted\nwithin training folds", 8.2)
    icon(fig, [0.705, 0.825, 0.064, 0.077], "bars")
    text(0.78, 0.878, "Panel diversity\nand species tests", 8.4)
    icon(fig, [0.705, 0.700, 0.064, 0.077], "network")
    text(0.78, 0.765, "Correlation clusters\nand sensitivity analyses", 8.0)
    icon(fig, [0.705, 0.579, 0.064, 0.077], "adjust")
    text(0.78, 0.644, "Clinical adjustment\nand multiplicity control", 8.1)
    arrow((0.329, 0.758), (0.354, 0.758))
    arrow((0.663, 0.758), (0.688, 0.758))
    card(0.024, 0.045, 0.637, 0.425, "MODELS AND INTERNAL VALIDATION", "#F0F6F0")
    card(0.692, 0.045, 0.284, 0.425, "PERFORMANCE", "#F2F2F7")
    arrow((0.51, 0.548), (0.51, 0.476))
    arrow((0.834, 0.548), (0.834, 0.476))
    icon(fig, [0.047, 0.327, 0.08, 0.080], "linear", GREEN)
    text(0.145, 0.396, "LINEAR MODELS", 9, weight="bold")
    text(0.145, 0.369, "Logistic and Ridge\n22 specifications / outcome", 8.0)
    icon(fig, [0.047, 0.210, 0.08, 0.080], "neural", ORANGE)
    text(0.145, 0.279, "TRANSFORMER", 9, weight="bold")
    text(0.145, 0.252, "19 configurations / outcome\nExploratory comparison", 8.0)
    a.plot([0.389, 0.389], [0.19, 0.405], color="#BDCEC2", lw=0.7)
    icon(fig, [0.439, 0.306, 0.174, 0.095], "folds")
    text(0.524, 0.294, "Outer 5-fold × 10 repeats\nInner 3-fold tuning", 8.3, ha="center")
    arrow((0.370, 0.36), (0.428, 0.36))
    arrow((0.370, 0.24), (0.428, 0.33))
    arrow((0.663, 0.27), (0.688, 0.27))
    text(
        0.342,
        0.163,
        "Scaling, clustering and selection are repeated\ninside the relevant training folds",
        8.5,
        ha="center",
    )
    text(
        0.342,
        0.096,
        "Clinical-only and Dummy references\nNested species selector for linear models",
        8.0,
        ha="center",
    )
    icon(fig, [0.710, 0.314, 0.106, 0.085], "roc")
    icon(fig, [0.851, 0.314, 0.106, 0.085], "scatter", GREEN)
    text(0.762, 0.306, "Halitosis\nAUROC", 8.2, ha="center")
    text(0.904, 0.306, "Gas\nR$^2$, MAE, RMSE", 8.0, ha="center")
    a.plot([0.708, 0.96], [0.247, 0.247], color="#C4C6D2", lw=0.7)
    icon(fig, [0.710, 0.179, 0.064, 0.062], "distribution")
    text(0.786, 0.231, "Linear: 95% CIs\n10,000 bootstraps", 7.9)
    icon(fig, [0.710, 0.117, 0.064, 0.057], "seeds")
    text(0.786, 0.164, "Transformer: SD\nacross three seeds", 7.9)
    text(0.834, 0.089, "100 label permutations\nSeparate 5-fold × 2 repeats", 7.5, ha="center")
    return fig
