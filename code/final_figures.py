"""Verify the public PCR dataset and rebuild revision figures.

Predictive performance is read from frozen final R2 reference JSON/CSV tables.
This script does not refit predictive models or reconstruct participant OOF data.
"""

from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy import stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    from common import ROOT, load_dataset, short_taxon, correlation_components, read_config

    parser.add_argument("--config", type=Path, default=ROOT / "config/revision_main.json")
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--reference", type=Path, default=ROOT / "reference/figure3_source_values.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--formats", nargs="+", choices=["png", "pdf", "tif"], default=["png", "pdf", "tif"]
    )
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty. Choose a new --output.")
    from pcr_statistics import panel_indices, representations, _associations

    out = args.output.resolve()
    figdir = out / "figures"
    auditdir = out / "analysis"
    figdir.mkdir(parents=True, exist_ok=True)
    auditdir.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)
    dataset = load_dataset(config, args.input)
    frame = dataset.frame
    columns = dataset.pcr_columns
    x = frame[columns].to_numpy(float)
    y = frame.Halitosis.to_numpy(int)
    spaces = representations(x, {})
    names = {
        c: short_taxon(c).replace("L. casei", "L. casei").replace("F. alocis", "F. alocis")
        for c in columns
    }

    def adjusted(p):
        p = np.asarray(p, float)
        order = np.argsort(p)
        q = np.empty(len(p))
        q[order] = np.minimum(
            1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        )
        return np.minimum(1, p * len(p)), q

    species = []
    for scale, matrix in spaces.items():
        for j, col in enumerate(columns):
            a, b = matrix[y == 0, j], matrix[y == 1, j]
            species.append(
                {
                    "scale": scale,
                    "species": names[col],
                    "column": col,
                    "mean0": a.mean(),
                    "sd0": a.std(ddof=1),
                    "mean1": b.mean(),
                    "sd1": b.std(ddof=1),
                    "difference": b.mean() - a.mean(),
                    "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue,
                    "mw_p": stats.mannwhitneyu(
                        a, b, alternative="two-sided", method="asymptotic"
                    ).pvalue,
                }
            )
    species = pd.DataFrame(species)
    for scale in species.scale.unique():
        mask = species.scale.eq(scale)
        for test in ["welch", "mw"]:
            species.loc[mask, test + "_bonferroni"], species.loc[mask, test + "_bh"] = adjusted(
                species.loc[mask, test + "_p"]
            )
    species.to_csv(auditdir / "verified_species_tests.csv", index=False)
    total = frame.PCR_panel_total.to_numpy()
    total_p = float(stats.ttest_ind(total[y == 0], total[y == 1], equal_var=False).pvalue)
    associations = _associations(frame, columns, [], config, auditdir)
    div = panel_indices(x)
    divrows = []
    for name, v in zip(["Richness", "Shannon", "Evenness"], div):
        a, b = v[y == 0], v[y == 1]
        divrows.append(
            {
                "index": name,
                "mean0": a.mean(),
                "sd0": a.std(ddof=1),
                "mean1": b.mean(),
                "sd1": b.std(ddof=1),
                "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue,
                "mw_p": stats.mannwhitneyu(
                    a, b, alternative="two-sided", method="asymptotic"
                ).pvalue,
            }
        )
    pd.DataFrame(divrows).to_csv(auditdir / "verified_panel_diversity.csv", index=False)
    vsc_only = (frame.VSCs_ppb >= frame.Sex.map({"F": 65.79, "M": 79.94})).to_numpy()
    raw = spaces["raw"]
    corr = pd.DataFrame(raw, columns=columns).corr()
    clusters, edges = correlation_components(corr, 0.4, absolute=False, inclusive=False)
    clustercounts = {
        str(t): len(
            next(
                iter(correlation_components(corr, t, absolute=False, inclusive=False)[0].values()),
                [],
            )
        )
        for t in [0.3, 0.4, 0.5, 0.6]
    }
    audit = {
        "participants": len(frame),
        "group_counts": {str(k): int((y == k).sum()) for k in [0, 1]},
        "group_definition_confirmed_by_author": "self-reported complaint and sex-specific VSC reference criterion",
        "labels_changed": False,
        "vsc_only_positive": int(vsc_only.sum()),
        "vsc_only_positive_primary_negative": int((vsc_only & (y == 0)).sum()),
        "primary_positive_below_vsc_reference": int(((~vsc_only) & (y == 1)).sum()),
        "vsc_sum_mismatches": dataset.audit["vsc_sum_mismatches"],
        "missing_analysis_values": dataset.audit["missing_by_column"],
        "panel_total_welch_p": total_p,
        "raw_pearson_largest_cluster_sizes": clustercounts,
        "public_workbook_sha256": hashlib.sha256(dataset.source_path.read_bytes()).hexdigest(),
        "public_repository_commit": "51a1e0ad7f8506cd2e537644bf8ae6535a2d5b7c",
        "predictive_source": "Frozen final R2 Supplementary Tables 2 and 8; plotted values retain their reported precision",
        "predictive_models_refitted_in_this_finalization": False,
        "adjusted_association_check": associations,
    }
    (auditdir / "revision_verification.json").write_text(
        json.dumps(audit, indent=2), encoding="utf8"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    BLUE = "#31688E"
    RED = "#C44E52"
    ORANGE = "#FF7900"
    GREEN = "#00B83F"
    GROUP_BLUE = "#003C8F"
    GROUP_RED = "#B00020"

    def save(fig, stem):
        for ext in args.formats:
            kwargs = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
            target = figdir / (stem + "." + ext)
            temporary = figdir / (stem + ".tmp." + ext)
            fig.savefig(temporary, dpi=args.dpi, bbox_inches="tight", facecolor="white", **kwargs)
            temporary.replace(target)
        plt.close(fig)

    def clean_name(s):
        s = (
            str(s)
            .replace("Single ", "")
            .replace("Largest cluster", "Cluster 1")
            .replace("Second cluster", "Cluster 2")
        )
        return (
            s.replace("Nested best-species selector", "Nested species selector")
            .replace("Dummy prevalence", "Dummy reference")
            .replace("Dummy training mean", "Dummy reference")
        )

    def first_number(s):
        return float(re.search(r"[-−]?\d*\.?\d+", str(s)).group().replace("−", "-"))

    def interval(s):
        nums = re.findall(r"[-−]?\d*\.?\d+", str(s))
        return tuple(float(z.replace("−", "-")) for z in nums[-2:])

    frozen = json.loads(args.reference.read_text(encoding="utf8"))
    modelorder, linear, transformer = frozen["order"], frozen["linear"], frozen["transformer"]
    if (
        len(modelorder) != 22
        or any(len(v) != 22 for v in linear.values())
        or any(len(v) != 19 for v in transformer.values())
    ):
        raise ValueError(
            "Reference model inventory must be 22 Linear and 19 Transformer per outcome."
        )
    (auditdir / "figure3_source_values.json").write_text(
        json.dumps({"order": modelorder, "linear": linear, "transformer": transformer}, indent=2),
        encoding="utf8",
    )

    def performance_axis(ax, target, show_labels=True, letter=None):
        isclass = target == "Halitosis"
        low, high = (0.25, 0.88) if isclass else (-0.5, 0.55)
        ax.set_xlim(low, high)
        ax.set_ylim(len(modelorder) - 0.4, -0.8)
        for i, name in enumerate(modelorder):
            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color="#F3F5F7", zorder=0)
            m, lo, hi = linear[target][name]
            yy = i - 0.13
            if low <= m <= high:
                ax.errorbar(
                    m,
                    yy,
                    xerr=[[m - max(lo, low)], [min(hi, high) - m]],
                    fmt="o",
                    color=GREEN,
                    ecolor=GREEN,
                    markersize=4.2,
                    capsize=2,
                    elinewidth=0.9,
                    zorder=3,
                )
                if lo < low:
                    ax.plot(low + 0.009, yy, "<", color=GREEN, ms=4, clip_on=False)
                if hi > high:
                    ax.plot(high - 0.009, yy, ">", color=GREEN, ms=4, clip_on=False)
            else:
                edge = low + 0.02 if m < low else high - 0.02
                ax.plot(edge, yy, "<" if m < low else ">", color=GREEN, ms=6)
                ax.text(
                    edge + 0.035 if m < low else edge - 0.035,
                    yy - 0.08,
                    f"{m:.2f}",
                    fontsize=6.5,
                    ha="left" if m < low else "right",
                    va="bottom",
                    color="black",
                )
            if name in transformer[target]:
                tm, sd = transformer[target][name]
                ax.errorbar(
                    tm,
                    i + 0.13,
                    xerr=sd,
                    fmt="s",
                    color=ORANGE,
                    ecolor=ORANGE,
                    markersize=4,
                    capsize=2,
                    elinewidth=0.85,
                    zorder=3,
                )
        ax.axvline(0.5 if isclass else 0, color="#555555", ls="--", lw=0.8, zorder=1)
        ax.set_yticks(range(len(modelorder)))
        ax.set_yticklabels(modelorder, fontsize=8.4)
        for tick in ax.get_yticklabels():
            if re.match(r"^[A-Z]\. ", tick.get_text()):
                tick.set_fontstyle("italic")
        ax.tick_params(axis="y", labelleft=show_labels)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.set_xticks([0.3, 0.5, 0.7, 0.85] if isclass else [-0.5, 0, 0.5])
        ax.tick_params(axis="x", labelsize=7.5)
        title = {
            "Halitosis": "VSC-confirmed\nhalitosis",
            "H2S": r"H$_2$S",
            "CH3SH": r"CH$_3$SH",
            "Total VSC": "Total VSC",
        }[target]
        ax.set_title(title, fontsize=9.5, pad=11)
        ax.set_xlabel("AUROC" if isclass else r"$R^2$", fontsize=9)
        if letter:
            ax.text(0, 1.075, letter, transform=ax.transAxes, fontweight="bold", fontsize=14)

    fig, axes = plt.subplots(1, 4, figsize=(9.4, 8.7), sharey=True)
    for ax, target, letter in zip(axes, ["Halitosis", "H2S", "CH3SH", "Total VSC"], "ABCD"):
        performance_axis(ax, target, show_labels=ax is axes[0], letter=letter)
    fig.subplots_adjust(left=0.24, right=0.985, bottom=0.14, top=0.88, wspace=0.2)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=GREEN,
            label="Linear model with 95% participant-bootstrap CI",
            lw=0.9,
        ),
        Line2D(
            [0], [0], marker="s", color=ORANGE, label="Transformer with training-seed SD", lw=0.9
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.57, 0.045),
        frameon=False,
        fontsize=8,
        ncol=1,
    )
    fig.text(
        0.24,
        0.012,
        "Arrowheads indicate values or CI limits outside the displayed range.\nClinical variables are included in all non-Dummy models.",
        fontsize=7.5,
    )
    save(fig, "Figure_3")
    for k, target in enumerate(["Halitosis", "H2S", "CH3SH", "Total VSC"], 1):
        fig, ax = plt.subplots(figsize=(8.0, 8.0))
        performance_axis(ax, target, True)
        fig.subplots_adjust(left=0.30, right=0.97, top=0.92, bottom=0.13)
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.64, 0.025),
            frameon=False,
            fontsize=8,
        )
        save(fig, f"Supplementary_Figure_3{chr(64+k)}")

    # Figure 2 uses the tested log10 scale directly, with all 16 adjusted p values.
    logstats = species[species.scale == "log10"].set_index("column")
    order = sorted(columns, key=lambda c: float(logstats.loc[c, "welch_p"]))
    fig = plt.figure(figsize=(9.1, 11.2))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.22, 1], width_ratios=[1.45, 1], hspace=0.37, wspace=0.42
    )
    ax = fig.add_subplot(gs[0, :])
    rng = np.random.default_rng(20261107)
    for i, col in enumerate(order):
        j = columns.index(col)
        for g, color, offset in [(0, GROUP_BLUE, -0.18), (1, GROUP_RED, 0.18)]:
            vals = spaces["log10"][y == g, j]
            bp = ax.boxplot(
                vals,
                positions=[i + offset],
                vert=False,
                widths=0.29,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 0.8},
                whiskerprops={"linewidth": 0.65, "color": "#555555"},
                capprops={"linewidth": 0.65, "color": "#555555"},
            )
            bp["boxes"][0].set(
                facecolor=matplotlib.colors.to_rgba(color, 0.58), edgecolor=color, linewidth=0.85
            )
            ax.scatter(
                vals,
                i + offset + rng.uniform(-0.075, 0.075, len(vals)),
                s=4,
                color=color,
                alpha=0.82,
                zorder=3,
                linewidths=0,
            )
        p = float(logstats.loc[col, "welch_bonferroni"])
        label = f"{p:.4f}"
        ax.text(
            8.1,
            i,
            label,
            fontsize=8.5,
            ha="right",
            va="center",
            color="black",
            fontweight="bold" if p < 0.05 else "normal",
        )
    ax.set_yticks(range(16))
    ax.set_yticklabels([names[c] for c in order], fontsize=9)
    for tick in ax.get_yticklabels():
        tick.set_fontstyle("italic")
    ax.set_ylim(15.7, -1.1)
    ax.set_xlim(-0.1, 8.2)
    ax.set_xticks(range(8))
    ax.set_xlabel(r"log$_{10}$(copies/mL + 1)", fontsize=10)
    ax.text(8.1, -0.85, "Adjusted P", fontsize=8.5, ha="right", fontweight="bold")
    ax.text(-0.2, 1.055, "A", transform=ax.transAxes, fontsize=15, fontweight="bold")
    ax.set_title(
        "Species concentrations and Bonferroni-adjusted Welch tests", fontsize=10.5, pad=34
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="s", color=GROUP_BLUE, label="Non-halitosis (n = 72)", lw=0),
            Line2D(
                [0],
                [0],
                marker="s",
                color=GROUP_RED,
                label="VSC-confirmed halitosis (n = 30)",
                lw=0,
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.03, 1.01),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.5)
    heat = fig.add_subplot(gs[1, 0])
    heatorder = (
        list(clusters["Cluster_1"])
        + list(clusters["Cluster_2"])
        + [c for c in columns if all(c not in v for v in clusters.values())]
    )
    mat = corr.loc[heatorder, heatorder].to_numpy()
    im = heat.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    heat.set_xticks(range(16))
    heat.set_yticks(range(16))
    heat.set_xticklabels([names[c] for c in heatorder], rotation=60, ha="right", fontsize=6.8)
    heat.set_yticklabels([names[c] for c in heatorder], fontsize=6.8)
    for tick in [*heat.get_xticklabels(), *heat.get_yticklabels()]:
        tick.set_fontstyle("italic")
    heat.tick_params(length=0)
    heat.set_title("Pearson correlation map", fontsize=10, pad=12)
    heat.text(-0.41, 1.10, "B", transform=heat.transAxes, fontweight="bold", fontsize=15)
    for boundary in [12.5, 14.5]:
        heat.axhline(boundary, color="black", lw=0.8)
        heat.axvline(boundary, color="black", lw=0.8)
    cb = fig.colorbar(im, ax=heat, orientation="horizontal", fraction=0.045, pad=0.44, shrink=0.8)
    cb.set_label("Pearson r", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    netgrid = gs[1, 1].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.32)
    for k, label in [(0, "Cluster_1"), (1, "Cluster_2")]:
        net = fig.add_subplot(netgrid[k, 0])
        members = clusters[label]
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(members), endpoint=False)
        points = {c: np.array([np.cos(t), np.sin(t)]) for c, t in zip(members, angles)}
        for edge in edges:
            if edge["source"] in members and edge["target"] in members:
                a, b = points[edge["source"]], points[edge["target"]]
                net.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    color="#566775",
                    alpha=0.65,
                    lw=0.4 + 5 * (edge["r"] - 0.4),
                    zorder=1,
                )
        for c, xy in points.items():
            net.scatter(
                *xy,
                s=145 if k == 0 else 240,
                color="#C1DFEA" if k == 0 else "#F4D5B4",
                edgecolor="white",
                zorder=2,
            )
            ha = "left" if xy[0] > 0.2 else "right" if xy[0] < -0.2 else "center"
            va = "center" if abs(xy[0]) > 0.2 else "bottom" if xy[1] > 0.1 else "top"
            net.text(
                *(xy * 1.15), names[c], fontsize=6.2, fontstyle="italic", ha=ha, va=va, zorder=3
            )
        net.set_xlim(-1.7, 1.7)
        net.set_ylim(-1.55, 1.5)
        net.set_aspect("equal")
        net.axis("off")
        net.set_title(
            f"Cluster {k+1} ({len(members)} species)", fontsize=9, pad=19 if k == 0 else 6
        )
        if k == 0:
            net.text(
                0.5,
                1.02,
                "Thicker lines = stronger Pearson r",
                transform=net.transAxes,
                ha="center",
                fontsize=6.3,
            )
        net.text(
            -0.12,
            1.08 if k == 0 else 1.30,
            "C" if k == 0 else "D",
            transform=net.transAxes,
            fontweight="bold",
            fontsize=15,
        )
    fig.subplots_adjust(left=0.20, right=0.97, top=0.95, bottom=0.09)
    save(fig, "Figure_2")

    # Icon-based vector workflow, with schematic miniature plots.
    from workflow_icons import draw_workflow

    fig = draw_workflow()
    save(fig, "Figure_1")

    # Renumbered supplementary gas distributions and panel composition.
    rng = np.random.default_rng(20261108)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.5))
    for ax, c, title in zip(
        axes, ["H2S_ppb", "CH3SH_ppb", "VSCs_ppb"], [r"H$_2$S", r"CH$_3$SH", "Total VSC"]
    ):
        vals = [frame.loc[y == g, c].to_numpy() for g in [0, 1]]
        p = stats.ttest_ind(*vals, equal_var=False).pvalue
        mantissa, exponent = f"{p:.2e}".split("e")
        b = ax.boxplot(vals, positions=[0, 1], widths=0.5, patch_artist=True, showfliers=False)
        for patch, color in zip(b["boxes"], [BLUE, RED]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for g, color in [(0, BLUE), (1, RED)]:
            ax.scatter(
                g + rng.uniform(-0.13, 0.13, (y == g).sum()), vals[g], s=10, color=color, alpha=0.6
            )
        ax.set_xticks([0, 1], ["Non-halitosis", "Halitosis"], fontsize=8)
        ax.set_ylabel("Concentration (ppb)", fontsize=8)
        ax.set_title(title, fontsize=11, pad=28)
        ax.text(
            0.5,
            1.02,
            rf"Welch $P = {mantissa} \times 10^{{{int(exponent)}}}$",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    save(fig, "Supplementary_Figure_1")
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.4), gridspec_kw={"hspace": 0.32})
    colors = plt.get_cmap("tab20").colors[:16]
    for ax, g in zip(axes, [0, 1]):
        a = raw[y == g]
        bottom = np.zeros(len(a))
        for j, c in enumerate(columns):
            ax.bar(
                np.arange(len(a)),
                a[:, j] / 1e6,
                bottom=bottom,
                color=colors[j],
                width=0.95,
                label=names[c],
            )
            bottom += a[:, j] / 1e6
        ax.set_title(
            f'{"Non-halitosis" if g==0 else "VSC-confirmed halitosis"} (n = {len(a)})', fontsize=10
        )
        ax.set_ylabel("Panel abundance\n(million copies/mL)", fontsize=8)
        ax.set_xticks([])
        ax.set_xlabel("Participants in original data order", fontsize=8)
    fig.subplots_adjust(bottom=0.25, left=0.09, right=0.98, top=0.94)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.53, 0.015),
    )
    save(fig, "Supplementary_Figure_2")
    print(
        f"Created final figures: {figdir}; model performance comes from frozen reported summaries, not new fits."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
