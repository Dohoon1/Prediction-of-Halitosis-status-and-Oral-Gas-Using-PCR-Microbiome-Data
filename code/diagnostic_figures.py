"""Render saved PCR study results without loading inputs or fitting models.

This module deliberately does not import common.py or any modeling library.
Every statistical estimate, ROC coordinate, confidence interval, and error used
below must already exist in the result files produced by 01_analysis.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GROUP_COLORS = {"0": "#2878A1", "1": "#C95B4C"}
REQUIRED = {
    "sample_flow.csv": ["stage", "n"],
    "taxon_summary.csv": ["taxon", "group", "n", "median", "q25", "q75"],
    "abundance_long.csv": ["participant_id", "group", "taxon", "log10_copies_plus1"],
    "correlation_edges.csv": ["source", "target", "r", "cluster"],
}
CLASSIFICATION = {
    "classification_summary.csv": [
        "model",
        "algorithm",
        "fold_mean_auc",
        "fold_sd_auc",
        "pooled_oof_auc",
        "auc_ci_low",
        "auc_ci_high",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "average_precision",
    ],
    "classification_oof.csv": ["participant_id", "model", "algorithm", "fold", "y_true", "y_prob"],
    "classification_roc.csv": ["model", "algorithm", "fpr", "tpr"],
}
REGRESSION = {
    "regression_summary.csv": [
        "target",
        "model",
        "algorithm",
        "fold_mean_r2",
        "fold_sd_r2",
        "pooled_oof_r2",
        "r2_ci_low",
        "r2_ci_high",
        "mae",
        "rmse",
    ],
    "regression_oof.csv": [
        "participant_id",
        "target",
        "model",
        "algorithm",
        "fold",
        "y_true",
        "y_pred",
        "absolute_error",
    ],
}


def _read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Required result file is missing: {path}")
    try:
        frame = pd.read_csv(path, dtype={"participant_id": "string"})
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot read saved results: {path.name}: {exc}") from exc
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    return frame


def _numbers(frame: pd.DataFrame, columns: list[str], name: str, allow_nan: bool = False) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=float)
        invalid = np.isinf(values).any() or (not allow_nan and np.isnan(values).any())
        if invalid:
            raise ValueError(f"{name}: invalid saved numeric value in {column}.")


def _group_key(value) -> str:
    text = str(value).strip()
    return text[:-2] if text in {"0.0", "1.0"} else text


def _group_label(value) -> str:
    key = _group_key(value)
    return {"0": "Without halitosis", "1": "With halitosis"}.get(key, key)


def _short_taxon(name: str) -> str:
    clean = re.sub(r"^PCR[_\s]+", "", str(name).strip(), flags=re.IGNORECASE)
    parts = clean.replace("_", " ").split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else str(name)


def _label(model, algorithm) -> str:
    return f"{model} ({algorithm})"


def _safe_name(value) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")[:130] or "result"


def _flow_layout(frame: pd.DataFrame):
    """Lay out only the saved parent relationships; never infer sample flow."""
    if frame.empty:
        raise ValueError("sample_flow.csv has no saved sample-flow rows.")
    tree_columns = {"stage_id", "parent_stage_id"} & set(frame.columns)
    if tree_columns and len(tree_columns) != 2:
        raise ValueError("sample_flow.csv must supply both stage_id and parent_stage_id.")
    if tree_columns:
        stage_ids = frame["stage_id"].fillna("").astype(str).str.strip().tolist()
        parent_ids = frame["parent_stage_id"].fillna("").astype(str).str.strip().tolist()
        if "" in stage_ids or len(set(stage_ids)) != len(stage_ids):
            raise ValueError("sample_flow.csv stage_id values must be nonempty and unique.")
    else:
        stage_ids = [f"row_{i}" for i in range(len(frame))]
        parent_ids = [""] + stage_ids[:-1]
    parents = dict(zip(stage_ids, parent_ids))
    unknown = set(parent_ids) - set(stage_ids) - {""}
    if unknown:
        raise ValueError(
            "sample_flow.csv refers to an unknown parent stage: " + ", ".join(sorted(unknown))
        )
    depths = {}

    def depth(stage_id, ancestors):
        if stage_id in ancestors:
            raise ValueError("sample_flow.csv contains a cycle in its parent links.")
        if stage_id not in depths:
            parent = parents[stage_id]
            depths[stage_id] = depth(parent, ancestors | {stage_id}) + 1 if parent else 0
        return depths[stage_id]

    for stage_id in stage_ids:
        depth(stage_id, set())
    children = {stage_id: [] for stage_id in stage_ids}
    for stage_id, parent in parents.items():
        if parent:
            children[parent].append(stage_id)
    leaves = [stage_id for stage_id in stage_ids if not children[stage_id]]
    xs = {stage_id: (i + 0.5) / len(leaves) for i, stage_id in enumerate(leaves)}

    def x_position(stage_id):
        if stage_id not in xs:
            xs[stage_id] = sum(x_position(child) for child in children[stage_id]) / len(
                children[stage_id]
            )
        return xs[stage_id]

    levels = max(depths.values()) + 1
    positions = {
        stage_id: (x_position(stage_id), levels - depths[stage_id] - 0.5) for stage_id in stage_ids
    }
    edges = [(parent, stage_id) for stage_id, parent in parents.items() if parent]
    return stage_ids, positions, edges, levels, len(leaves)


class Renderer:
    def __init__(self, results: Path, output: Path, dpi: int, formats: list[str]):
        self.results, self.output, self.dpi, self.formats = results, output, dpi, formats
        self.written: list[str] = []
        self.skipped: list[dict[str, str]] = []
        self.notes: list[str] = []
        self.captions: dict[str, str] = {}
        manifest_path = results / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"Missing manifest.json: run 01_analysis.py to completion first ({results})."
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = (
            self.manifest.get("status") == "complete" or self.manifest.get("complete") is True
        )
        if not complete or self.manifest.get("complete") is False:
            raise ValueError(
                "The analysis manifest is not complete. Incomplete or mixed results cannot be rendered."
            )
        self.tables = {name: _read_table(results / name, cols) for name, cols in REQUIRED.items()}
        for required_json in ["clusters.json", "data_validation.json"]:
            path = results / required_json
            if not path.is_file():
                raise ValueError(f"Required result file is missing: {path}")
        self.clusters = json.loads((results / "clusters.json").read_text(encoding="utf-8"))
        if not isinstance(self.clusters, dict) or not all(
            isinstance(v, list) for v in self.clusters.values()
        ):
            raise ValueError("clusters.json must map cluster names to saved taxon lists.")
        self.validation = json.loads((results / "data_validation.json").read_text(encoding="utf-8"))
        correlation_path = results / "correlation.csv"
        if not correlation_path.is_file():
            raise ValueError(f"Required result file is missing: {correlation_path}")
        self.correlation = pd.read_csv(correlation_path, index_col=0)
        self.correlation.index = self.correlation.index.astype(str)
        self.correlation.columns = self.correlation.columns.astype(str)
        if list(self.correlation.index) != list(self.correlation.columns):
            raise ValueError(
                "correlation.csv must be square with identical ordered row and column names."
            )
        _numbers(
            self.correlation, self.correlation.columns.tolist(), "correlation.csv", allow_nan=True
        )
        self.classification = self._optional_group(CLASSIFICATION, "Classification figures")
        self.regression = self._optional_group(REGRESSION, "Regression figures")
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "figure_manifest.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "source_results": str(self.results.resolve()),
                    "analysis_performed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "savefig.facecolor": "white",
            }
        )

    def _optional_group(self, contract: dict, purpose: str) -> dict:
        present = [name for name in contract if (self.results / name).is_file()]
        if not present:
            self.skipped.append(
                {"figure": purpose, "reason": "No saved model outputs; descriptive-only results."}
            )
            return {}
        missing = [name for name in contract if name not in present]
        if missing:
            raise ValueError(
                f"Incomplete {purpose.lower()}: missing {', '.join(missing)}. Partial model outputs are not rendered."
            )
        frames = {}
        for name, columns in contract.items():
            path = self.results / name
            if not path.read_text(encoding="utf-8-sig").strip():
                frames[name] = None
            else:
                frames[name] = _read_table(path, columns)
        if all(frame is None or frame.empty for frame in frames.values()):
            self.skipped.append(
                {"figure": purpose, "reason": "The analysis saved no rows for this model task."}
            )
            return {}
        empty = [name for name, frame in frames.items() if frame is None or frame.empty]
        if empty:
            raise ValueError(
                f"Incomplete {purpose.lower()}: empty {', '.join(empty)} alongside nonempty model results."
            )
        return frames

    def save(self, fig, name: str, caption: str):
        for fmt in self.formats:
            path = self.output / f"{name}.{fmt}"
            kwargs = {"dpi": self.dpi, "bbox_inches": "tight"}
            if fmt == "tiff":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(path, **kwargs)
            self.written.append(path.name)
        plt.close(fig)
        self.captions[name] = caption
        print(f"Rendered {name} ({', '.join(self.formats)})", flush=True)

    def sample_flow(self):
        frame = self.tables["sample_flow.csv"]
        _numbers(frame, ["n"], "sample_flow.csv")
        stage_ids, positions, edges, levels, leaves = _flow_layout(frame)
        fig, ax = plt.subplots(figsize=(max(9, 4.7 * leaves), max(4.5, 1.3 * levels)))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, levels)
        width = min(0.76, 0.78 / leaves)
        for parent, child in edges:
            parent_x, parent_y = positions[parent]
            child_x, child_y = positions[child]
            ax.add_patch(
                FancyArrowPatch(
                    (parent_x, parent_y - 0.31),
                    (child_x, child_y + 0.31),
                    arrowstyle="-|>",
                    mutation_scale=15,
                    color="#2C5C78",
                )
            )
        for stage_id, row in zip(stage_ids, frame.itertuples(index=False)):
            x, y = positions[stage_id]
            box = FancyBboxPatch(
                (x - width / 2, y - 0.27),
                width,
                0.54,
                boxstyle="round,pad=0.015",
                facecolor="#EDF4F8",
                edgecolor="#2C5C78",
                linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(
                x,
                y,
                f"{textwrap.fill(str(row.stage), 55 if leaves == 1 else 33)}\nn = {int(row.n)}",
                ha="center",
                va="center",
                fontsize=12,
            )
        ax.set_title("Participant flow", fontsize=16, pad=16)
        ax.axis("off")
        self.save(
            fig,
            "Figure1_sample_flow",
            "Participant counts and parent-child relationships are taken directly from sample_flow.csv. Where stage_id and parent_stage_id are supplied, arrows follow those saved links and sibling groups are displayed as branches, not successive exclusions. Legacy tables without parent identifiers use their saved row order. No participant counts or cohort relationships are inferred by the renderer.",
        )

    def abundance(self):
        summary = self.tables["taxon_summary.csv"].copy()
        points = self.tables["abundance_long.csv"].copy()
        _numbers(summary, ["median", "q25", "q75", "n"], "taxon_summary.csv")
        _numbers(points, ["log10_copies_plus1"], "abundance_long.csv")
        if (summary[["median", "q25", "q75"]] < 0).any().any():
            raise ValueError("Abundance quartiles cannot be negative for copy-number display.")
        taxa = list(dict.fromkeys(summary["taxon"].astype(str)))
        groups = sorted(points["group"].unique(), key=_group_key)
        if not taxa or not groups:
            raise ValueError("Saved abundance results have no taxa or groups.")
        ncols = min(4, len(taxa))
        nrows = (len(taxa) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.0 * nrows), squeeze=False)
        rng = np.random.default_rng(42)  # Only graphical jitter; not statistical resampling.
        for ax, taxon in zip(axes.flat, taxa):
            for position, group in enumerate(groups):
                values = points.loc[
                    (points["taxon"] == taxon)
                    & (points["group"].map(_group_key) == _group_key(group)),
                    "log10_copies_plus1",
                ].to_numpy()
                color = GROUP_COLORS.get(_group_key(group), "#777777")
                ax.scatter(
                    position + rng.uniform(-0.12, 0.12, len(values)),
                    values,
                    color=color,
                    s=11,
                    alpha=0.5,
                    linewidths=0,
                    rasterized=True,
                )
                stats_row = summary.loc[
                    (summary["taxon"] == taxon)
                    & (summary["group"].map(_group_key) == _group_key(group))
                ]
                if len(stats_row) != 1:
                    raise ValueError(f"Expected one saved summary for {taxon}, group {group}.")
                row = stats_row.iloc[0]
                med, q1, q3 = np.log10(
                    np.asarray([row["median"], row["q25"], row["q75"]], dtype=float) + 1
                )
                ax.bxp(
                    [{"med": med, "q1": q1, "q3": q3, "whislo": q1, "whishi": q3, "fliers": []}],
                    positions=[position],
                    widths=0.48,
                    showfliers=False,
                    showcaps=False,
                    manage_ticks=False,
                    patch_artist=True,
                    boxprops={"facecolor": "white", "edgecolor": color, "alpha": 0.85},
                    medianprops={"color": "black", "linewidth": 1.4},
                    whiskerprops={"linewidth": 0},
                )
            ax.set_title(_short_taxon(taxon), fontstyle="italic", fontsize=11)
            ax.set_xticks(
                range(len(groups)),
                [_group_label(g).replace(" halitosis", "\nhalitosis") for g in groups],
                fontsize=9,
            )
            ax.set_ylabel("log₁₀(copies + 1)")
            ax.grid(axis="y", alpha=0.18)
        for ax in list(axes.flat)[len(taxa) :]:
            ax.axis("off")
        fig.suptitle("PCR target abundance by halitosis group", fontsize=16)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        self.save(
            fig,
            "Figure2A_PCR_abundance",
            "All saved PCR targets are displayed. Points are saved individual log10(copy number + 1) values; boxes display saved raw-scale medians and interquartile ranges transformed to the displayed scale. Whiskers are omitted. Horizontal jitter serves only to separate points. Group comparisons must be taken from the analysis tables; no tests are performed during rendering.",
        )

    def correlation_network(self):
        corr = self.correlation
        labels = [_short_taxon(t) for t in corr.index]
        size = max(8, 0.46 * len(corr))
        fig, ax = plt.subplots(figsize=(size + 1, size))
        masked = np.ma.masked_invalid(corr.to_numpy(dtype=float))
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#D3D3D3")
        im = ax.imshow(masked, vmin=-1, vmax=1, cmap=cmap)
        ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontstyle="italic")
        ax.set_yticks(range(len(labels)), labels, fontstyle="italic")
        ax.set_title("Pearson correlation of PCR targets", pad=16)
        if len(corr) <= 20:
            for i in range(len(corr)):
                for j in range(len(corr)):
                    value = corr.iloc[i, j]
                    if pd.notna(value):
                        ax.text(
                            j,
                            i,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if abs(value) > 0.65 else "black",
                        )
        fig.colorbar(im, ax=ax, shrink=0.75, label="Saved Pearson r")
        fig.tight_layout()
        self.save(
            fig,
            "Figure2B_Pearson_correlation",
            "The complete saved correlation matrix is displayed. Gray cells denote undefined correlations. Coefficients are read from correlation.csv; no correlation or significance test is calculated by the renderer.",
        )

        edges = self.tables["correlation_edges.csv"].copy()
        _numbers(edges, ["r"], "correlation_edges.csv")
        nodes = corr.index.tolist()
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(nodes), endpoint=False)
        positions = {
            node: np.array([np.cos(angle), np.sin(angle)]) for node, angle in zip(nodes, angles)
        }
        fig, ax = plt.subplots(figsize=(11, 10))
        cluster_colors = {
            cluster: plt.get_cmap("tab20")(i % 20) for i, cluster in enumerate(self.clusters)
        }
        memberships = {
            node: cluster for cluster, members in self.clusters.items() for node in members
        }
        for row in edges.itertuples(index=False):
            if row.source not in positions or row.target not in positions:
                raise ValueError(
                    "A saved graph edge refers to a taxon absent from correlation.csv."
                )
            if row.source == row.target:
                continue
            start, end = positions[row.source], positions[row.target]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#4485A2" if row.r >= 0 else "#C47841",
                linewidth=0.5 + 2 * abs(row.r),
                alpha=0.45,
                zorder=1,
            )
        for node, xy in positions.items():
            color = cluster_colors.get(memberships.get(node), "#BFC4C8")
            ax.scatter(*xy, s=240, color=color, edgecolors="white", zorder=3)
            label_xy = xy * 1.16
            ax.text(
                *label_xy,
                _short_taxon(node),
                ha="left" if xy[0] > 0.1 else "right" if xy[0] < -0.1 else "center",
                va="center",
                fontsize=10,
                fontstyle="italic",
            )
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color=color, label=name, markersize=8)
            for name, color in cluster_colors.items()
        ]
        handles += [
            Line2D([0], [0], color="#4485A2", label="Positive edge"),
            Line2D([0], [0], color="#C47841", label="Negative edge"),
        ]
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=min(4, max(1, len(handles))),
            frameon=False,
        )
        ax.set(
            xlim=(-1.6, 1.6),
            ylim=(-1.3, 1.4),
            aspect="equal",
            title="Saved correlation graph and connected components",
        )
        ax.axis("off")
        self.save(
            fig,
            "Figure2C_correlation_network",
            "Nodes are PCR targets; edges and connected-component memberships are read from saved full-cohort descriptive outputs. Predictive nested cluster membership is discovered separately within train folds and saved in selected_features.csv. Node placement is a deterministic circle and does not encode biological distance. Nodes in a connected component need not all be directly connected to each other. This full-cohort graph is not an independently validated predictive panel.",
        )

    def classification_figures(self):
        if not self.classification:
            return
        summary = self.classification["classification_summary.csv"].copy()
        roc = self.classification["classification_roc.csv"].copy()
        if summary.empty:
            self.skipped.append(
                {
                    "figure": "Classification figures",
                    "reason": "Saved classification summary is empty.",
                }
            )
            return
        _numbers(
            summary,
            ["pooled_oof_auc", "fold_mean_auc", "fold_sd_auc"],
            "classification_summary.csv",
        )
        _numbers(
            summary, ["auc_ci_low", "auc_ci_high"], "classification_summary.csv", allow_nan=True
        )
        _numbers(roc, ["fpr", "tpr"], "classification_roc.csv")
        algorithms = summary["algorithm"].drop_duplicates().tolist()
        fig, axes = plt.subplots(
            1, len(algorithms), figsize=(7.6 * len(algorithms), 6), squeeze=False
        )
        for ax, algorithm in zip(axes.flat, algorithms):
            rows = summary.loc[summary.algorithm == algorithm]
            for index, row in enumerate(rows.itertuples(index=False)):
                curve = roc.loc[(roc.model == row.model) & (roc.algorithm == algorithm)]
                if curve.empty:
                    raise ValueError(f"No saved ROC coordinates for {row.model}, {algorithm}.")
                color = plt.get_cmap("tab20")(index % 20)
                ax.plot(
                    curve.fpr,
                    curve.tpr,
                    color=color,
                    lw=1.4,
                    label=f"{row.model} ({row.pooled_oof_auc:.2f})",
                )
            ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1.01),
                xlabel="False-positive rate",
                ylabel="True-positive rate",
                title=str(algorithm),
                aspect="equal",
            )
            ax.legend(
                fontsize=7,
                loc="lower right",
                title="All panels (pooled OOF AUROC)",
                title_fontsize=8,
            )
        fig.suptitle("Out-of-fold ROC curves: all saved classification models", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        self.save(
            fig,
            "Figure3A_classification_ROC",
            "All models in the classification summary are displayed, grouped by algorithm. Curves use precomputed ROC coordinates and legend values use pooled out-of-fold AUROC. Halitosis is the positive class. Model comparisons and inference are not recomputed by the renderer.",
        )
        primary = "repeat_mean_auc" if "repeat_mean_auc" in summary else "pooled_oof_auc"
        self._metric_panels(
            summary,
            primary,
            "auc_ci_low",
            "auc_ci_high",
            "Mean repeat-specific OOF AUROC",
            "Figure3B_classification_performance",
            None,
            "All saved classification models and nested selection procedures are displayed. Points are mean repeat-specific pooled OOF AUROC; error bars are saved participant-block bootstrap confidence limits. These conditional intervals keep each person's repeated predictions together and do not refit or reselect models. ROC panels separately use each participant's averaged OOF prediction, which is a different summary. Fold means and SD are separately available.",
        )

    def _metric_panels(self, summary, metric, low, high, xlabel, name, target_column, caption):
        targets = summary[target_column].drop_duplicates().tolist() if target_column else [None]
        max_rows = max(
            len(summary.loc[summary[target_column] == target]) if target_column else len(summary)
            for target in targets
        )
        fig, axes = plt.subplots(
            1,
            len(targets),
            figsize=(max(9, 7.4 * len(targets)), max(4, 0.34 * max_rows + 1.5)),
            squeeze=False,
        )
        for ax, target in zip(axes.flat, targets):
            frame = (
                summary.loc[summary[target_column] == target].copy()
                if target_column
                else summary.copy()
            )
            frame = frame.sort_values(metric, ascending=True, kind="stable")
            y = np.arange(len(frame))
            ax.scatter(frame[metric], y, s=25, color="#286B91", zorder=3)
            for position, row in enumerate(frame.to_dict("records")):
                if pd.notna(row[low]) and pd.notna(row[high]):
                    ax.plot(
                        [row[low], row[high]],
                        [position, position],
                        color="#286B91",
                        linewidth=1.3,
                        zorder=2,
                    )
            ax.set_yticks(
                y, [_label(m, a) for m, a in zip(frame.model, frame.algorithm)], fontsize=8
            )
            ax.set_xlabel(xlabel)
            ax.axvline(0.5 if metric.endswith("auc") else 0, color="#999999", linestyle="--", lw=1)
            ax.grid(axis="x", alpha=0.2)
            ax.set_title(
                str(target).replace("_ppb", " (ppb)")
                if target is not None
                else "All models; exploratory ordering by saved AUROC"
            )
        fig.tight_layout()
        self.save(fig, name, caption)

    def regression_figures(self):
        if not self.regression:
            return
        summary = self.regression["regression_summary.csv"].copy()
        oof = self.regression["regression_oof.csv"].copy()
        if summary.empty:
            self.skipped.append(
                {"figure": "Regression figures", "reason": "Saved regression summary is empty."}
            )
            return
        _numbers(
            summary,
            ["fold_mean_r2", "fold_sd_r2", "pooled_oof_r2", "mae", "rmse"],
            "regression_summary.csv",
        )
        _numbers(summary, ["r2_ci_low", "r2_ci_high"], "regression_summary.csv", allow_nan=True)
        _numbers(oof, ["y_true", "y_pred", "absolute_error"], "regression_oof.csv")
        primary = "repeat_mean_r2" if "repeat_mean_r2" in summary else "pooled_oof_r2"
        self._metric_panels(
            summary,
            primary,
            "r2_ci_low",
            "r2_ci_high",
            "Mean repeat-specific OOF R²",
            "Figure4A_regression_performance",
            "target",
            "All saved regression models are displayed. Points are mean repeat-specific pooled OOF R2 and saved participant-block confidence limits. Repeated predictions stay with their participant in resampling; fits and model selection are not repeated. Model ordering is exploratory. R2 uses the configured raw or log outcome scale; these scales must be reported separately.",
        )
        participant_path = self.results / "participant_average_regression_oof.csv"
        if participant_path.is_file():
            oof = _read_table(
                participant_path,
                [
                    "participant_id",
                    "target",
                    "model",
                    "algorithm",
                    "y_true",
                    "y_pred",
                    "mean_absolute_error_across_repeats",
                ],
            )
        targets = summary.target.drop_duplicates().tolist()
        algorithms = summary.algorithm.drop_duplicates().tolist()
        fig, axes = plt.subplots(
            len(targets),
            len(algorithms),
            figsize=(5.3 * len(algorithms), 4.6 * len(targets)),
            squeeze=False,
        )
        selected = []
        for i, target in enumerate(targets):
            for j, algorithm in enumerate(algorithms):
                ax = axes[i, j]
                candidates = summary.loc[
                    (summary.target == target) & (summary.algorithm == algorithm)
                ]
                if candidates.empty:
                    ax.axis("off")
                    continue
                best = candidates.sort_values(
                    ["fold_mean_r2", "model"], ascending=[False, True], kind="stable"
                ).iloc[0]
                frame = oof.loc[
                    (oof.target == target)
                    & (oof.algorithm == algorithm)
                    & (oof.model == best.model)
                ]
                if frame.empty:
                    raise ValueError(
                        f"No saved regression OOF rows for {target}, {algorithm}, {best.model}."
                    )
                ax.scatter(
                    frame.y_true,
                    frame.y_pred,
                    s=22,
                    alpha=0.6,
                    color=plt.get_cmap("tab10")(j),
                    linewidth=0,
                )
                values = np.r_[frame.y_true.to_numpy(), frame.y_pred.to_numpy()]
                vmin, vmax = float(np.min(values)), float(np.max(values))
                if vmax == vmin:
                    vmax += 1
                ax.plot([vmin, vmax], [vmin, vmax], "--", color="#777777", linewidth=1)
                ax.set(
                    xlabel="Observed concentration (ppb)",
                    ylabel="Mean OOF predicted concentration (ppb)",
                    title=f"{target} | {algorithm}\n{best.model}; fold mean R² = {best.fold_mean_r2:.2f}",
                )
                ax.grid(alpha=0.15)
                selected.append(
                    {"target": target, "algorithm": algorithm, "model": str(best.model)}
                )
        fig.suptitle(
            "Exploratory display: highest fold-mean R² per target and algorithm", fontsize=14
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self.save(
            fig,
            "Figure4B_selected_regression_scatter",
            "For visualization only, one model per target and algorithm is selected by the highest saved fold-mean R2 (ties broken by model name). This is an exploratory selection from the same CV results. Points are precomputed participant-average OOF predictions in ppb; dashed lines indicate identity. Repeated predictions are not displayed as independent participants.",
        )
        self.notes.append("Figure4B selection: " + json.dumps(selected, ensure_ascii=False))

        max_rows = max(len(summary.loc[summary.target == target]) for target in targets)
        fig, axes = plt.subplots(
            1,
            len(targets),
            figsize=(max(9, 7.5 * len(targets)), max(4, 0.34 * max_rows + 1.5)),
            squeeze=False,
        )
        rng = np.random.default_rng(52)
        for ax, target in zip(axes.flat, targets):
            ranked = summary.loc[summary.target == target].sort_values(
                "mae", ascending=False, kind="stable"
            )
            for position, row in enumerate(ranked.itertuples(index=False)):
                error_column = (
                    "mean_absolute_error_across_repeats"
                    if "mean_absolute_error_across_repeats" in oof
                    else "absolute_error"
                )
                errors = oof.loc[
                    (oof.target == target)
                    & (oof.algorithm == row.algorithm)
                    & (oof.model == row.model),
                    error_column,
                ]
                if errors.empty:
                    raise ValueError(
                        f"Missing saved errors for {target}, {row.algorithm}, {row.model}."
                    )
                ax.scatter(
                    errors,
                    position + rng.uniform(-0.13, 0.13, len(errors)),
                    color="#8EADB8",
                    s=8,
                    alpha=0.45,
                    linewidths=0,
                    rasterized=True,
                )
                ax.scatter([row.mae], [position], marker="D", color="#254F6C", s=23, zorder=3)
            ax.set_yticks(
                np.arange(len(ranked)),
                [_label(m, a) for m, a in zip(ranked.model, ranked.algorithm)],
                fontsize=8,
            )
            ax.set(
                xlabel="Absolute OOF prediction error (ppb)",
                title=str(target).replace("_ppb", " (ppb)"),
            )
            ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        self.save(
            fig,
            "Figure4C_regression_error_ranking",
            "All saved regression models are displayed in exploratory order by saved MAE. Light points are participant-level mean absolute OOF errors across repeats and dark diamonds are saved MAEs. Jitter only separates points. Statistical comparisons are performed on the same participant-level absolute errors in the analysis code, not on R2. No tests are calculated during rendering.",
        )

    def importance_figures(self):
        path = self.results / "importance.csv"
        if not path.is_file():
            self.skipped.append(
                {"figure": "Feature importance", "reason": "Optional importance.csv not supplied."}
            )
            return
        frame = _read_table(path, ["target", "model", "algorithm", "feature", "importance", "kind"])
        _numbers(frame, ["importance"], "importance.csv")
        for keys, subset in frame.groupby(
            ["target", "model", "algorithm", "kind"], sort=False, dropna=False
        ):
            target, model, algorithm, kind = keys
            subset = subset.sort_values("importance", kind="stable")
            fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(subset) + 1.5)))
            ax.barh(
                np.arange(len(subset)),
                subset.importance,
                color=["#C56A5A" if v >= 0 else "#3D7C9D" for v in subset.importance],
            )
            ax.set_yticks(
                np.arange(len(subset)), [_short_taxon(v) for v in subset.feature], fontsize=9
            )
            ax.axvline(0, color="#888888", lw=1)
            ax.set(xlabel=f"Saved {kind}", title=f"{target}: {model} ({algorithm})")
            fig.tight_layout()
            name = "Supplement_importance_" + _safe_name("_".join(map(str, keys)))
            self.save(
                fig,
                name,
                "All features in the saved importance group are displayed. Values, method, model, and target are read from importance.csv. Interpretation is conditional on the fitted model and does not establish causation or independent biological validation. The importance kind and run manifest determine the units.",
            )

    def revision_figures(self):
        path = self.results / "PCR_panel_diversity_summary.csv"
        if path.is_file():
            table = _read_table(
                path,
                [
                    "metric",
                    "nonhalitosis_mean",
                    "nonhalitosis_sd",
                    "halitosis_mean",
                    "halitosis_sd",
                ],
            )
            fig, axes = plt.subplots(1, len(table), figsize=(4.1 * len(table), 4.1), squeeze=False)
            titles = {
                "PCR_detected_targets": "Detected PCR targets",
                "PCR_panel_Shannon_ln": "Shannon (natural log)",
                "PCR_panel_Pielou_evenness": "Pielou evenness",
            }
            for ax, row in zip(axes.flat, table.itertuples()):
                ax.errorbar(
                    [0, 1],
                    [row.nonhalitosis_mean, row.halitosis_mean],
                    yerr=[row.nonhalitosis_sd, row.halitosis_sd],
                    fmt="o",
                    capsize=4,
                    color="#286B91",
                )
                ax.set(
                    xticks=[0, 1],
                    xticklabels=["Without\nhalitosis", "With\nhalitosis"],
                    title=titles.get(row.metric, row.metric),
                    ylabel="Mean ± SD",
                )
                ax.grid(axis="y", alpha=0.15)
            fig.tight_layout()
            self.save(
                fig,
                "Supplement_PCR_panel_diversity",
                "Saved means and sample SDs for the 16-target PCR panel. Detected-target count, natural-log Shannon entropy and Pielou evenness characterize this targeted panel. Chao1 is excluded. Error bars are SD, not confidence limits. Group tests and multiplicity corrections are reported in the saved diversity table.",
            )
        path = self.results / "species_comparisons.csv"
        if path.is_file():
            table = _read_table(
                path,
                [
                    "representation",
                    "taxon",
                    "cliffs_delta",
                    "cliffs_delta_ci_low",
                    "cliffs_delta_ci_high",
                ],
            )
            reps = list(table.representation.drop_duplicates())
            fig, axes = plt.subplots(1, len(reps), figsize=(5.2 * len(reps), 7), squeeze=False)
            for ax, rep in zip(axes.flat, reps):
                sub = table.loc[table.representation == rep].sort_values("taxon")
                y = np.arange(len(sub))
                ax.scatter(sub.cliffs_delta, y, color="#286B91", s=20)
                for position, row in enumerate(sub.itertuples()):
                    if np.isfinite(row.cliffs_delta_ci_low) and np.isfinite(
                        row.cliffs_delta_ci_high
                    ):
                        ax.plot(
                            [row.cliffs_delta_ci_low, row.cliffs_delta_ci_high],
                            [position, position],
                            color="#286B91",
                            lw=1,
                        )
                ax.set_yticks(y, [_short_taxon(t) for t in sub.taxon], fontsize=8)
                ax.set(
                    title=str(rep),
                    xlabel="Cliff's delta (halitosis minus non-halitosis)",
                    xlim=(-1.05, 1.05),
                )
                ax.axvline(0, color="gray", ls="--", lw=1)
            fig.tight_layout()
            self.save(
                fig,
                "Supplement_normalization_effects",
                "Saved Cliff delta and participant-bootstrap percentile intervals across PCR representations. Positive values indicate larger values in the halitosis group. Relative abundance and CLR describe only the 16-target panel. Monotone log transformations preserve ranks; equal raw/log rank statistics do not constitute independent replication. Multiplicity-adjusted tests are reported separately.",
            )
        path = self.results / "cluster_stability_summary.csv"
        if path.is_file():
            table = _read_table(
                path,
                [
                    "configuration",
                    "reference_cluster",
                    "jaccard_mean",
                    "jaccard_p025",
                    "jaccard_p975",
                ],
            )
            table = table.loc[table.configuration.astype(str).str.contains("|0.4|", regex=False)]
            if not table.empty:
                fig, ax = plt.subplots(figsize=(11, max(4, 0.28 * len(table) + 1)))
                y = np.arange(len(table))
                ax.scatter(table.jaccard_mean, y, color="#286B91", s=20)
                for i, row in enumerate(table.itertuples()):
                    ax.plot([row.jaccard_p025, row.jaccard_p975], [i, i], color="#286B91", lw=1)
                ax.set_yticks(
                    y,
                    [f"{r.configuration} / {r.reference_cluster}" for r in table.itertuples()],
                    fontsize=7,
                )
                ax.set(
                    xlim=(-0.02, 1.02),
                    xlabel="Bootstrap membership overlap (Jaccard)",
                    title="Cluster stability at the prespecified 0.4 threshold",
                )
                fig.tight_layout()
                self.save(
                    fig,
                    "Supplement_cluster_stability",
                    "Mean bootstrap Jaccard overlap and the 2.5th/97.5th percentiles of its bootstrap distribution for stored cluster definitions at threshold 0.4. These ranges summarize membership stability and are not predictive-performance confidence intervals. Other thresholds and co-clustering matrices remain in the saved tables.",
                )
        path = self.results / "calibration_bins.csv"
        if path.is_file():
            table = _read_table(
                path, ["model", "algorithm", "mean_predicted", "fraction_positive", "n"]
            )
            table = table.loc[table.model == "Clinical_only"]
            if not table.empty:
                fig, ax = plt.subplots(figsize=(6, 5))
                for algorithm, sub in table.groupby("algorithm"):
                    ax.plot(sub.mean_predicted, sub.fraction_positive, "o-", label=algorithm)
                ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
                ax.set(
                    xlim=(0, 1),
                    ylim=(0, 1),
                    xlabel="Mean predicted probability",
                    ylabel="Observed fraction with halitosis",
                    title="Clinical-only probability diagnostic",
                )
                ax.legend()
                fig.tight_layout()
                self.save(
                    fig,
                    "Supplement_clinical_probability_diagnostic",
                    "Descriptive calibration bins from saved participant-average OOF probabilities for clinical-only models. Bin sizes are saved in calibration_bins.csv. No calibration model or confidence interval is fitted in this figure; small bins can be unstable.",
                )

    def run(self):
        self.sample_flow()
        self.abundance()
        self.correlation_network()
        self.classification_figures()
        self.regression_figures()
        self.importance_figures()
        self.revision_figures()
        report = {
            "status": "complete",
            "source_results": str(self.results.resolve()),
            "source_manifest_status": self.manifest.get("status"),
            "files": self.written,
            "skipped_optional": self.skipped,
            "notes": self.notes,
            "dpi": self.dpi,
            "formats": self.formats,
            "selection": "All models in performance/error summaries; one highest fold-mean R² model per target/algorithm only in the illustrative scatter panels.",
            "analysis_performed": False,
        }
        (self.output / "figure_manifest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        caption_text = "\n\n".join(f"{name}\n{caption}" for name, caption in self.captions.items())
        (self.output / "figure_captions.txt").write_text(caption_text + "\n", encoding="utf-8")
        for item in self.skipped:
            print(f"Optional output skipped: {item['figure']}: {item['reason']}")
        print(f"Figure files: {len(self.written)}. Output: {self.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "paper")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "paper")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", choices=["png", "tiff"], default=["png", "tiff"])
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive.")
    if args.results.resolve() == args.output.resolve():
        parser.error("--output must differ from --results.")
    try:
        Renderer(args.results, args.output, args.dpi, list(dict.fromkeys(args.formats))).run()
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Figure rendering failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
