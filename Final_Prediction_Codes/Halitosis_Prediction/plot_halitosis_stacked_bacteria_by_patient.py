import os
import json
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


RAW_DATA_PATH = Path("PCR_NGS_Data.xlsx")
OUTPUT_DIR = Path("Figure_Stacked_Bacteria_ByPatient")
OUTPUT_STEM = "Halitosis_NonHalitosis_16Bacteria_Stacked_ByPatient"
REFERENCE_COLOR_JSON = Path("gas_model_color_map.json")
REFERENCE_COLOR_CSV = Path("Visualization_NewMethod_0.4_Nonclinical/Unified_Model_Color_Mapping.csv")
P_GINGIVALIS_COLOR = "#B55A5A"  # toned-down red
S_SOBRINUS_COLOR = "#005F73"    # deep teal (far from red)


def format_bacterial_name(raw_name: str) -> str:
    clean = (
        str(raw_name)
        .replace("PCR_", "")
        .replace("PCR", "")
        .replace("__", " ")
        .replace("_", " ")
        .strip()
    )
    parts = clean.split()
    if len(parts) >= 2:
        return f"{parts[0][0].upper()}. {parts[1].lower()}"
    return clean


def model_key_to_bacteria_name(model_key: str):
    key = str(model_key).strip()
    if not key.startswith("Single_"):
        return None

    body = key[len("Single_") :]
    parts = body.split("_")
    if len(parts) >= 2 and parts[-1] in {"Ridge", "Transformer", "Logistic", "KRR"}:
        body = "_".join(parts[:-1])

    body = body.replace("_", " ").strip()
    if not body:
        return None
    return format_bacterial_name(body)


def load_reference_bacteria_colors():
    ref = {}

    if REFERENCE_COLOR_JSON.exists():
        with open(REFERENCE_COLOR_JSON, "r", encoding="utf-8") as f:
            color_map = json.load(f)
        for model_key, color in color_map.items():
            bac_name = model_key_to_bacteria_name(model_key)
            if bac_name and bac_name not in ref:
                ref[bac_name] = color

    if REFERENCE_COLOR_CSV.exists():
        try:
            df_color = pd.read_csv(REFERENCE_COLOR_CSV)
            for _, row in df_color.iterrows():
                color = str(row.get("ColorHex", "")).strip()
                if not color:
                    continue

                key1 = str(row.get("UniqueKey", "")).strip()
                key2 = str(row.get("ModelName", "")).strip()
                bac_name = model_key_to_bacteria_name(key1)
                if bac_name is None:
                    bac_name = model_key_to_bacteria_name(key2)
                if bac_name and bac_name not in ref:
                    ref[bac_name] = color
        except Exception:
            pass

    return ref


def load_raw_any_format(filepath: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(filepath, header=None)
    except Exception:
        pass

    for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(filepath, header=None, encoding=enc, engine="python")
        except Exception:
            continue
    raise ValueError(f"Cannot load file: {filepath}")


def load_clean_dataframe(filepath: Path):
    df_raw = load_raw_any_format(filepath)

    header_idx = -1
    for i, row in df_raw.iterrows():
        row_str = str(row.values)
        if "Sex" in row_str and "Age" in row_str:
            header_idx = i
            break
    if header_idx == -1:
        header_idx = 6

    df_raw.columns = df_raw.iloc[header_idx].astype(str).str.strip()
    df = df_raw.iloc[header_idx + 1 :].reset_index(drop=True)

    sample_col = next((c for c in df.columns if str(c).strip().lower() == "sample"), None)
    halitosis_col = next(
        (c for c in df.columns if "halitosis" in str(c).strip().lower()),
        None,
    )
    if sample_col is None or halitosis_col is None:
        raise ValueError("Required columns ('Sample', 'Halitosis') were not found.")

    selected = {
        "Sample": df[sample_col].astype(str).str.strip(),
        "Halitosis": pd.to_numeric(df[halitosis_col], errors="coerce"),
    }

    bacteria_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        if col_str.startswith("PCR"):
            bac_name = format_bacterial_name(col_str)
            selected[bac_name] = pd.to_numeric(df[col], errors="coerce")
            bacteria_cols.append(bac_name)

    if not bacteria_cols:
        raise ValueError("No PCR bacteria columns were found.")

    clean = pd.DataFrame(selected).dropna(subset=["Halitosis"]).copy()
    clean["Halitosis"] = clean["Halitosis"].astype(int)
    clean = clean[clean["Halitosis"].isin([0, 1])].reset_index(drop=True)

    clean["Sample"] = clean["Sample"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    clean["Sample"] = clean["Sample"].fillna(pd.Series(range(1, len(clean) + 1)).map("S{:03d}".format))

    for col in bacteria_cols:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0.0)

    return clean, bacteria_cols


def build_color_map(bacteria_cols):
    ref_bacteria_colors = load_reference_bacteria_colors()
    color_map = {}
    used_colors = set()

    # 1) Reference colors from violin plots are applied first.
    for bac in bacteria_cols:
        if bac in ref_bacteria_colors:
            col = mcolors.to_hex(ref_bacteria_colors[bac]).lower()
            color_map[bac] = col
            used_colors.add(col)

    # 2) Fill remaining bacteria with non-overlapping fallback colors.
    fallback_candidates = (
        [mcolors.to_hex(c).lower() for c in sns.color_palette("tab20", 20)]
        + [mcolors.to_hex(c).lower() for c in sns.color_palette("tab20b", 20)]
        + [mcolors.to_hex(c).lower() for c in sns.color_palette("tab20c", 20)]
    )
    cand_idx = 0
    for bac in bacteria_cols:
        if bac in color_map:
            continue
        while cand_idx < len(fallback_candidates) and fallback_candidates[cand_idx] in used_colors:
            cand_idx += 1
        if cand_idx >= len(fallback_candidates):
            # Extremely unlikely for 16 species, but keep a deterministic fallback.
            col = mcolors.to_hex(sns.color_palette("husl", n_colors=len(bacteria_cols))[len(color_map)]).lower()
        else:
            col = fallback_candidates[cand_idx]
            cand_idx += 1
        color_map[bac] = col
        used_colors.add(col)

    # 3) If no reference color exists for P. gingivalis, enforce toned-down red.
    if "P. gingivalis" in color_map and "P. gingivalis" not in ref_bacteria_colors:
        color_map["P. gingivalis"] = P_GINGIVALIS_COLOR.lower()

    # 4) Force S. sobrinus to a clearly distinct hue from P. gingivalis.
    if "S. sobrinus" in color_map:
        color_map["S. sobrinus"] = S_SOBRINUS_COLOR.lower()
    return color_map


def plot_stacked_bars(df_clean, bacteria_cols, color_map, output_dir: Path):
    group_map = {0: "Non-halitosis", 1: "Halitosis"}
    group_counts = df_clean["Halitosis"].value_counts().to_dict()
    max_group_n = max(group_counts.values())

    fig_width = max(16, max_group_n * 0.30)
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 12), dpi=300, sharey=True)

    for ax, group_value in zip(axes, [0, 1]):
        grp = df_clean[df_clean["Halitosis"] == group_value].copy()
        grp = grp.reset_index(drop=True)

        x = np.arange(len(grp))
        bottom = np.zeros(len(grp), dtype=float)

        for bac in bacteria_cols:
            values = grp[bac].to_numpy(dtype=float)
            ax.bar(
                x,
                values,
                bottom=bottom,
                color=color_map[bac],
                width=0.9,
                linewidth=0,
                label=bac,
            )
            bottom += values

        ax.set_title(f"{group_map[group_value]} (n={len(grp)})", fontsize=13, fontweight="bold")
        ax.set_ylabel("Abundance (copies/mL)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(grp["Sample"], rotation=90, fontsize=7)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Patient", fontsize=11, fontweight="bold")

    legend_handles = [Patch(facecolor=color_map[b], label=b) for b in bacteria_cols]
    fig.legend(
        handles=legend_handles,
        title="Bacteria (16 species)",
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False,
    )

    fig.suptitle(
        "Stacked Abundance of 16 PCR Bacteria by Patient Group",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.05, right=0.84, bottom=0.20, top=0.93, hspace=0.28)

    output_png = output_dir / f"{OUTPUT_STEM}.png"
    output_pdf = output_dir / f"{OUTPUT_STEM}.pdf"
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    return output_png, output_pdf


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_clean, bacteria_cols = load_clean_dataframe(RAW_DATA_PATH)
    color_map = build_color_map(bacteria_cols)
    output_png, output_pdf = plot_stacked_bars(df_clean, bacteria_cols, color_map, OUTPUT_DIR)

    color_df = pd.DataFrame(
        [{"Bacteria": b, "ColorHex": color_map[b]} for b in bacteria_cols]
    )
    color_df.to_csv(OUTPUT_DIR / "Stacked_Bacteria_Color_Mapping.csv", index=False)

    print(f"Saved figure: {output_png}")
    print(f"Saved figure: {output_pdf}")
    print(f"Saved color map: {OUTPUT_DIR / 'Stacked_Bacteria_Color_Mapping.csv'}")


if __name__ == "__main__":
    main()
