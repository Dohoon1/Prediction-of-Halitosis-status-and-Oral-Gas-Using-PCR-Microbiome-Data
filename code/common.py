"""Explicit configuration and input validation for the PCR study.

Importing this module neither reads participant data nor creates directories.
NGS sequencing-depth filters are intentionally absent from this PCR pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLINICAL_COLUMNS = ["Age", "Sex", "Smoking", "Oral_hygiene"]
GAS_COLUMNS = ["H2S_ppb", "CH3SH_ppb", "VSCs_ppb"]
PCR_TARGETS = [
    "Aggregatibacter_actinomycetemcomitans",
    "Prevotella_intermedia",
    "Prevotella_nigrescens",
    "Lactobacillus_casei",
    "Fusobacterium_nucleatum",
    "Streptococcus_mutans",
    "Streptococcus_sobrinus",
    "Treponema_denticola",
    "Porphyromonas_gingivalis",
    "Tannerella_forsythia",
    "Eikenella_corrodens",
    "Parvimonas_micra",
    "Campylobacter_rectus",
    "Eubacterium_nodatum",
    "Fusobacterium_alocis",
    "Porphyromonas_endodontalis",
]


@dataclass
class Dataset:
    frame: pd.DataFrame
    pcr_columns: list[str]
    clinical_columns: list[str]
    source_path: Path
    audit: dict = field(default_factory=dict)


def read_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(path)
    if config.get("cv", {}).get("outer_folds", 0) < 2:
        raise ValueError("cv.outer_folds must be at least 2.")
    if config.get("cv", {}).get("inner_folds", 0) < 2:
        raise ValueError("cv.inner_folds must be at least 2.")
    if config.get("bootstrap", {}).get("iterations", 0) < 0:
        raise ValueError("bootstrap.iterations must be nonnegative.")
    if config.get("modeling", {}).get("sampling", "none") not in {"smote", "smotenc", "none"}:
        raise ValueError("modeling.sampling must be smotenc or none (smote is a legacy option).")
    settings = config.get("modeling", {})
    if settings.get("panels", "fixed_exploratory") not in {"fixed_exploratory", "nested"}:
        raise ValueError("modeling.panels must be fixed_exploratory or nested.")
    if settings.get("pcr_transform", "none") not in {"none", "log1p", "log10", "relative", "clr"}:
        raise ValueError("modeling.pcr_transform must be none, log1p, log10, relative, or clr.")
    if settings.get("class_weight") not in {None, "balanced"}:
        raise ValueError("modeling.class_weight must be null or balanced.")
    if settings.get("sampling", "none") != "none" and settings.get("class_weight"):
        raise ValueError("Do not combine oversampling and class weights in this protocol.")
    if int(config.get("cv", {}).get("outer_repeats", 1)) < 1:
        raise ValueError("cv.outer_repeats must be positive.")
    if settings.get("target_transform", "none") not in {"none", "log1p"}:
        raise ValueError("modeling.target_transform must be none or log1p.")
    perm = config.get("permutation", {})
    if int(perm.get("iterations", 1000)) < 1 or int(perm.get("outer_repeats", 2)) < 1:
        raise ValueError("Permutation iterations/repeats must be positive.")
    if settings.get("missing_policy", "error") not in {"error", "impute"}:
        raise ValueError("modeling.missing_policy must be error or impute.")
    if not 0 < float(settings.get("threshold", 0.5)) < 1:
        raise ValueError("modeling.threshold must be between 0 and 1.")
    if not 0 <= float(settings.get("correlation_threshold", 0.4)) <= 1:
        raise ValueError("modeling.correlation_threshold must be between 0 and 1.")
    selectors = settings.get("selector_names", ["Nested_best_single", "Nested_best_candidate"])
    if len(set(selectors)) != len(selectors) or set(selectors) - {
        "Nested_best_single",
        "Nested_best_candidate",
    }:
        raise ValueError("Unknown or duplicate selector_names.")
    return config


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value) -> None:
    """Write strict JSON atomically; nonfinite floats must be resolved by callers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    def encode(item):
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=encode, allow_nan=False)
    temporary.replace(path)


def normalize_header(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def canonical_column(value) -> str | None:
    text = normalize_header(value)
    lower = text.lower()
    compact = re.sub(r"[^a-z0-9]", "", lower)
    if lower.startswith("pcr"):
        species = re.sub(r"^pcr[\s_:-]*", "", text, flags=re.I)
        species = re.sub(r"[\s_]+", "_", species).strip("_")
        if not species:
            return None
        # Preserve the measured target's historical name in the internal key.
        return "PCR_" + species
    if compact in {"participantid", "sample", "sampleid", "idpt", "subjectid", "patientid"}:
        return "participant_id"
    if compact in {"sex", "gender"}:
        return "Sex"
    if compact == "age":
        return "Age"
    if compact in {"smoking", "smokingstatus"}:
        return "Smoking"
    if compact in {"oralhygiene", "oralhygienestatus"}:
        return "Oral_hygiene"
    if compact in {"halitosis", "halitosisstatus"}:
        return "Halitosis"
    if compact in {"h2s", "h2sppb"}:
        return "H2S_ppb"
    if compact in {"ch3sh", "ch3shppb"}:
        return "CH3SH_ppb"
    if compact in {"vsc", "vscs", "vscppb", "vscsppb", "totalvscppb", "totalvscsppb"}:
        return "VSCs_ppb"
    if compact.startswith("totalbacteria"):
        return "Total_bacteria_copies_mL"
    return None


def _read_selected(path: Path, data_config: dict) -> tuple[pd.DataFrame, int]:
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, header=None, dtype=object, encoding="utf-8-sig")
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None, sheet_name=data_config.get("sheet", 0), dtype=object)
    else:
        raise ValueError("Input must be .csv, .xlsx, or .xls.")
    configured = data_config.get("header_row")
    if configured is None:
        candidates = []
        for index in range(min(30, len(raw))):
            names = [canonical_column(v) for v in raw.iloc[index]]
            if "Age" in names and "Sex" in names and any(n and n.startswith("PCR_") for n in names):
                candidates.append(index)
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one header row containing Age, Sex and PCR targets; found {candidates}. Set data.header_row explicitly (zero based)."
            )
        header_row = candidates[0]
    else:
        header_row = int(configured)
        if not 0 <= header_row < len(raw):
            raise ValueError("data.header_row is outside the input table.")
    selected: dict[str, pd.Series] = {}
    for index, header in enumerate(raw.iloc[header_row]):
        name = canonical_column(header)
        if name is None:
            continue
        if name in selected:
            raise ValueError(f"Duplicate recognized input column: {name}.")
        selected[name] = raw.iloc[header_row + 1 :, index].reset_index(drop=True)
    frame = pd.DataFrame(selected)
    frame = frame.dropna(how="all").reset_index(drop=True)
    return frame, header_row


def _numeric(series: pd.Series, name: str, allow_missing: bool) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce").astype(float)
    nonempty = series.notna() & series.astype(str).str.strip().ne("")
    invalid = nonempty & converted.isna()
    if invalid.any():
        raise ValueError(
            f"{name}: {int(invalid.sum())} nonnumeric values; these are not replaced by zero."
        )
    if np.isinf(converted.to_numpy()).any():
        raise ValueError(f"{name}: infinite values are not allowed.")
    if not allow_missing and converted.isna().any():
        raise ValueError(
            f"{name}: missing values found; configure training-fold imputation explicitly if appropriate."
        )
    return converted


def _category(
    series: pd.Series, name: str, mapping: dict[str, str], allow_missing: bool
) -> pd.Series:
    def convert(value):
        if pd.isna(value) or str(value).strip() == "":
            return np.nan
        token = str(value).strip().lower()
        if token not in mapping:
            raise ValueError(
                f"{name}: unrecognized category {str(value)!r}; update its explicit mapping."
            )
        return mapping[token]

    result = series.map(convert)
    if result.isna().any() and not allow_missing:
        raise ValueError(f"{name}: missing category; it is not encoded as a reference category.")
    return result


def load_dataset(config: dict, input_path: str | Path | None = None) -> Dataset:
    data_config = config.get("data", {})
    path = Path(input_path or data_config.get("path", "data/PCR_NGS_Data.xlsx")).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Input not found: {path}. Supply --input with the original workbook path."
        )
    frame, header_row = _read_selected(path, data_config)
    independent_total_present = "Total_bacteria_copies_mL" in frame
    # Excluded by the study team's explicit analysis policy. Never a covariate,
    # denominator, imputation input, or substitute for the measured panel sum.
    frame = frame.drop(columns=["Total_bacteria_copies_mL"], errors="ignore")
    input_n = len(frame)
    if "participant_id" not in frame:
        raise ValueError(
            "A sample/participant ID column is required to retain prediction alignment."
        )
    if frame.participant_id.isna().any():
        raise ValueError("Participant IDs contain missing values.")
    frame["participant_id"] = frame.participant_id.astype(str).str.strip()
    if frame.participant_id.eq("").any() or frame.participant_id.duplicated().any():
        raise ValueError("Participant IDs must be nonempty and unique.")
    pcr_columns = [c for c in frame if c.startswith("PCR_")]
    expected = data_config.get("expected_pcr_targets", 16)
    if expected is not None and len(pcr_columns) != int(expected):
        raise ValueError(
            f"Expected {expected} PCR targets, found {len(pcr_columns)}. Do not substitute NGS columns."
        )
    if not pcr_columns:
        raise ValueError("No PCR target columns found.")
    if data_config.get("enforce_pcr_whitelist", True):
        wanted = {"PCR_" + t for t in PCR_TARGETS}
        if set(pcr_columns) != wanted:
            raise ValueError(
                f"PCR whitelist mismatch: missing={sorted(wanted-set(pcr_columns))}; unexpected={sorted(set(pcr_columns)-wanted)}"
            )
    required = ["Halitosis", *GAS_COLUMNS]
    clinical_columns = (
        CLINICAL_COLUMNS if config.get("modeling", {}).get("include_clinical", True) else []
    )
    required += clinical_columns
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"Required columns missing: {missing}.")
    allow_missing = config.get("modeling", {}).get("missing_policy", "error") == "impute"
    for name in [*pcr_columns, "Age", "Total_bacteria_copies_mL", *GAS_COLUMNS, "Halitosis"]:
        if name not in frame:
            continue
        frame[name] = _numeric(
            frame[name],
            name,
            allow_missing=name not in {"Halitosis"} and (allow_missing or name in GAS_COLUMNS),
        )
        if name.startswith("PCR_") or name in GAS_COLUMNS or name == "Total_bacteria_copies_mL":
            if (frame[name].dropna() < 0).any():
                raise ValueError(f"{name}: negative concentrations are not allowed.")
    if not frame.Halitosis.isin([0, 1]).all():
        raise ValueError("Halitosis labels must be 0 or 1.")
    frame["Halitosis"] = frame.Halitosis.astype(int)
    label_records = sorted(zip(frame.participant_id.tolist(), frame.Halitosis.tolist()))
    labels_sha256 = hashlib.sha256(
        json.dumps(label_records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        data_config.get("expected_label_sha256")
        and labels_sha256 != data_config["expected_label_sha256"]
    ):
        raise ValueError("Original participant-label mapping changed; labels must be retained.")
    frame["PCR_panel_total"] = frame[pcr_columns].sum(axis=1, min_count=len(pcr_columns))
    expected_n = data_config.get("expected_participants")
    if expected_n is not None and len(frame) != int(expected_n):
        raise ValueError(f"Expected {expected_n} participants, found {len(frame)}.")
    expected_counts = data_config.get("expected_class_counts")
    observed_counts = {str(k): int(v) for k, v in frame.Halitosis.value_counts().items()}
    if expected_counts is not None and observed_counts != {
        str(k): int(v) for k, v in expected_counts.items()
    }:
        raise ValueError("Supplied label counts differ from the locked original-label cohort.")
    sex_map = {
        "m": "M",
        "male": "M",
        "man": "M",
        "남": "M",
        "남성": "M",
        "f": "F",
        "female": "F",
        "woman": "F",
        "여": "F",
        "여성": "F",
    }
    smoking_map = {
        "0": "No",
        "0.0": "No",
        "no": "No",
        "n": "No",
        "non-smoker": "No",
        "nonsmoker": "No",
        "1": "Yes",
        "1.0": "Yes",
        "yes": "Yes",
        "y": "Yes",
        "smoker": "Yes",
    }
    hygiene_map = {"good": "Good", "moderate": "Moderate", "poor": "Poor"}
    maps = {"Sex": sex_map, "Smoking": smoking_map, "Oral_hygiene": hygiene_map}
    for name, mapping in maps.items():
        if name in frame:
            custom = data_config.get("category_mappings", {}).get(name, {})
            mapping = {**mapping, **{str(k).lower(): str(v) for k, v in custom.items()}}
            frame[name] = _category(frame[name], name, mapping, allow_missing)
    if "Age" in frame and ((frame.Age.dropna() < 0) | (frame.Age.dropna() > 120)).any():
        raise ValueError("Age is outside 0–120 years.")
    complete_gases = frame[GAS_COLUMNS].notna().all(axis=1)
    discrepancy = np.abs(frame.VSCs_ppb - frame.H2S_ppb - frame.CH3SH_ppb)
    gas_mismatch = int(
        (complete_gases & (discrepancy > data_config.get("vsc_tolerance_ppb", 0.01))).sum()
    )
    thresholds = data_config.get("sex_vsc_thresholds", {"F": 65.79, "M": 79.94})
    label_mismatch = None
    if "Sex" in frame:
        eligible = frame.Sex.isin(thresholds) & frame.VSCs_ppb.notna()
        derived = frame.loc[eligible, "VSCs_ppb"] >= frame.loc[eligible, "Sex"].map(thresholds)
        label_mismatch = int((derived.astype(int) != frame.loc[eligible, "Halitosis"]).sum())
    audit = {
        "input_rows": input_n,
        "retained_rows": len(frame),
        "header_row_zero_based": header_row,
        "participant_label_sha256": labels_sha256,
        "pcr_targets": len(pcr_columns),
        "clinical_columns": clinical_columns,
        "class_counts": {
            str(k): int(v) for k, v in frame.Halitosis.value_counts().sort_index().items()
        },
        "missing_by_column": {c: int(v) for c, v in frame.isna().sum().items() if v},
        "vsc_sum_mismatches": gas_mismatch,
        "sex_threshold_label_mismatches": label_mismatch,
        "sex_vsc_thresholds_ppb": thresholds,
        "label_policy": "retain supplied labels; audit derived labels without overwriting",
        "label_rule_discrepancy_acknowledged": bool(
            data_config.get("label_rule_discrepancy_acknowledged", False)
        ),
        "ngs_read_filter_applied": False,
        "independent_total_present_in_source": independent_total_present,
        "independent_total_used": False,
        "panel_total_definition": "sum across the 16 targeted PCR concentrations; not universal 16S total load",
        "target_transform": config.get("modeling", {}).get("target_transform", "none"),
        "pcr_transform": config.get("modeling", {}).get("pcr_transform", "none"),
        "warnings": [],
    }
    if gas_mismatch:
        audit["warnings"].append(
            "Reported total VSC differs from H2S+CH3SH; values are retained, not silently replaced."
        )
    audit["primary_outcome_definition"] = data_config.get(
        "primary_outcome_definition", "supplied original labels"
    )
    audit["vsc_only_rule_is_primary_outcome"] = False
    expected_mismatch = data_config.get("expected_vsc_only_label_differences")
    if expected_mismatch is not None and label_mismatch != expected_mismatch:
        raise ValueError("VSC-only audit differs from the locked original-label cohort.")
    if label_mismatch:
        audit["warnings"].append(
            "VSC-only classification differs from the retained primary complaint-plus-VSC labels for "
            + str(label_mismatch)
            + " participants. This is an outcome-definition audit; primary labels are unchanged."
        )
    return Dataset(frame.reset_index(drop=True), pcr_columns, clinical_columns, path, audit)


def short_taxon(name: str) -> str:
    text = name.removeprefix("PCR_").replace("_", " ")
    parts = text.split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else text


def correlation_components(
    correlation: pd.DataFrame, threshold: float, absolute: bool = True, inclusive: bool = True
) -> tuple[dict, list[dict]]:
    """Connected components, not cliques; preserve signed edge coefficients."""
    columns = list(correlation.columns)
    adjacent = {column: set() for column in columns}
    edges = []
    for i, source in enumerate(columns):
        for target in columns[i + 1 :]:
            coefficient = float(correlation.loc[source, target])
            if not np.isfinite(coefficient):
                continue
            strength = abs(coefficient) if absolute else coefficient
            passes = strength >= threshold if inclusive else strength > threshold
            if passes:
                adjacent[source].add(target)
                adjacent[target].add(source)
                edges.append({"source": source, "target": target, "r": coefficient})
    unseen = set(columns)
    components = []
    for start in columns:
        if start not in unseen:
            continue
        pending, group = [start], set()
        while pending:
            node = pending.pop()
            if node not in unseen:
                continue
            unseen.remove(node)
            group.add(node)
            pending.extend(sorted(adjacent[node] & unseen))
        if len(group) >= 2:
            components.append([c for c in columns if c in group])
    components.sort(key=lambda values: (-len(values), tuple(values)))
    clusters = {f"Cluster_{i + 1}": values for i, values in enumerate(components)}
    membership = {taxon: key for key, values in clusters.items() for taxon in values}
    for edge in edges:
        edge["cluster"] = membership[edge["source"]]
    return clusters, edges
