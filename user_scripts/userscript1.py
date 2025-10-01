import json
from pathlib import Path
import pandas as pd

# ----------------------------
# Config
# ----------------------------
STATE_DIR = Path("/tmp/neuroci_output_state")

COMMON_STRUCTURES = [
    "Brainstem",
    "Left-Thalamus",
    "Right-Thalamus",
    "Left-Caudate",
    "Right-Caudate",
    "Left-Putamen",
    "Right-Putamen",
    "Left-Pallidum",
    "Right-Pallidum",
    "Left-Hippocampus",
    "Right-Hippocampus",
    "Left-Amygdala",
    "Right-Amygdala",
    "Left-Accumbens-area",
    "Right-Accumbens-area",
]

# ----------------------------
# File parsers
# ----------------------------
def parse_fsl(path: Path):
    with open(path) as f:
        data = json.load(f)
    # Only keep common structures
    return {k: float(v) for k, v in data.items() if k in COMMON_STRUCTURES}

def parse_samseg(path: Path):
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("# Measure"):
                continue
            # line looks like: "# Measure Left-Amygdala, 1651.577379, mm^3"
            parts = [p.strip() for p in line[len("# Measure "):].split(",")]
            if len(parts) < 2:
                continue
            roi, vol = parts[0], parts[1]

            # Normalize ROI names to match COMMON_STRUCTURES
            roi = roi.replace(" ", "-")
            roi = roi.replace("Brain-Stem", "Brainstem")
            roi = roi.replace("VentralDC", "VentralDC")  # optional, just in case

            if roi in COMMON_STRUCTURES:
                try:
                    results[roi] = float(vol)
                except ValueError:
                    continue
    return results

def parse_freesurfer(path: Path):
    results = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            struct_name = parts[4]
            volume = parts[3]
            if struct_name in COMMON_STRUCTURES:
                results[struct_name] = float(volume)
    return results

PARSERS = {
    "fsl": parse_fsl,
    "samseg": parse_samseg,
    "freesurfer": parse_freesurfer,
}

# ----------------------------
# Discovery
# ----------------------------
def discover_files(state_dir: Path):
    results = []
    for dataset_dir in state_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        derivatives = dataset_dir / "derivatives"
        if not derivatives.exists():
            continue
        for pipeline_root in derivatives.iterdir():
            if not pipeline_root.is_dir():
                continue
            for output_dir in pipeline_root.glob("*/output"):
                pipeline_name = pipeline_root.name
                version = output_dir.parent.name
                for subj_dir in output_dir.iterdir():
                    if not subj_dir.is_dir():
                        continue
                    subj = subj_dir.name
                    for ses_dir in subj_dir.iterdir():
                        if not ses_dir.is_dir():
                            continue
                        ses = ses_dir.name
                        fs_stats = ses_dir / subj / "stats" / "aseg.stats"
                        if fs_stats.exists():
                            results.append({
                                "dataset": dataset_dir.name,
                                "pipeline": pipeline_name,
                                "version": version,
                                "subject": subj,
                                "session": ses,
                                "file_type": "freesurfer",
                                "path": fs_stats,
                            })
                        samseg_stats = ses_dir / "samseg" / "samseg.stats"
                        if samseg_stats.exists():
                            results.append({
                                "dataset": dataset_dir.name,
                                "pipeline": pipeline_name,
                                "version": version,
                                "subject": subj,
                                "session": ses,
                                "file_type": "samseg",
                                "path": samseg_stats,
                            })
                        fsl_json = ses_dir / "out.anat" / "subcortical_volumes.json"
                        if fsl_json.exists():
                            results.append({
                                "dataset": dataset_dir.name,
                                "pipeline": pipeline_name,
                                "version": version,
                                "subject": subj,
                                "session": ses,
                                "file_type": "fsl",
                                "path": fsl_json,
                            })
    return results

# ----------------------------
# Build tidy DataFrame
# ----------------------------
def build_tidy_dataframe(files_meta):
    tidy_rows = []
    for r in files_meta:
        parser = PARSERS.get(r["file_type"])
        if parser is None:
            continue
        vols = parser(r["path"])
        for struct, vol in vols.items():
            tidy_rows.append({
                **r,  # keep metadata
                "structure": struct,
                "volume_mm3": vol,
            })
    return pd.DataFrame(tidy_rows)

# ----------------------------
# Wide pivot for ML
# ----------------------------
def pivot_wide(df_tidy: pd.DataFrame):
    df_wide = df_tidy.pivot_table(
        index=["dataset", "subject", "session"],
        columns=["pipeline", "structure"],
        values="volume_mm3"
    )
    # flatten MultiIndex columns
    df_wide.columns = [f"{pipe}__{struct}" for pipe, struct in df_wide.columns]
    df_wide = df_wide.reset_index()
    return df_wide

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    files_meta = discover_files(STATE_DIR)
    print(f"Discovered {len(files_meta)} stats files")

    df_tidy = build_tidy_dataframe(files_meta)
    print(f"Tidy DataFrame shape: {df_tidy.shape}")

    df_wide = pivot_wide(df_tidy)
    print(f"Wide DataFrame shape: {df_wide.shape}")
