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
    try:
        # Read the CSV file
        df = pd.read_csv(path)
        
        # Normalize ROI names and filter for common structures
        for _, row in df.iterrows():
            roi = row['ROI']
            
            # Normalize ROI names
            # Replace Brain-Stem with Brainstem
            roi = roi.replace("Brain-Stem", "Brainstem")
            # Replace underscores with hyphens
            roi = roi.replace("_", "-")
            
            if roi in COMMON_STRUCTURES:
                results[roi] = float(row['volume_mm3'])
                
    except Exception as e:
        print(f"Error parsing SAMSEG CSV {path}: {e}")
        
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
            pipeline_name = pipeline_root.name
            for version_dir in pipeline_root.iterdir():
                if not version_dir.is_dir():
                    continue
                version = version_dir.name
                output_dir = version_dir / "output"
                if not output_dir.exists():
                    continue
                    
                for subj_dir in output_dir.iterdir():
                    if not subj_dir.is_dir():
                        continue
                    subj = subj_dir.name
                    for ses_dir in subj_dir.iterdir():
                        if not ses_dir.is_dir():
                            continue
                        ses = ses_dir.name
                        
                        # Look for FreeSurfer files
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
                        
                        # Look for SAMSEG CSV files
                        samseg_csv = ses_dir / "samseg" / "samseg.csv"
                        if samseg_csv.exists():
                            results.append({
                                "dataset": dataset_dir.name,
                                "pipeline": pipeline_name,
                                "version": version,
                                "subject": subj,
                                "session": ses,
                                "file_type": "samseg",
                                "path": samseg_csv,
                            })
                        
                        # Look for FSL files
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
        try:
            vols = parser(r["path"])
            for struct, vol in vols.items():
                tidy_rows.append({
                    **r,  # keep metadata
                    "structure": struct,
                    "volume_mm3": vol,
                })
        except Exception as e:
            print(f"Error parsing {r['path']}: {e}")
            continue
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
    
    # Save tidy DataFrame
    df_tidy.to_csv("df_tidy.csv", index=False)
    print("Saved tidy DataFrame to df_tidy.csv")

    df_wide = pivot_wide(df_tidy)
    print(f"Wide DataFrame shape: {df_wide.shape}")
    
    # Save wide DataFrame
    df_wide.to_csv("df_wide.csv", index=False)
    print("Saved wide DataFrame to df_wide.csv")
    
    # Print pipeline counts for verification
    print("\nPipeline counts:")
    print(df_tidy['pipeline'].value_counts())
    
    # Print file type counts for verification  
    print("\nFile type counts:")
    print(df_tidy['file_type'].value_counts())
    
    # Print structure counts to verify SAMSEG is working
    print("\nStructure counts by pipeline:")
    samseg_structures = df_tidy[df_tidy['pipeline'].str.contains('samseg')]['structure'].value_counts()
    print("SAMSEG structures:")
    print(samseg_structures)
