from pathlib import Path

state_dir = Path("/tmp/neuroci_output_state")

# Which file names we care about
TARGET_FILES = {
    "freesurfer": "stats/aseg.stats",
    "samseg": "samseg/samseg.stats",
    "fsl": "out.anat/subcortical_volumes.json",
}

results = []

for dataset_dir in state_dir.iterdir():
    if not dataset_dir.is_dir():
        continue

    derivatives = dataset_dir / "derivatives"
    if not derivatives.exists():
        continue

    # pipelines like "freesurfer741ants243", "fslanat6071ants243", etc
    for pipeline_root in derivatives.iterdir():
        if not pipeline_root.is_dir():
            continue

        # pipeline version (e.g. "7.4.1"), then "output"
        for output_dir in pipeline_root.glob("*/output"):
            pipeline_name = pipeline_root.name  # e.g. freesurfer741ants243
            version = output_dir.parent.name

            # subjects inside output/
            for subj_dir in output_dir.iterdir():
                if not subj_dir.is_dir():
                    continue
                subj = subj_dir.name  # e.g. sub-09114

                for ses_dir in subj_dir.iterdir():
                    if not ses_dir.is_dir():
                        continue
                    ses = ses_dir.name  # e.g. ses-1pre

                    # Now we need to find the right file depending on pipeline
                    # Check freesurfer style
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

                    # Check samseg style
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

                    # Check fsl style
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

print(f"Found {len(results)} stats files")
for r in results[:10]:  # just preview
    print(r)
