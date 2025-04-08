import os
from pathlib import Path
import matplotlib.pyplot as plt

# Define base paths
idp_root = Path("/tmp/neuroci_idp_state")
experiment_state_root = Path(__file__).resolve().parents[1] / "experiment_state" / "figures"
experiment_state_root.mkdir(parents=True, exist_ok=True)

data = {}

# Walk through datasets and pipelines
for dataset_dir in idp_root.iterdir():
    if not dataset_dir.is_dir():
        continue

    for pipeline_dir in dataset_dir.glob("derivatives/*/*/idp"):
        file_count_path = pipeline_dir / "file_count.txt"
        if file_count_path.exists():
            try:
                count = int(file_count_path.read_text().strip())
                label = f"{dataset_dir.name}/{pipeline_dir.parts[-4]}-{pipeline_dir.parts[-3]}"
                data[label] = count
            except Exception as e:
                print(f"Error reading {file_count_path}: {e}")

# Plotting
if data:
    labels = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("File Count")
    plt.title("File Counts per Pipeline")
    plt.tight_layout()

    output_path = experiment_state_root / "file_counts.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
else:
    print("No file_count.txt data found.")

