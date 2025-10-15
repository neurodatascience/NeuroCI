import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def create_distribution_figures(df_tidy, output_dir):
    """Create overlapping histograms (counts) for all pipelines per structure."""
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12

    pipeline_mapping = {
        'fslanat6071ants243': 'FSL6071',
        'freesurfer741ants243': 'FS741',
        'freesurfer8001ants243': 'FS8001', 
        'samseg8001ants243': 'SAMSEG8'
    }
    pipeline_order = list(pipeline_mapping.values())
    palette = sns.color_palette("tab10", len(pipeline_order))

    for dataset in df_tidy['dataset'].unique():
        dataset_data = df_tidy[df_tidy['dataset'] == dataset]
        print(f"Creating count histograms for {dataset} with {len(dataset_data)} rows...")
        
        plot_data = dataset_data.copy()
        plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)
        
        structures = plot_data['structure'].unique()
        n_cols = 5
        n_rows = (len(structures) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
        
        for i, structure in enumerate(structures):
            ax = axes[i // n_cols, i % n_cols]
            structure_data = plot_data[plot_data['structure'] == structure]
            
            for j, pipeline in enumerate(pipeline_order):
                subset = structure_data[structure_data['pipeline_short'] == pipeline]
                sns.histplot(
                    data=subset,
                    x='volume_mm3',
                    bins=30,
                    element='step',   # outlines instead of bars (less clutter)
                    fill=True,
                    alpha=0.4,
                    color=palette[j],
                    label=pipeline,
                    ax=ax,
                    stat='count'      # <-- this makes it counts, not density
                )
            
            ax.set_title(structure)
            ax.set_xlabel('Volume (mm³)')
            ax.set_ylabel('Count')
            ax.legend()
        
        total_plots = n_rows * n_cols
        for k in range(len(structures), total_plots):
            fig.delaxes(axes[k // n_cols, k % n_cols])
        
        fig.suptitle(f'Pipeline Distributions (Counts) - {dataset}', fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_path = output_dir / f'distribution_comparison_counts_{dataset}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

# ----------------------------
# Main - using same config as data extraction script
# ----------------------------
if __name__ == "__main__":
    # Use identical directory structure
    EXPERIMENT_STATE_ROOT = Path(__file__).resolve().parents[1] / "experiment_state" / "figures"
    tidy_path = EXPERIMENT_STATE_ROOT / "df_tidy.csv"
    
    print(f"Looking for data file: {tidy_path}")
    
    if tidy_path.exists():
        # Load the data
        df_tidy = pd.read_csv(tidy_path)
        print(f"✓ Loaded df_tidy.csv")
        print(f"  Shape: {df_tidy.shape}")
        print(f"  Datasets: {df_tidy['dataset'].unique().tolist()}")
        print(f"  Pipelines: {df_tidy['pipeline'].unique().tolist()}")
        
        # Create output directory if it doesn't exist
        EXPERIMENT_STATE_ROOT.mkdir(parents=True, exist_ok=True)
        
        # Generate figures
        create_distribution_figures(df_tidy, EXPERIMENT_STATE_ROOT)
        print("✓ Boxen plots generated successfully!")
        
    else:
        print(f"✗ File not found: {tidy_path}")
        print("Please run the data extraction script first to generate df_tidy.csv")
