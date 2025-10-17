import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def filter_complete_pipelines(df_tidy):
    """Return subset of df_tidy where all 4 pipelines are present per subject/structure/dataset."""
    required_pipelines = {
        'fslanat6071ants243',
        'freesurfer741ants243',
        'freesurfer8001ants243',
        'samseg8001ants243'
    }

    # Count how many unique pipelines exist per subject/structure/dataset
    counts = (
        df_tidy.groupby(['dataset', 'subject', 'structure'])['pipeline']
        .nunique()
        .reset_index(name='n_pipelines')
    )

    # Keep only those combinations where all 4 pipelines are present
    complete = counts[counts['n_pipelines'] == len(required_pipelines)]

    # Merge back to original data to keep only valid rows
    df_filtered = df_tidy.merge(complete[['dataset', 'subject', 'structure']], on=['dataset', 'subject', 'structure'])
    
    print(f"Filtered from {len(df_tidy)} → {len(df_filtered)} rows (only complete 4-pipeline cases).")
    return df_filtered
    

def create_distribution_figures(df_tidy, output_dir):
    """Create overlapping histograms (counts) for all pipelines per structure with accurate n."""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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
        dataset_data = df_tidy[df_tidy['dataset'] == dataset].copy()
        dataset_data['pipeline_short'] = dataset_data['pipeline'].map(pipeline_mapping)

        structures = dataset_data['structure'].unique()
        n_cols = 5
        n_rows = (len(structures) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)

        for i, structure in enumerate(structures):
            ax = axes[i // n_cols, i % n_cols]
            structure_data = dataset_data[dataset_data['structure'] == structure]

            # n_points is the number of unique subjects (same for all pipelines)
            n_points = structure_data['subject'].nunique()

            for j, pipeline in enumerate(pipeline_order):
                subset = structure_data[structure_data['pipeline_short'] == pipeline]
                sns.histplot(
                    data=subset,
                    x='volume_mm3',
                    bins=30,
                    element='step',
                    fill=True,
                    alpha=0.4,
                    color=palette[j],
                    label=pipeline,
                    ax=ax,
                    stat='count'
                )

            ax.set_title(f"{structure}\n(n={n_points})")
            ax.set_xlabel('Volume (mm³)')
            ax.set_ylabel('Count')
            ax.legend()

        # Remove unused axes
        total_plots = n_rows * n_cols
        for k in range(len(structures), total_plots):
            fig.delaxes(axes[k // n_cols, k % n_cols])

        fig.suptitle(f'Pipeline Distributions (Counts) - {dataset}', fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = output_dir / f'distribution_comparison_counts_{dataset}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved: {output_path}")


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
            
            n_points = len(structure_data[structure_data['pipeline_short'] == pipeline_order[0]]['subject'].unique())
            ax.set_title(f"{structure}\n(n={n_points})")
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

def create_correlation_figures(df_tidy, output_dir):
    """Create inter-pipeline Pearson correlation matrix plots for all brain structures per dataset."""

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

    for dataset in df_tidy['dataset'].unique():
        dataset_data = df_tidy[df_tidy['dataset'] == dataset]
        print(f"Creating correlation matrices for {dataset} with {len(dataset_data)} rows...")

        plot_data = dataset_data.copy()
        plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)

        structures = plot_data['structure'].unique()
        n_cols = 5
        n_rows = (len(structures) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)

        # Define color scale and palette
        vmin, vmax, center = -1, 1, 0
        cmap = sns.color_palette("RdBu_r", as_cmap=True)

        for i, structure in enumerate(structures):
            ax = axes[i // n_cols, i % n_cols]
            structure_data = plot_data[plot_data['structure'] == structure]

            # Pivot: subjects as rows, pipelines as columns
            pivot_df = structure_data.pivot_table(
                index='subject',
                columns='pipeline_short',
                values='volume_mm3'
            )

            # Compute Pearson correlation
            corr = pivot_df.corr(method='pearson')

            # Plot heatmap
            sns.heatmap(
                corr,
                vmin=vmin, vmax=vmax, center=center,
                cmap=cmap,
                annot=True, fmt=".2f",
                square=True,
                cbar=False,
                ax=ax
            )

            # Improve label readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xlabel("")
            ax.set_ylabel("")

            n_points = pivot_df.shape[0]
            ax.set_title(f"{structure}\n(n={n_points})")

        # Remove empty subplots
        total_plots = n_rows * n_cols
        for k in range(len(structures), total_plots):
            fig.delaxes(axes[k // n_cols, k % n_cols])

        # Shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])  # [left, bottom, width, height]
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Pearson correlation", rotation=270, labelpad=15)

        # Overall title
        fig.suptitle(f'Inter-Pipeline Correlations (Pearson) - {dataset}', fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])  # leave room for colorbar on the right

        output_path = output_dir / f'correlation_comparison_{dataset}.png'
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
        df_complete = filter_complete_pipelines(df_tidy) # Where all 4 pipelines have completed succesfully
        create_distribution_figures(df_complete, EXPERIMENT_STATE_ROOT)
        print("✓ Boxen plots generated successfully!")
        create_correlation_figures(df_complete, EXPERIMENT_STATE_ROOT)
        print("✓ Volume Pearson correlation plots generated successfully!")
        
    else:
        print(f"✗ File not found: {tidy_path}")
        print("Please run the data extraction script first to generate df_tidy.csv")
