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

def get_sorted_structures(structures):
    """Sort structures: left-right pairs together, ordered by structure name, include non-bilateral structures."""
    if len(structures) == 0:
        return []
    
    # Extract unique structure names (without hemisphere) - handle Title Case
    base_structures = sorted(set([s.replace('Left-', '').replace('Right-', '') for s in structures]))
    
    # Create pairs: left then right for each base structure
    sorted_structures = []
    for base in base_structures:
        left = f"Left-{base}"
        right = f"Right-{base}"
        if left in structures:
            sorted_structures.append(left)
        if right in structures:
            sorted_structures.append(right)
    
    # Add non-bilateral structures that don't follow the Left-/Right- pattern
    bilateral_bases = [f"Left-{base}" for base in base_structures] + [f"Right-{base}" for base in base_structures]
    non_bilateral_structures = [s for s in structures if s not in bilateral_bases]
    
    # Add non-bilateral structures at the end
    sorted_structures.extend(sorted(non_bilateral_structures))
    
    return sorted_structures

def count_unique_subjects(structure_data):
    """Count unique subject-dataset combinations to avoid double-counting same subject ID across datasets."""
    return len(structure_data[['dataset', 'subject']].drop_duplicates())

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

    # Individual dataset figures
    for dataset in df_tidy['dataset'].unique():
        dataset_data = df_tidy[df_tidy['dataset'] == dataset]
        print(f"Creating count histograms for {dataset} with {len(dataset_data)} rows...")
        
        plot_data = dataset_data.copy()
        plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)
        
        structures = get_sorted_structures(plot_data['structure'].unique())
        
        # Skip if no structures found
        if len(structures) == 0:
            print(f"  No structures found for dataset {dataset}, skipping...")
            continue
            
        n_cols = 5
        n_rows = (len(structures) + n_cols - 1) // n_cols
        
        # Ensure at least 1 row
        n_rows = max(1, n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
        
        for i, structure in enumerate(structures):
            ax = axes[i // n_cols, i % n_cols]
            structure_data = plot_data[plot_data['structure'] == structure]
            
            for j, pipeline in enumerate(pipeline_order):
                subset = structure_data[structure_data['pipeline_short'] == pipeline]
                if len(subset) > 0:  # Only plot if data exists
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
            
            if len(structure_data) > 0:
                # For individual datasets, just count unique subjects
                n_points = len(structure_data['subject'].unique())
                ax.set_title(f"{structure}\n(n={n_points})")
                ax.set_xlabel('Volume (mm³)')
                ax.set_ylabel('Count')
                ax.legend()
        
        # Remove empty subplots
        total_plots = n_rows * n_cols
        for k in range(len(structures), total_plots):
            fig.delaxes(axes[k // n_cols, k % n_cols])
        
        fig.suptitle(f'Pipeline Distributions (Counts) - {dataset}', fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_path = output_dir / f'distribution_comparison_counts_{dataset}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

    # Combined dataset figure (pooled across all datasets)
    print("Creating combined count histograms (all datasets pooled)...")
    plot_data = df_tidy.copy()
    plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)
    
    structures = get_sorted_structures(plot_data['structure'].unique())
    
    if len(structures) == 0:
        print("  No structures found for combined data, skipping...")
        return
        
    n_cols = 5
    n_rows = (len(structures) + n_cols - 1) // n_cols
    n_rows = max(1, n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    
    for i, structure in enumerate(structures):
        ax = axes[i // n_cols, i % n_cols]
        structure_data = plot_data[plot_data['structure'] == structure]
        
        for j, pipeline in enumerate(pipeline_order):
            subset = structure_data[structure_data['pipeline_short'] == pipeline]
            if len(subset) > 0:  # Only plot if data exists
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
        
        if len(structure_data) > 0:
            # For combined data, count unique subject-dataset combinations
            n_points = count_unique_subjects(structure_data)
            ax.set_title(f"{structure}\n(n={n_points})")
            ax.set_xlabel('Volume (mm³)')
            ax.set_ylabel('Count')
            ax.legend()
    
    total_plots = n_rows * n_cols
    for k in range(len(structures), total_plots):
        fig.delaxes(axes[k // n_cols, k % n_cols])
    
    fig.suptitle('Pipeline Distributions (Counts) - All Datasets Combined', fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = output_dir / f'distribution_comparison_counts_ALL_DATASETS.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def create_correlation_figures(df_tidy, output_dir):
    """Create inter-pipeline Pearson and Spearman correlation matrix plots for all brain structures per dataset."""

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

    # Define color scale and palette
    vmin, vmax, center = -1, 1, 0
    cmap = sns.color_palette("RdBu_r", as_cmap=True)

    # Individual dataset figures
    for dataset in df_tidy['dataset'].unique():
        dataset_data = df_tidy[df_tidy['dataset'] == dataset]
        print(f"Creating correlation matrices for {dataset} with {len(dataset_data)} rows...")

        plot_data = dataset_data.copy()
        plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)

        structures = get_sorted_structures(plot_data['structure'].unique())
        
        if len(structures) == 0:
            print(f"  No structures found for dataset {dataset}, skipping...")
            continue
            
        n_cols = 5
        n_rows = (len(structures) + n_cols - 1) // n_cols
        n_rows = max(1, n_rows)
        
        # Create both Pearson and Spearman figures for this dataset
        for corr_method in ['pearson', 'spearman']:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)

            for i, structure in enumerate(structures):
                ax = axes[i // n_cols, i % n_cols]
                structure_data = plot_data[plot_data['structure'] == structure]

                # Pivot: subjects as rows, pipelines as columns
                pivot_df = structure_data.pivot_table(
                    index='subject',  # For individual datasets, subject ID alone is sufficient
                    columns='pipeline_short',
                    values='volume_mm3'
                )

                # Only compute correlation if we have data for all pipelines
                if len(pivot_df.columns) == len(pipeline_order) and len(pivot_df) > 1:
                    # Compute correlation (Pearson or Spearman)
                    corr = pivot_df.corr(method=corr_method)

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
                else:
                    ax.text(0.5, 0.5, "Insufficient data", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{structure}\n(no data)")

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
            cbar.set_label(f"{corr_method.capitalize()} correlation", rotation=270, labelpad=15)

            # Overall title
            fig.suptitle(f'Inter-Pipeline Correlations ({corr_method.capitalize()}) - {dataset}', fontsize=18)
            fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])  # leave room for colorbar on the right

            output_path = output_dir / f'correlation_comparison_{corr_method}_{dataset}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved: {output_path}")

    # Combined dataset figures (pooled across all datasets)
    print("Creating combined correlation matrices (all datasets pooled)...")
    plot_data = df_tidy.copy()
    plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)
    
    structures = get_sorted_structures(plot_data['structure'].unique())
    
    if len(structures) == 0:
        print("  No structures found for combined data, skipping...")
        return
        
    n_cols = 5
    n_rows = (len(structures) + n_cols - 1) // n_cols
    n_rows = max(1, n_rows)
    
    # Create both Pearson and Spearman figures for combined data
    for corr_method in ['pearson', 'spearman']:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)

        for i, structure in enumerate(structures):
            ax = axes[i // n_cols, i % n_cols]
            structure_data = plot_data[plot_data['structure'] == structure]

            # For combined data, create unique subject IDs by combining dataset and subject
            structure_data = structure_data.copy()
            structure_data['subject_dataset'] = structure_data['dataset'] + '_' + structure_data['subject']

            # Pivot: subject-dataset combinations as rows, pipelines as columns
            pivot_df = structure_data.pivot_table(
                index='subject_dataset',  # Use unique subject-dataset combinations
                columns='pipeline_short',
                values='volume_mm3'
            )

            # Only compute correlation if we have data for all pipelines
            if len(pivot_df.columns) == len(pipeline_order) and len(pivot_df) > 1:
                # Compute correlation (Pearson or Spearman)
                corr = pivot_df.corr(method=corr_method)

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
            else:
                ax.text(0.5, 0.5, "Insufficient data", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{structure}\n(no data)")

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
        cbar.set_label(f"{corr_method.capitalize()} correlation", rotation=270, labelpad=15)

        # Overall title
        fig.suptitle(f'Inter-Pipeline Correlations ({corr_method.capitalize()}) - All Datasets Combined', fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])

        output_path = output_dir / f'correlation_comparison_{corr_method}_ALL_DATASETS.png'
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
        print("✓ Distribution plots generated successfully!")
        create_correlation_figures(df_complete, EXPERIMENT_STATE_ROOT)
        print("✓ Correlation plots (Pearson & Spearman) generated successfully!")
        
    else:
        print(f"✗ File not found: {tidy_path}")
        print("Please run the data extraction script first to generate df_tidy.csv")
