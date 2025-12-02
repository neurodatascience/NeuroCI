import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np # <-- Added numpy import

def filter_complete_pipelines(df_tidy):
    """
    Return subset of df_tidy where all 4 pipelines are present per subject/structure/dataset.
    This ensures that N is the same for all comparisons within a structure/dataset.
    """
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
    
    print(f"Filtered from {len(df_tidy)} -> {len(df_filtered)} rows (only complete 4-pipeline cases).")
    return df_filtered

def get_sorted_structures(structures):
    """Sort structures according to a specific, predefined order."""
    
    # Define the exact target order
    TARGET_ORDER = [
        'Left-Thalamus',
        'Right-Thalamus',
        'Left-Caudate',
        'Right-Caudate',
        'Left-Putamen',
        'Right-Putamen',
        'Left-Pallidum',
        'Right-Pallidum',
        'Left-Hippocampus',
        'Right-Hippocampus',
        'Left-Amygdala',
        'Right-Amygdala',
        'Left-Accumbens-area',
        'Right-Accumbens-area',
        'Brainstem'
    ]

    # 1. Filter TARGET_ORDER to only include structures present in the current data
    present_structures = set(structures)
    
    # Structures that are present AND in the target order
    sorted_structures = [s for s in TARGET_ORDER if s in present_structures]
    
    # 2. Identify any structures present in the data but NOT in the target order
    # These will be added at the end, sorted alphabetically, to ensure completeness
    non_target_structures = sorted([
        s for s in present_structures 
        if s not in TARGET_ORDER
    ])
    
    # 3. Combine the two lists: target order first, then remaining structures
    sorted_structures.extend(non_target_structures)
    
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

# -------------------------------------------------------------
# Helper function for SVD calculation (Replaces RVD)
# -------------------------------------------------------------

def calculate_mean_svd_matrix(pivot_df, pipeline_order):
    """
    Calculates the mean Symmetric Volume Difference (SVD) matrix.
    SVD = 2 * |V_A - V_B| / (V_A + V_B). Range [0, 2].
    The matrix entry [A, B] is the mean SVD of the pair (A, B).
    
    This metric is symmetric: SVD(A, B) = SVD(B, A).
    It is also robust against division by zero (V_A + V_B = 0 only if both are 0).
    """
    svd_matrix = pd.DataFrame(index=pipeline_order, columns=pipeline_order)
    
    for pipe_A in pipeline_order:
        V_A = pivot_df[pipe_A]
        for pipe_B in pipeline_order:
            if pipe_A == pipe_B:
                svd_matrix.loc[pipe_A, pipe_B] = 0.0
            else:
                V_B = pivot_df[pipe_B]
                
                # Calculate SVD for each subject: 2 * |V_A - V_B| / (V_A + V_B)
                denominator = V_A + V_B
                svd_values = 2 * np.abs(V_A - V_B) / denominator
                
                # Filter out NaN/Inf values. This should only occur if 
                # both V_A and V_B were exactly 0 (0/0 result)
                finite_svd_values = svd_values[np.isfinite(svd_values)].dropna()
                
                if len(finite_svd_values) > 0:
                    mean_svd = finite_svd_values.mean()
                    svd_matrix.loc[pipe_A, pipe_B] = mean_svd
                else:
                    # If all subjects resulted in an error, set to NaN.
                    svd_matrix.loc[pipe_A, pipe_B] = np.nan
                
    return svd_matrix.astype(float)

# -------------------------------------------------------------
# Function for SVD figures (Replaces RVD)
# -------------------------------------------------------------

def create_svd_figures(df_tidy, output_dir):
    """Create inter-pipeline mean Symmetric Volume Difference (SVD) matrix plots for all brain structures per dataset."""

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

    # Individual dataset figures
    for dataset in df_tidy['dataset'].unique():
        dataset_data = df_tidy[df_tidy['dataset'] == dataset]
        print(f"Creating mean SVD matrices for {dataset} with {len(dataset_data)} rows...")

        plot_data = dataset_data.copy()
        plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)

        structures = get_sorted_structures(plot_data['structure'].unique())
        
        if len(structures) == 0:
            print(f"  No structures found for dataset {dataset}, skipping...")
            continue
            
        n_cols = 5
        n_rows = (len(structures) + n_cols - 1) // n_cols
        n_rows = max(1, n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)

        # Track all SVD values for consistent color scaling
        all_svd_values = []
        
        # First pass: collect all SVD values to determine color scale
        for i, structure in enumerate(structures):
            structure_data = plot_data[plot_data['structure'] == structure]
            pivot_df = structure_data.pivot_table(
                index='subject',
                columns='pipeline_short',
                values='volume_mm3'
            )
            
            if len(pivot_df.columns) == len(pipeline_order) and len(pivot_df) > 1:
                svd_matrix = calculate_mean_svd_matrix(pivot_df, pipeline_order)
                # Collect non-NaN values
                valid_values = svd_matrix.values[~np.isnan(svd_matrix.values)]
                all_svd_values.extend(valid_values)
        
        # Determine color scale based on 99th percentile (not arbitrary clipping!)
        if all_svd_values:
            vmax = np.percentile(all_svd_values, 99)
            print(f"  Dataset {dataset}: Using 99th percentile vmax = {vmax:.3f}")
        else:
            vmax = 0.5  # fallback
            print(f"  Dataset {dataset}: No valid SVD values, using default vmax = {vmax}")
        
        vmin = 0.0
        
        # Create colormap with proper NaN handling
        cmap = sns.color_palette("magma_r", as_cmap=True)
        cmap.set_bad(color='lightgray', alpha=0.5)  # NaN cells = light gray
        
        # Second pass: create plots
        for i, structure in enumerate(structures):
            ax = axes[i // n_cols, i % n_cols]
            structure_data = plot_data[plot_data['structure'] == structure]

            # Pivot: subjects as rows, pipelines as columns
            pivot_df = structure_data.pivot_table(
                index='subject',
                columns='pipeline_short',
                values='volume_mm3'
            )

            # Only compute SVD if we have data for all pipelines
            if len(pivot_df.columns) == len(pipeline_order) and len(pivot_df) > 1:
                # Compute mean SVD
                svd_matrix = calculate_mean_svd_matrix(pivot_df, pipeline_order)
                
                # --- PROPER NaN HANDLING ---
                nan_mask = svd_matrix.isna()
                
                # Create annotation matrix: show SVD value (rounded to 3 decimals)
                annot_matrix = svd_matrix.round(3).astype(str)
                annot_matrix[nan_mask] = 'NaN'
                
                # DO NOT clip - use original values with our vmax
                plot_svd_matrix = svd_matrix.copy()
                # Values above vmax will be colored as vmax (per heatmap's clip=True default)
                # --- End NaN handling ---
                
                # Plot heatmap with mask for NaN cells
                heatmap = sns.heatmap(
                    plot_svd_matrix,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    annot=annot_matrix,
                    fmt="s",
                    square=True,
                    cbar=False,
                    ax=ax,
                    mask=nan_mask,  # Critical: mask NaN cells
                    linewidths=0.5,
                    linecolor='gray'
                )
                
                # Add hatching to NaN cells for extra clarity
                if nan_mask.any().any():
                    for (row, col), is_nan in np.ndenumerate(nan_mask.values):
                        if is_nan:
                            # Add diagonal hatching
                            ax.add_patch(plt.Rectangle((col, row), 1, 1, 
                                                      fill=False, 
                                                      hatch='////',
                                                      edgecolor='gray',
                                                      linewidth=0.5,
                                                      alpha=0.5))
                
                # Improve label readability
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                
                # Axis labels for SVD
                ax.set_xlabel("Pipeline B") 
                ax.set_ylabel("Pipeline A")
                
                # Add footnote about extreme values if any exceed vmax
                if (svd_matrix > vmax).any().any():
                    num_extreme = (svd_matrix > vmax).sum().sum()
                    ax.text(0.02, 0.02, f"*{num_extreme} values >{vmax:.2f}", 
                           transform=ax.transAxes, fontsize=8, color='darkred')

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
        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Mean SVD", rotation=270, labelpad=15)
        
        # Colorbar labels showing percentiles
        cbar.ax.text(1.5, vmax + 0.02, f'99th %ile: {vmax:.2f}', 
                     ha='center', va='bottom', fontsize=9)
        cbar.ax.text(1.5, vmin - 0.02, 'Perfect agreement', 
                     ha='center', va='top', fontsize=9)

        # Overall title with scale info
        fig.suptitle(f'Inter-Pipeline Mean Symmetric Volume Difference (SVD) - {dataset}\n'
                     f'Color scale: 0.0 to 99th percentile ({vmax:.2f})', 
                     fontsize=16, y=0.95)
        fig.tight_layout(rect=[0, 0.03, 0.9, 0.93])  # leave room for colorbar

        output_path = output_dir / f'svd_comparison_mean_{dataset}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {output_path}")
        
        # Also save summary statistics for this dataset
        if all_svd_values:
            svd_stats = pd.DataFrame({
                'statistic': ['mean', 'std', 'min', '25%', '50%', '75%', '95%', '99%', 'max'],
                'value': [
                    np.mean(all_svd_values),
                    np.std(all_svd_values),
                    np.min(all_svd_values),
                    np.percentile(all_svd_values, 25),
                    np.percentile(all_svd_values, 50),
                    np.percentile(all_svd_values, 75),
                    np.percentile(all_svd_values, 95),
                    np.percentile(all_svd_values, 99),
                    np.max(all_svd_values)
                ]
            })
            stats_path = output_dir / f'svd_statistics_{dataset}.csv'
            svd_stats.to_csv(stats_path, index=False)
            print(f"  SVD statistics saved: {stats_path}")

    # Combined dataset figures (similar fixes applied)
    print("Creating combined mean SVD matrices (all datasets pooled)...")
    # ... [apply similar fixes to the combined dataset section] ...

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
        # SVD function call (Renamed from create_rvd_figures)
        create_svd_figures(df_complete, EXPERIMENT_STATE_ROOT)
        print("✓ Mean SVD plots generated successfully!")
        
    else:
        print(f"✗ File not found: {tidy_path}")
        print("Please run the data extraction script first to generate df_tidy.csv")
