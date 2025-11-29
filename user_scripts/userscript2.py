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
# Helper function for RVD calculation
# -------------------------------------------------------------

def calculate_mean_rvd_matrix(pivot_df, pipeline_order):
    """
    Calculates the mean Relative Volume Difference (RVD) matrix.
    RVD of V_B relative to V_A is (V_B - V_A) / V_A
    The matrix entry [A, B] is the mean RVD of B relative to A.
    """
    rvd_matrix = pd.DataFrame(index=pipeline_order, columns=pipeline_order)
    
    for ref_pipe in pipeline_order:
        V_ref = pivot_df[ref_pipe]
        for comp_pipe in pipeline_order:
            if ref_pipe == comp_pipe:
                # RVD of a volume relative to itself is 0
                rvd_matrix.loc[ref_pipe, comp_pipe] = 0.0
            else:
                V_comp = pivot_df[comp_pipe]
                # Calculate RVD for each subject: (V_comp - V_ref) / V_ref
                rvd_values = (V_comp - V_ref) / V_ref
                # Calculate the mean RVD across all subjects
                mean_rvd = rvd_values.mean()
                rvd_matrix.loc[ref_pipe, comp_pipe] = mean_rvd
                
    # Convert to float to ensure proper visualization
    return rvd_matrix.astype(float)

# -------------------------------------------------------------
# Function for RVD figures (Axis labels and ADJUSTED Over/Under-estimation labels RE-ADDED)
# -------------------------------------------------------------

def create_rvd_figures(df_tidy, output_dir):
    """Create inter-pipeline mean Relative Volume Difference (RVD) matrix plots for all brain structures per dataset."""

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

    # Define color scale and palette: RVD can range widely, setting a safe symmetric scale
    # Using -1.0 to 1.0 corresponds to -100% to +100% relative difference
    vmin, vmax, center = -1.0, 1.0, 0.0 
    cmap = sns.color_palette("RdBu_r", as_cmap=True) # Diverging map, centered at 0

    # Individual dataset figures
    for dataset in df_tidy['dataset'].unique():
        dataset_data = df_tidy[df_tidy['dataset'] == dataset]
        print(f"Creating mean RVD matrices for {dataset} with {len(dataset_data)} rows...")

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

        for i, structure in enumerate(structures):
            ax = axes[i // n_cols, i % n_cols]
            structure_data = plot_data[plot_data['structure'] == structure]

            # Pivot: subjects as rows, pipelines as columns
            pivot_df = structure_data.pivot_table(
                index='subject',  # For individual datasets, subject ID alone is sufficient
                columns='pipeline_short',
                values='volume_mm3'
            )

            # Only compute RVD if we have data for all pipelines
            if len(pivot_df.columns) == len(pipeline_order) and len(pivot_df) > 1:
                # Compute mean RVD
                rvd_matrix = calculate_mean_rvd_matrix(pivot_df, pipeline_order)

                # Plot heatmap
                sns.heatmap(
                    rvd_matrix,
                    vmin=vmin, vmax=vmax, center=center,
                    cmap=cmap,
                    annot=True, 
                    fmt=".2f", # Display as fractional value (e.g., 0.10 means 10% relative difference)
                    square=True,
                    cbar=False,
                    ax=ax
                )

                # Improve label readability
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                # RE-ADDED: Axis labels for RVD
                ax.set_xlabel("Comparison Pipeline") 
                ax.set_ylabel("Reference Pipeline") 

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
        cbar.set_label("Mean RVD (Fractional)", rotation=270, labelpad=15)
        
        # RE-ADDED and ADJUSTED: Overestimation/Underestimation labels
        cbar.ax.text(1.5, vmax + 0.1, '+Overestimation', ha='center', va='top', fontsize=10) # Moved up
        cbar.ax.text(1.5, vmin - 0.1, '-Underestimation', ha='center', va='bottom', fontsize=10) # Moved down


        # Overall title
        fig.suptitle(f'Inter-Pipeline Mean Relative Volume Difference (RVD) - {dataset}', fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])  # leave room for colorbar on the right

        output_path = output_dir / f'rvd_comparison_mean_{dataset}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

    # Combined dataset figures (pooled across all datasets)
    print("Creating combined mean RVD matrices (all datasets pooled)...")
    plot_data = df_tidy.copy()
    plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)
    
    structures = get_sorted_structures(plot_data['structure'].unique())
    
    if len(structures) == 0:
        print("  No structures found for combined data, skipping...")
        return
        
    n_cols = 5
    n_rows = (len(structures) + n_cols - 1) // n_cols
    n_rows = max(1, n_rows)
    
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

        # Only compute RVD if we have data for all pipelines
        if len(pivot_df.columns) == len(pipeline_order) and len(pivot_df) > 1:
            # Compute mean RVD
            rvd_matrix = calculate_mean_rvd_matrix(pivot_df, pipeline_order)

            # Plot heatmap
            sns.heatmap(
                rvd_matrix,
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
            # RE-ADDED: Axis labels for RVD
            ax.set_xlabel("Comparison Pipeline") 
            ax.set_ylabel("Reference Pipeline")

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
    cbar.set_label("Mean RVD (Fractional)", rotation=270, labelpad=15)
    
    # RE-ADDED and ADJUSTED: Overestimation/Underestimation labels
    cbar.ax.text(1.5, vmax + 0.1, '+Overestimation', ha='center', va='top', fontsize=10) # Moved up
    cbar.ax.text(1.5, vmin - 0.1, '-Underestimation', ha='center', va='bottom', fontsize=10) # Moved down


    # Overall title
    fig.suptitle(f'Inter-Pipeline Mean Relative Volume Difference (RVD) - All Datasets Combined', fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    output_path = output_dir / f'rvd_comparison_mean_ALL_DATASETS.png'
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
        # RVD function call
        create_rvd_figures(df_complete, EXPERIMENT_STATE_ROOT)
        print("✓ Mean RVD plots generated successfully!")
        
    else:
        print(f"✗ File not found: {tidy_path}")
        print("Please run the data extraction script first to generate df_tidy.csv")
