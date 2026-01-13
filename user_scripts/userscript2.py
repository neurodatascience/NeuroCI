import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np 

def filter_complete_pipelines(df_tidy):
    """
    Return subset of df_tidy using STRICT Session-Level Complete Case Analysis.
    """
    required_pipelines = {
        'fslanat6071ants243',
        'freesurfer741ants243',
        'freesurfer8001ants243',
        'samseg8001ants243'
    }

    print(f"Starting strict filtering. Initial rows: {len(df_tidy)}")

    # 1. Count pipelines per structure per session
    counts = (
        df_tidy.groupby(['dataset', 'subject', 'session', 'structure'])['pipeline']
        .nunique()
        .reset_index(name='n_pipelines')
    )

    # 2. Identify "Bad Sessions": Any session that has at least one structure with < 4 pipelines
    incomplete_structures = counts[counts['n_pipelines'] < len(required_pipelines)]
    
    bad_sessions = incomplete_structures[['dataset', 'subject', 'session']].drop_duplicates()
    
    if len(bad_sessions) > 0:
        print(f"Found {len(bad_sessions)} sessions with at least one incomplete structure.")
        
        # 3. Perform Anti-Join to drop these bad sessions entirely
        merged = df_tidy.merge(bad_sessions, 
                               on=['dataset', 'subject', 'session'], 
                               how='left', 
                               indicator=True)
        
        df_filtered = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        print("No incomplete sessions found.")
        df_filtered = df_tidy.copy()

    # 4. Final Sanity Check
    final_counts = (
        df_filtered.groupby(['dataset', 'subject', 'session', 'structure'])['pipeline']
        .nunique()
        .reset_index(name='n_pipelines')
    )
    if not final_counts.empty:
        assert final_counts['n_pipelines'].min() == len(required_pipelines), "Logic Error: Incomplete pipelines remain!"

    print(f"Filtered from {len(df_tidy)} -> {len(df_filtered)} rows.")
    
    return df_filtered

def get_sorted_structures(structures):
    """Sort structures according to a specific, predefined order."""
    
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

    present_structures = set(structures)
    sorted_structures = [s for s in TARGET_ORDER if s in present_structures]
    non_target_structures = sorted([s for s in present_structures if s not in TARGET_ORDER])
    sorted_structures.extend(non_target_structures)
    
    return sorted_structures

def count_unique_scans(structure_data):
    """Count unique dataset-subject-session combinations."""
    return len(structure_data[['dataset', 'subject', 'session']].drop_duplicates())

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

    # Helper for layout logic
    def plot_distributions(data_subset, title_suffix, filename):
        structures = get_sorted_structures(data_subset['structure'].unique())
        if len(structures) == 0: return

        # Calculate N once
        n_points = count_unique_scans(data_subset)

        n_cols = 5
        n_rows = max(1, (len(structures) + n_cols - 1) // n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
        
        for i, structure in enumerate(structures):
            ax = axes[i // n_cols, i % n_cols]
            structure_data = data_subset[data_subset['structure'] == structure]
            structure_data = structure_data.copy()
            structure_data['pipeline_short'] = structure_data['pipeline'].map(pipeline_mapping)
            
            for j, pipeline in enumerate(pipeline_order):
                subset = structure_data[structure_data['pipeline_short'] == pipeline]
                if len(subset) > 0:
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
            
            # Simplified Title (removed n)
            ax.set_title(f"{structure}")
            ax.set_xlabel('Volume (mm³)')
            ax.set_ylabel('Count')
            # Keeping legend on all plots as per original script
            ax.legend()
        
        total_plots = n_rows * n_cols
        for k in range(len(structures), total_plots):
            fig.delaxes(axes[k // n_cols, k % n_cols])
        
        fig.suptitle(f'Pipeline Distributions (Counts) - {title_suffix}', fontsize=18)
        
        # N Summary at bottom
        fig.text(0.5, 0.02, f"N = {n_points} Scans", ha='center', fontsize=14, fontweight='bold')

        # Adjusted margin for bottom text
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        output_path = output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

    # Individual dataset figures
    for dataset in df_tidy['dataset'].unique():
        dataset_data = df_tidy[df_tidy['dataset'] == dataset].copy()
        print(f"Creating count histograms for {dataset}...")
        plot_distributions(dataset_data, dataset, f'distribution_comparison_counts_{dataset}.png')

    # Combined dataset figure
    print("Creating combined count histograms...")
    plot_distributions(df_tidy.copy(), "All Datasets Combined", 'distribution_comparison_counts_ALL_DATASETS.png')

def create_correlation_figures(df_tidy, output_dir):
    """Create inter-pipeline Pearson and Spearman correlation matrix plots."""

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12

    pipeline_mapping = {
        'fslanat6071ants243': 'FSL6071',
        'freesurfer741ants243': 'FS741',
        'freesurfer8001ants243': 'FS8001',
        'samseg8001ants243': 'SAMSEG8'
    }
    # Correlation uses alphabetical order from pivot_table columns
    vmin, vmax, center = -1, 1, 0
    cmap = sns.color_palette("RdBu_r", as_cmap=True)

    def plot_correlations(data_subset, title_suffix, filename_prefix):
        structures = get_sorted_structures(data_subset['structure'].unique())
        if len(structures) == 0: return
        
        n_points = count_unique_scans(data_subset)
        data_subset = data_subset.copy()
        data_subset['pipeline_short'] = data_subset['pipeline'].map(pipeline_mapping)

        n_cols = 5
        n_rows = max(1, (len(structures) + n_cols - 1) // n_cols)
        
        for corr_method in ['pearson', 'spearman']:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)
            
            for i, structure in enumerate(structures):
                ax = axes[i // n_cols, i % n_cols]
                structure_data = data_subset[data_subset['structure'] == structure].copy()
                
                if 'dataset' in data_subset.columns and data_subset['dataset'].nunique() > 1:
                     structure_data['scan_id'] = structure_data['dataset'] + '_' + structure_data['subject'] + '_' + structure_data['session']
                else:
                     structure_data['scan_id'] = structure_data['subject'].astype(str) + '_' + structure_data['session'].astype(str)

                pivot_df = structure_data.pivot_table(
                    index='scan_id', 
                    columns='pipeline_short',
                    values='volume_mm3'
                )

                if len(pivot_df.columns) >= 2 and len(pivot_df) > 1:
                    corr = pivot_df.corr(method=corr_method)
                    sns.heatmap(corr, vmin=vmin, vmax=vmax, center=center, cmap=cmap, annot=True, fmt=".2f", square=True, cbar=False, ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                    ax.set_xlabel("") 
                    ax.set_ylabel("") 
                    # Simplified Title (removed n)
                    ax.set_title(f"{structure}")
                else:
                    ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{structure}\n(no data)")

            total_plots = n_rows * n_cols
            for k in range(len(structures), total_plots):
                fig.delaxes(axes[k // n_cols, k % n_cols])

            cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(f"{corr_method.capitalize()} correlation", rotation=270, labelpad=15)

            fig.suptitle(f'Inter-Pipeline Correlations ({corr_method.capitalize()}) - {title_suffix}', fontsize=18)
            
            # N Summary at bottom
            fig.text(0.5, 0.02, f"N = {n_points} Scans", ha='center', fontsize=14, fontweight='bold')

            # Adjusted margin for bottom text
            fig.tight_layout(rect=[0, 0.05, 0.9, 0.95])
            
            if title_suffix == "All Datasets Combined":
                 output_path = output_dir / f'{filename_prefix}_{corr_method}_ALL_DATASETS.png'
            else:
                 output_path = output_dir / f'{filename_prefix}_{corr_method}_{title_suffix}.png'

            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved: {output_path}")

    for dataset in df_tidy['dataset'].unique():
        dataset_data = df_tidy[df_tidy['dataset'] == dataset].copy()
        print(f"Creating correlation matrices for {dataset}...")
        plot_correlations(dataset_data, dataset, 'correlation_comparison')

    print("Creating combined correlation matrices...")
    plot_correlations(df_tidy.copy(), "All Datasets Combined", 'correlation_comparison')

def calculate_mean_svd_matrix(pivot_df, pipeline_order):
    svd_matrix = pd.DataFrame(index=pipeline_order, columns=pipeline_order)
    for pipe_A in pipeline_order:
        V_A = pivot_df[pipe_A]
        for pipe_B in pipeline_order:
            if pipe_A == pipe_B:
                svd_matrix.loc[pipe_A, pipe_B] = 0.0
            else:
                V_B = pivot_df[pipe_B]
                denominator = V_A + V_B
                svd_values = 2 * np.abs(V_A - V_B) / denominator
                finite_svd_values = svd_values[np.isfinite(svd_values)].dropna()
                if len(finite_svd_values) > 0:
                    svd_matrix.loc[pipe_A, pipe_B] = finite_svd_values.mean()
                else:
                    svd_matrix.loc[pipe_A, pipe_B] = np.nan
    return svd_matrix.astype(float)

def create_svd_figures(df_tidy, output_dir):
    """Create inter-pipeline mean SVD matrix plots."""

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12

    pipeline_mapping = {
        'fslanat6071ants243': 'FSL6071',
        'freesurfer741ants243': 'FS741',
        'freesurfer8001ants243': 'FS8001',
        'samseg8001ants243': 'SAMSEG8'
    }
    
    vmin, vmax = 0.0, 0.5
    cmap = sns.color_palette("magma_r", as_cmap=True) 
    label_offset = 0.03

    def plot_svd(data_subset, title_suffix, filename):
        structures = get_sorted_structures(data_subset['structure'].unique())
        if len(structures) == 0: return

        n_points = count_unique_scans(data_subset)
        data_subset = data_subset.copy()
        data_subset['pipeline_short'] = data_subset['pipeline'].map(pipeline_mapping)
        
        n_cols = 5
        n_rows = max(1, (len(structures) + n_cols - 1) // n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)

        for i, structure in enumerate(structures):
            ax = axes[i // n_cols, i % n_cols]
            structure_data = data_subset[data_subset['structure'] == structure].copy()
            
            if 'dataset' in data_subset.columns and data_subset['dataset'].nunique() > 1:
                     structure_data['scan_id'] = structure_data['dataset'] + '_' + structure_data['subject'] + '_' + structure_data['session']
            else:
                     structure_data['scan_id'] = structure_data['subject'].astype(str) + '_' + structure_data['session'].astype(str)

            pivot_df = structure_data.pivot_table(
                index='scan_id', 
                columns='pipeline_short',
                values='volume_mm3'
            )

            # --- SVD ORDER FIX ---
            # Using pivot_df.columns forces the order to be identical to the Correlation figures (Alphabetical)
            current_order = pivot_df.columns.tolist()

            if len(pivot_df.columns) >= 2 and len(pivot_df) > 1:
                svd_matrix = calculate_mean_svd_matrix(pivot_df, current_order)
                nan_mask = svd_matrix.isna()
                annot_matrix = svd_matrix.round(2).astype(str)
                annot_matrix[nan_mask] = 'Err'
                plot_svd_matrix = svd_matrix.clip(lower=vmin, upper=vmax) 
                plot_svd_matrix[nan_mask] = vmin 
                
                sns.heatmap(
                    plot_svd_matrix,
                    vmin=vmin, vmax=vmax,
                    cmap=cmap, annot=annot_matrix, fmt="s",
                    square=True, cbar=False, ax=ax
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                ax.set_xlabel("Pipeline B") 
                ax.set_ylabel("Pipeline A") 
                # Simplified Title (removed n)
                ax.set_title(f"{structure}")
            else:
                ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{structure}\n(no data)")

        total_plots = n_rows * n_cols
        for k in range(len(structures), total_plots):
            fig.delaxes(axes[k // n_cols, k % n_cols])

        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Mean SVD (Fractional Difference)", rotation=270, labelpad=15)
        cbar.ax.text(1.5, vmax + label_offset, f'High Difference (≥ {vmax:.1f})', ha='center', va='bottom', fontsize=10) 
        cbar.ax.text(1.5, vmin - label_offset, f'Low Difference (0.0)', ha='center', va='top', fontsize=10)

        fig.suptitle(f'Inter-Pipeline Mean Symmetric Volume Difference (SVD) - {title_suffix}', fontsize=18)
        
        # N Summary at bottom
        fig.text(0.5, 0.02, f"N = {n_points} Scans", ha='center', fontsize=14, fontweight='bold')

        # Adjusted margin for bottom text
        fig.tight_layout(rect=[0, 0.05, 0.9, 0.95])

        output_path = output_dir / filename
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved: {output_path}")

    for dataset in df_tidy['dataset'].unique():
        dataset_data = df_tidy[df_tidy['dataset'] == dataset].copy()
        print(f"Creating mean SVD matrices for {dataset}...")
        plot_svd(dataset_data, dataset, f'svd_comparison_mean_{dataset}.png')

    print("Creating combined mean SVD matrices...")
    plot_svd(df_tidy.copy(), "All Datasets Combined", f'svd_comparison_mean_ALL_DATASETS.png')

if __name__ == "__main__":
    EXPERIMENT_STATE_ROOT = Path(__file__).resolve().parents[1] / "experiment_state" / "figures"
    tidy_path = EXPERIMENT_STATE_ROOT / "df_tidy.csv"
    
    print(f"Looking for data file: {tidy_path}")
    
    if tidy_path.exists():
        df_tidy = pd.read_csv(tidy_path)
        print(f"✓ Loaded df_tidy.csv")
        
        print(f"  Sanitizing data (Removing rows with volume <= 0 or NaN)...")
        df_tidy['volume_mm3'] = pd.to_numeric(df_tidy['volume_mm3'], errors='coerce')
        initial_len = len(df_tidy)
        df_tidy = df_tidy.dropna(subset=['volume_mm3'])
        df_tidy = df_tidy[df_tidy['volume_mm3'] > 0]
        print(f"  Dropped {initial_len - len(df_tidy)} invalid rows.")
        
        EXPERIMENT_STATE_ROOT.mkdir(parents=True, exist_ok=True)
        
        # New Strict Filtering (Session-Level)
        df_complete = filter_complete_pipelines(df_tidy)
        
        create_distribution_figures(df_complete, EXPERIMENT_STATE_ROOT)
        print("✓ Distribution plots generated successfully!")
        create_correlation_figures(df_complete, EXPERIMENT_STATE_ROOT)
        print("✓ Correlation plots (Pearson & Spearman) generated successfully!")
        create_svd_figures(df_complete, EXPERIMENT_STATE_ROOT)
        print("✓ Mean SVD plots generated successfully!")
        
    else:
        print(f"✗ File not found: {tidy_path}")
        print("Please run the data extraction script first to generate df_tidy.csv")
