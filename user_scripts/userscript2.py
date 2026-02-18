import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np 

def filter_complete_pipelines(df_tidy):
    """
    Return subset of df_tidy using STRICT Session-Level Complete Case Analysis.
    
    Logic:
    1. Check every structure for every session.
    2. If ANY structure in a session has fewer than 4 pipelines (missing/zero/nan),
       DROP THE ENTIRE SESSION (all structures for that subject/session).
    3. This ensures that the N reported is the number of fully successful sessions.
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
        # We merge with indicator=True, then keep only rows that DID NOT match the bad list.
        merged = df_tidy.merge(bad_sessions, 
                               on=['dataset', 'subject', 'session'], 
                               how='left', 
                               indicator=True)
        
        df_filtered = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        print("No incomplete sessions found.")
        df_filtered = df_tidy.copy()

    # 4. Final Sanity Check: Ensure remaining rows are strictly complete (should be guaranteed by above)
    # We can do a quick check to see if any stragglers remain (unlikely logic-wise, but good for safety)
    final_counts = (
        df_filtered.groupby(['dataset', 'subject', 'session', 'structure'])['pipeline']
        .nunique()
        .reset_index(name='n_pipelines')
    )
    assert final_counts['n_pipelines'].min() == len(required_pipelines), "Logic Error: Incomplete pipelines remain!"

    print(f"Filtered from {len(df_tidy)} -> {len(df_filtered)} rows.")
    print(f"Dropped {len(df_tidy) - len(df_filtered)} rows belonging to incomplete sessions.")
    
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
    """Create histograms with no vertical gaps, 2-column pairing, and centered Brainstem."""
    import matplotlib.gridspec as gridspec
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

    # 1. Organize structures into L/R pairs
    all_structures = get_sorted_structures(df_tidy['structure'].unique())
    paired_rows = []
    i = 0
    while i < len(all_structures):
        s1 = all_structures[i]
        if i + 1 < len(all_structures):
            s2 = all_structures[i+1]
            if s1.replace('Left-', '') == s2.replace('Right-', ''):
                paired_rows.append([s1, s2])
                i += 2
            else:
                paired_rows.append([s1])
                i += 1
        else:
            paired_rows.append([s1])
            i += 1

    datasets = list(df_tidy['dataset'].unique()) + ['ALL_DATASETS']
    
    for dataset in datasets:
        is_combined = (dataset == 'ALL_DATASETS')
        plot_data = df_tidy.copy() if is_combined else df_tidy[df_tidy['dataset'] == dataset].copy()
        plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)
        
        n_rows = len(paired_rows)
        # Tighten height to 3.0 per row to pull rows together
        fig = plt.figure(figsize=(10, 3.0 * n_rows))
        
        # 4-column GridSpec for flexible centering
        gs = gridspec.GridSpec(n_rows, 4, figure=fig)
        
        n_points = 0 
        for row_idx, pair in enumerate(paired_rows):
            if len(pair) == 2:
                for col_idx, struct in enumerate(pair):
                    col_span = slice(0, 2) if col_idx == 0 else slice(2, 4)
                    ax = fig.add_subplot(gs[row_idx, col_span])
                    structure_data = plot_data[plot_data['structure'] == struct]
                    if not structure_data.empty:
                        sns.histplot(data=structure_data, x='volume_mm3', hue='pipeline_short', 
                                     hue_order=pipeline_order, binwidth=50, common_bins=True, 
                                     common_norm=False, element='step', fill=True, alpha=0.4, 
                                     palette=palette, ax=ax, stat='count')
                        ax.set_title(f"{struct}")
                        ax.set_xlabel('Volume (mm³)')
                        ax.set_ylabel('Count')
                        n_points = count_unique_scans(structure_data)
            else:
                # Center Brainstem (single structure) in middle 2 of 4 columns
                ax = fig.add_subplot(gs[row_idx, 1:3])
                struct = pair[0]
                structure_data = plot_data[plot_data['structure'] == struct]
                if not structure_data.empty:
                    sns.histplot(data=structure_data, x='volume_mm3', hue='pipeline_short', 
                                 hue_order=pipeline_order, binwidth=50, common_bins=True, 
                                 common_norm=False, element='step', fill=True, alpha=0.4, 
                                 palette=palette, ax=ax, stat='count')
                    ax.set_title(f"{struct}")
                    ax.set_xlabel('Volume (mm³)')
                    ax.set_ylabel('Count')
                    n_points = count_unique_scans(structure_data)

        # Restore original exact titles
        title_str = 'Pipeline Distributions (Counts) - All Datasets Combined' if is_combined else f'Pipeline Distributions (Counts) - {dataset}'
        fig.suptitle(title_str, fontsize=18, y=0.97)
        
        # Explicit footer with N and bin width
        fig.text(0.5, 0.02, f"N = {n_points} Scans | Bin Width = 50mm³", ha='center', fontsize=14, fontweight='bold')
        
        # Manual adjustment to ELIMINATE gaps:
        # top=0.94 pulls plots up to the title; bottom=0.06 pulls plots down to footer
        fig.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.92, hspace=0.4, wspace=0.5)
        
        suffix = "_ALL_DATASETS" if is_combined else f"_{dataset}"
        # Save with minimum padding
        plt.savefig(output_dir / f'distribution_comparison_counts{suffix}.png', bbox_inches='tight', pad_inches=0.05, dpi=300)
        plt.close()

def create_correlation_figures(df_tidy, output_dir):
    """Create inter-pipeline Pearson and Spearman correlation matrix plots."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12

    pipeline_mapping = {'fslanat6071ants243': 'FSL6071', 'freesurfer741ants243': 'FS741', 
                        'freesurfer8001ants243': 'FS8001', 'samseg8001ants243': 'SAMSEG8'}
    pipeline_order = list(pipeline_mapping.values())
    vmin, vmax, center = -1, 1, 0
    cmap = sns.color_palette("RdBu_r", as_cmap=True)

    layouts = [{'cols': 5, 'suffix': ''}, {'cols': 3, 'suffix': '_paper'}]

    for layout in layouts:
        n_cols = layout['cols']
        suffix = layout['suffix']

        for dataset in list(df_tidy['dataset'].unique()) + ['ALL_DATASETS']:
            is_combined = (dataset == 'ALL_DATASETS')
            dataset_data = df_tidy if is_combined else df_tidy[df_tidy['dataset'] == dataset]
            
            plot_data = dataset_data.copy()
            plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)
            structures = get_sorted_structures(plot_data['structure'].unique())
            if not structures: continue
                
            n_rows = (len(structures) + n_cols - 1) // n_cols
            
            for corr_method in ['pearson', 'spearman']:
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)
                n_points = 0
                for i, structure in enumerate(structures):
                    ax = axes[i // n_cols, i % n_cols]
                    structure_data = plot_data[plot_data['structure'] == structure].copy()
                    id_cols = ['dataset', 'subject', 'session'] if is_combined else ['subject', 'session']
                    structure_data['scan_id'] = structure_data[id_cols].astype(str).agg('_'.join, axis=1)

                    pivot_df = structure_data.pivot_table(index='scan_id', columns='pipeline_short', values='volume_mm3')
                    if len(pivot_df.columns) == len(pipeline_order) and len(pivot_df) > 1:
                        corr = pivot_df.corr(method=corr_method)
                        sns.heatmap(corr, vmin=vmin, vmax=vmax, center=center, cmap=cmap, annot=True, fmt=".2f", square=True, cbar=False, ax=ax)
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                        n_points = pivot_df.shape[0]
                        ax.set_title(f"{structure}")
                    else:
                        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f"{structure}\n(no data)")

                for k in range(len(structures), n_rows * n_cols):
                    fig.delaxes(axes[k // n_cols, k % n_cols])

                cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
                fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=cbar_ax).set_label(f"{corr_method.capitalize()} correlation", rotation=270, labelpad=15)
                fig.suptitle(f'Inter-Pipeline Correlations ({corr_method.capitalize()}) - {dataset}', fontsize=18)
                fig.text(0.5, 0.02, f"N = {n_points} Scans", ha='center', fontsize=14, fontweight='bold')
                fig.tight_layout(rect=[0, 0.05, 0.9, 0.95])
                plt.savefig(output_dir / f'correlation_comparison_{corr_method}_{dataset}{suffix}.png', bbox_inches='tight', dpi=300)
                plt.close()

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

    pipeline_mapping = {'fslanat6071ants243': 'FSL6071', 'freesurfer741ants243': 'FS741', 
                        'freesurfer8001ants243': 'FS8001', 'samseg8001ants243': 'SAMSEG8'}
    pipeline_order = list(pipeline_mapping.values())
    vmin, vmax = 0.0, 0.5
    cmap = sns.color_palette("magma_r", as_cmap=True)

    layouts = [{'cols': 5, 'suffix': ''}, {'cols': 3, 'suffix': '_paper'}]

    for layout in layouts:
        n_cols = layout['cols']
        suffix = layout['suffix']

        for dataset in list(df_tidy['dataset'].unique()) + ['ALL_DATASETS']:
            is_combined = (dataset == 'ALL_DATASETS')
            dataset_data = df_tidy if is_combined else df_tidy[df_tidy['dataset'] == dataset]
            
            plot_data = dataset_data.copy()
            plot_data['pipeline_short'] = plot_data['pipeline'].map(pipeline_mapping)
            structures = get_sorted_structures(plot_data['structure'].unique())
            if not structures: continue
                
            n_rows = (len(structures) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)
            
            n_points = 0
            for i, structure in enumerate(structures):
                ax = axes[i // n_cols, i % n_cols]
                structure_data = plot_data[plot_data['structure'] == structure].copy()
                id_cols = ['dataset', 'subject', 'session'] if is_combined else ['subject', 'session']
                structure_data['scan_id'] = structure_data[id_cols].astype(str).agg('_'.join, axis=1)

                pivot_df = structure_data.pivot_table(index='scan_id', columns='pipeline_short', values='volume_mm3')
                current_pipeline_order = pivot_df.columns.tolist()

                if len(pivot_df.columns) == len(pipeline_order) and len(pivot_df) > 1:
                    svd_matrix = calculate_mean_svd_matrix(pivot_df, current_pipeline_order)
                    sns.heatmap(svd_matrix.clip(lower=vmin, upper=vmax), vmin=vmin, vmax=vmax, cmap=cmap, 
                                annot=svd_matrix.round(2).astype(str).mask(svd_matrix.isna(), 'Err'), 
                                fmt="s", square=True, cbar=False, ax=ax)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                    n_points = pivot_df.shape[0]
                    ax.set_title(f"{structure}")
                else:
                    ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{structure}\n(no data)")

            for k in range(len(structures), n_rows * n_cols):
                fig.delaxes(axes[k // n_cols, k % n_cols])

            cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
            fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)), cax=cbar_ax).set_label("Mean SVD", rotation=270, labelpad=15)
            fig.suptitle(f'Inter-Pipeline Mean SVD - {dataset}', fontsize=18)
            fig.text(0.5, 0.02, f"N = {n_points} Scans", ha='center', fontsize=14, fontweight='bold')
            fig.tight_layout(rect=[0, 0.05, 0.9, 0.95])
            plt.savefig(output_dir / f'svd_comparison_mean_{dataset}{suffix}.png', bbox_inches='tight', dpi=300)
            plt.close()

if __name__ == "__main__":
    EXPERIMENT_STATE_ROOT = Path(__file__).resolve().parents[1] / "experiment_state" / "figures"
    tidy_path = EXPERIMENT_STATE_ROOT / "df_tidy.csv"
    
    print(f"Looking for data file: {tidy_path}")
    
    if tidy_path.exists():
        df_tidy = pd.read_csv(tidy_path)
        print(f"✓ Loaded df_tidy.csv")
        print(f"  Shape: {df_tidy.shape}")
        
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
