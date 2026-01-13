import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from scipy.stats import spearmanr, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration (kept as is)
# -----------------------------------------------------------------------------
EXPERIMENT_STATE_ROOT = Path(__file__).resolve().parents[1] / "experiment_state" / "figures"
EXPERIMENT_STATE_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Helper functions (kept as is)
# -----------------------------------------------------------------------------
def load_tabular_data(dataset_path: Path):
    """Load all TSV files inside a dataset's 'tabular' directory into a dictionary."""
    tabular_path = dataset_path / "tabular"
    dfs = {}
    for tsv in tabular_path.glob("*.tsv"):
        df = pd.read_csv(tsv, sep='\t')
        dfs[tsv.stem] = df
    return dfs

def extract_demographics(dataset_name: str, dfs: dict):
    """Dataset-specific logic to extract age and sex information."""
    df_demo = None
    dataset_lower = dataset_name.lower()

    # PREVENT-AD logic (UPDATED: Dual Age Sources + Sex)
    if dataset_lower == 'preventad':
        # --- 1. Extract Sex (Subject-Level) ---
        df_sex = pd.DataFrame(columns=['subject', 'sex'])
        if 'demographics' in dfs:
            d = dfs['demographics'].copy()
            if 'participant_id' in d.columns and 'Sex' in d.columns:
                # Normalize ID
                d['participant_id'] = d['participant_id'].astype(str).str.strip().apply(
                    lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                )
                df_sex = d[['participant_id', 'Sex']].rename(columns={'participant_id': 'subject', 'Sex': 'sex'})

        # --- 2. Extract Age (Session-Level) ---
        age_dfs = []

        # Source A: mci_status.tsv
        if 'mci_status' in dfs:
            mci = dfs['mci_status'].copy()
            
            # Normalize ID
            if 'participant_id' in mci.columns:
                mci['subject'] = mci['participant_id'].astype(str).str.strip().apply(
                    lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                )
            
            # Calculate Age
            age_col = None
            if 'Candidate_Age' in mci.columns: age_col = 'Candidate_Age'
            elif 'Age' in mci.columns: age_col = 'Age'
            elif 'age_months' in mci.columns: age_col = 'age_months'
            elif len(mci.columns) > 3 and mci.iloc[:, 3].dtype.kind in 'fi':
                 if 'Unnamed: 3' in mci.columns: age_col = 'Unnamed: 3'
            
            if age_col and 'subject' in mci.columns and 'visit_id' in mci.columns:
                mci['age'] = pd.to_numeric(mci[age_col], errors='coerce') / 12.0
                
                # Generate Session Variations
                v1 = mci[['subject', 'visit_id', 'age']].copy()
                v1['session'] = v1['visit_id'].astype(str).str.strip().apply(
                    lambda x: f"ses-{x}" if not x.startswith('ses-') else x
                )
                
                v2 = mci[['subject', 'visit_id', 'age']].copy()
                v2['session'] = v2['visit_id'].astype(str).str.strip().apply(
                    lambda x: x.replace('PREFU', 'FU')
                ).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                
                v3 = mci[['subject', 'visit_id', 'age']].copy()
                v3['session'] = v3['visit_id'].astype(str).str.strip().apply(
                    lambda x: x.replace('NAPFU', 'FU')
                ).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)

                age_dfs.extend([v1[['subject', 'session', 'age']], 
                                v2[['subject', 'session', 'age']], 
                                v3[['subject', 'session', 'age']]])

        # Source B: mri_sessions-phase1.tsv
        if 'mri_sessions-phase1' in dfs:
            mri = dfs['mri_sessions-phase1'].copy()
            
            # Normalize ID
            if 'participant_id' in mri.columns:
                mri['subject'] = mri['participant_id'].astype(str).str.strip().apply(
                    lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                )
            
            # Calculate Age
            if 'age' in mri.columns and 'subject' in mri.columns and 'session_id' in mri.columns:
                mri['age_years'] = pd.to_numeric(mri['age'], errors='coerce') / 12.0
                
                v1 = mri[['subject', 'session_id', 'age_years']].copy()
                v1['session'] = v1['session_id'].astype(str).str.strip().apply(
                    lambda x: f"ses-{x}" if not x.startswith('ses-') else x
                )
                
                v2 = mri[['subject', 'session_id', 'age_years']].copy()
                v2['session'] = v2['session_id'].astype(str).str.strip().apply(
                    lambda x: x.replace('PREFU', 'FU')
                ).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                
                v3 = mri[['subject', 'session_id', 'age_years']].copy()
                v3['session'] = v3['session_id'].astype(str).str.strip().apply(
                    lambda x: x.replace('NAPFU', 'FU')
                ).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                
                age_dfs.extend([v1[['subject', 'session', 'age_years']].rename(columns={'age_years':'age'}), 
                                v2[['subject', 'session', 'age_years']].rename(columns={'age_years':'age'}), 
                                v3[['subject', 'session', 'age_years']].rename(columns={'age_years':'age'})])

        # --- 3. Combine & Finalize ---
        if age_dfs:
            df_combined_ages = pd.concat(age_dfs, ignore_index=True)
            df_combined_ages = df_combined_ages.drop_duplicates(subset=['subject', 'session'], keep='first')
            
            if not df_sex.empty:
                df_demo = df_combined_ages.merge(df_sex, on='subject', how='left')
            else:
                df_demo = df_combined_ages
                df_demo['sex'] = np.nan
        else:
            df_demo = pd.DataFrame(columns=['subject', 'session', 'age', 'sex'])

        for col in ['subject', 'session', 'age', 'sex']:
            if col not in df_demo.columns:
                df_demo[col] = np.nan
        df_demo = df_demo[['subject', 'session', 'age', 'sex']]

    elif dataset_lower == 'ds005752':
        df = dfs.get('participants', pd.DataFrame())
        if not df.empty:
            df_demo = df[['participant_id', 'age', 'sex']].rename(columns={'participant_id': 'subject'})
    elif dataset_lower == 'ds003592':
        df = dfs.get('participants', pd.DataFrame())
        if not df.empty:
            df_demo = df[['participant_id', 'age', 'sex']].rename(columns={'participant_id': 'subject'})
    
    elif 'rockland' in dataset_lower or 'nki' in dataset_lower:
        key = next((k for k in dfs.keys() if 'participants' in k), None)
        if key:
            df = dfs[key]
            if 'participant_id' in df.columns:
                df_demo = df.rename(columns={'participant_id': 'subject'})
                if 'subject' in df_demo.columns:
                    df_demo['subject'] = df_demo['subject'].astype(str).apply(
                        lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                    )

                if 'session_id' in df_demo.columns:
                    df_demo = df_demo.rename(columns={'session_id': 'session'})
                    df_demo['session'] = df_demo['session'].astype(str).apply(
                        lambda x: f"ses-{x}" if not x.startswith('ses-') else x
                    )
                
                cols = ['subject']
                if 'session' in df_demo.columns: cols.append('session')
                if 'age' in df_demo.columns: cols.append('age')
                if 'sex' in df_demo.columns: cols.append('sex')
                df_demo = df_demo[cols]

    else:
        if 'participants' in dfs:
            df_demo = dfs['participants'].rename(columns={'participant_id': 'subject'})
            if 'age' not in df_demo.columns:
                df_demo['age'] = np.nan
            if 'sex' not in df_demo.columns:
                df_demo['sex'] = np.nan
    
    if df_demo is not None and not df_demo.empty:
        if 'age' in df_demo.columns:
            df_demo['age'] = df_demo['age'].astype(str).str.split(',').str[0]
            df_demo['age'] = df_demo['age'].replace('n/a', np.nan)
            df_demo['age'] = pd.to_numeric(df_demo['age'], errors='coerce')
        
        if 'sex' in df_demo.columns:
            df_demo['sex'] = df_demo['sex'].astype(str).str.split(',').str[0]
            df_demo['sex'] = df_demo['sex'].replace('n/a', np.nan)
            non_nan_mask = df_demo['sex'].notna()
            df_demo.loc[non_nan_mask, 'sex'] = (
                df_demo.loc[non_nan_mask, 'sex'].astype(str).str.upper()
            )
            standardization_map = {'MALE': 'M', 'FEMALE': 'F'}
            df_demo.loc[non_nan_mask, 'sex'] = (
                df_demo.loc[non_nan_mask, 'sex'].replace(standardization_map)
            )
    
    if df_demo is not None and not df_demo.empty:
        if 'session' not in df_demo.columns:
            df_demo['session'] = np.nan
        return df_demo[['subject', 'session', 'age', 'sex']]
    else:
        return pd.DataFrame(columns=['subject', 'session', 'age', 'sex'])

def shorten_pipeline_name(pipeline_name):
    """Shorten pipeline names for display in figures."""
    mapping = {
        'freesurfer741ants243': 'FS741',
        'samseg8001ants243': 'Samseg8', 
        'freesurfer8001ants243': 'FS8001',
        'fslanat6071ants243': 'FSL6071'
    }
    return mapping.get(pipeline_name, pipeline_name)

def get_structure_order():
    """Define the desired order of structures for heatmaps."""
    base_structures = [
        'Thalamus', 'Caudate', 'Putamen', 'Pallidum', 
        'Hippocampus', 'Amygdala', 'Accumbens-area'
    ]
    ordered_structures = []
    for struct in base_structures:
        ordered_structures.extend([f'Left-{struct}', f'Right-{struct}'])
    ordered_structures.append('Brainstem')
    return ordered_structures

def create_age_distribution_plot(df, output_dir):
    """Create a single histogram showing age distributions for all datasets overlaid."""
    subject_ages = df[['dataset', 'subject', 'session', 'age']].drop_duplicates().dropna(subset=['age'])
    
    if subject_ages.empty:
        print("No age data available for plotting.")
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12
    
    plt.figure(figsize=(12, 8))
    datasets = subject_ages['dataset'].unique()
    palette = sns.color_palette("tab10", len(datasets))
    
    for i, dataset in enumerate(datasets):
        dataset_ages = subject_ages[subject_ages['dataset'] == dataset]['age']
        n_scans = len(dataset_ages)
        
        sns.histplot(
            data=dataset_ages,
            bins=20,
            element='step',
            fill=True,
            alpha=0.4,
            color=palette[i],
            label=f'{dataset} (n={n_scans})',
            stat='count'
        )
    
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.title('Age Distribution by Dataset (Unique Scans)')
    plt.legend()
    plt.tight_layout()
    output_path = output_dir / 'age_distribution_by_dataset.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved age distribution plot: {output_path}")

# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def main():
    print("Loading data...")
    df_tidy = pd.read_csv(Path('/home/runner/work/NeuroCI/NeuroCI/experiment_state/figures/df_tidy.csv'))
    df_all = []
    root = Path('/tmp/neuroci_output_state')

    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dfs = load_tabular_data(dataset_dir)
        df_demo = extract_demographics(dataset_dir.name, dfs)
        if df_demo is None or df_demo.empty:
            continue
        
        df_dataset = df_tidy[df_tidy['dataset'] == dataset_dir.name].copy()
        
        # ---------------------------------------------------------------------
        # CRITICAL FIX: Robust Type Enforcement Before Merge
        # ---------------------------------------------------------------------
        # 1. Normalize Subject to String
        df_dataset['subject'] = df_dataset['subject'].astype(str).str.strip()
        df_demo['subject'] = df_demo['subject'].astype(str).str.strip()
        
        # 2. Determine Merge Keys
        merge_on = ['subject']
        if 'session' in df_demo.columns and 'session' in df_dataset.columns:
            # Only use session if it's NOT entirely NaN in df_dataset
            if not df_dataset['session'].isna().all():
                merge_on = ['subject', 'session']
                
                # Force Session to String (handling NaNs as 'nan' temporarily or preserving NaNs)
                # Best practice: Fill NaNs with a placeholder, convert to string, then revert NaNs if needed
                # BUT since we are merging, we want 'ses-01' to match 'ses-01'.
                
                # Convert dataset sessions to string, keeping NaNs as NaN
                df_dataset['session'] = df_dataset['session'].astype(str)
                df_dataset.loc[df_dataset['session'] == 'nan', 'session'] = np.nan
                
                # Convert demo sessions to string
                df_demo['session'] = df_demo['session'].astype(str)
                df_demo.loc[df_demo['session'] == 'nan', 'session'] = np.nan
        
        # 3. Print Pre-Merge Diagnostics
        print(f"[{dataset_dir.name}] Pre-merge Dataset rows: {len(df_dataset)}, Demo rows: {len(df_demo)}")
        
        # 4. Merge
        df_dataset = df_dataset.merge(df_demo, how='left', on=merge_on)
        print(f"[{dataset_dir.name}] Post-merge rows: {len(df_dataset)} (Age populated: {df_dataset['age'].notna().sum()})")
        
        df_all.append(df_dataset)

    if not df_all:
        print("No datasets with usable demographic data found.")
        return

    df = pd.concat(df_all, ignore_index=True)

    # -------------------------------------------------------------------------
    # GLOBAL FILTER: GRAND INTERSECTION (STRICT)
    # -------------------------------------------------------------------------
    # Drop any scan that is missing ANY of the 15 structures in ANY pipeline.
    # -------------------------------------------------------------------------
    print("Applying Grand Intersection Filter...")
    required_structures = get_structure_order()
    unique_pipelines = df['pipeline'].unique()
    
    # 1. Filter to relevant structures only
    df = df[df['structure'].isin(required_structures)].copy()
    
    # 2. Ensure volume data is present
    if 'volume_mm3' in df.columns:
        df = df.dropna(subset=['volume_mm3'])
        
    # 3. Identify scans (dataset, subject, session) that have complete data
    # Group including Session to handle longitudinal data correctly
    completeness = df.groupby(['dataset', 'subject', 'session']).size()
    expected_rows = len(unique_pipelines) * len(required_structures)
    
    valid_scans = completeness[completeness == expected_rows].index
    
    # 4. Apply filter globally
    df = df.set_index(['dataset', 'subject', 'session'])
    df = df.loc[valid_scans].reset_index()
    
    print(f"Global Filter: Retained {len(valid_scans)} scans common to all {len(unique_pipelines)} pipelines.")

    # Create age distribution plot
    create_age_distribution_plot(df, EXPERIMENT_STATE_ROOT)

    # -------------------------------------------------------------------------
    # Compute pairwise pipeline differences (absolute values)
    # -------------------------------------------------------------------------
    pairwise_diffs = []
    pipelines = df['pipeline'].unique()

    for (pipe1, pipe2) in itertools.combinations(pipelines, 2):
        df1 = df[df['pipeline'] == pipe1]
        df2 = df[df['pipeline'] == pipe2]
        merged = df1.merge(df2, on=['dataset', 'subject', 'session', 'structure'], suffixes=(f'_{pipe1}', f'_{pipe2}'))

        # Compute absolute volume difference
        merged['volume_diff'] = (merged['volume_mm3_' + pipe1] - merged['volume_mm3_' + pipe2]).abs()
        
        short_pipe1 = shorten_pipeline_name(pipe1)
        short_pipe2 = shorten_pipeline_name(pipe2)
        merged['pipeline_pair'] = f"{short_pipe1}_vs_{short_pipe2}"
        
        merged['age'] = merged['age_' + pipe1].combine_first(merged['age_' + pipe2])
        merged['sex'] = merged['sex_' + pipe1].combine_first(merged['sex_' + pipe2])

        pairwise_diffs.append(merged[['dataset', 'subject', 'session', 'structure', 'pipeline_pair', 'volume_diff', 'age', 'sex']])

    df_diff = pd.concat(pairwise_diffs, ignore_index=True)

    # -------------------------------------------------------------------------
    # Correlation with age (Spearman)
    # -------------------------------------------------------------------------
    df_age = df_diff[['dataset','subject','session','pipeline_pair','structure','volume_diff','age']].copy()
    df_age['volume_diff'] = pd.to_numeric(df_age['volume_diff'], errors='coerce')
    df_age['age'] = pd.to_numeric(df_age['age'], errors='coerce')
    
    corr_results = []
    for (pair, struct), g in df_age.groupby(['pipeline_pair','structure']):
        # Note: Scans are already pre-filtered for completeness.
        g = g.dropna(subset=['volume_diff','age'])
        n = g.groupby(['dataset','subject','session']).ngroups  # unique scans
        if n < 3:
            continue
        
        try:
            r, p = spearmanr(g['volume_diff'], g['age'])
        except ValueError:
            r, p = np.nan, np.nan
        
        corr_results.append({'pipeline_pair': pair, 'structure': struct, 'r': r, 'p': p, 'n': n})
    
    corr_df = pd.DataFrame(corr_results).dropna(subset=['r', 'p'])
    if not corr_df.empty:
        corr_df['p_adj'] = np.minimum(corr_df['p'] * len(corr_df), 1.0)
        corr_df.to_csv(EXPERIMENT_STATE_ROOT / 'age_correlation_summary_spearman.csv', index=False)
        
        structure_order = get_structure_order()
        corr_df_ordered = corr_df[corr_df['structure'].isin(structure_order)].copy()
        corr_df_ordered['structure'] = pd.Categorical(
            corr_df_ordered['structure'], 
            categories=structure_order, 
            ordered=True
        )
        corr_df_ordered = corr_df_ordered.sort_values('structure')
        
        corr_pivot = corr_df_ordered.pivot(index='structure', columns='pipeline_pair', values='r')
        p_pivot = corr_df_ordered.pivot(index='structure', columns='pipeline_pair', values='p_adj')
        
        corr_rounded = corr_pivot.round(2)
        annot_matrix = corr_rounded.astype(str)
        annot_matrix[p_pivot >= 0.05] = ''
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_pivot, annot=annot_matrix, fmt='', cmap='coolwarm', center=0,
                    cbar_kws={'label': 'Spearman r'})
        plt.title('Spearman Correlation of Volume Differences with Age')
        plt.tight_layout()
        plt.savefig(EXPERIMENT_STATE_ROOT / 'corr_age_spearman_heatmap.png', dpi=300)
        plt.close()


    # -------------------------------------------------------------------------
    # Sex effects
    # -------------------------------------------------------------------------
    df_sex = df_diff[['dataset','subject','session','pipeline_pair','structure','volume_diff','sex']].copy()
    df_sex['volume_diff'] = pd.to_numeric(df_sex['volume_diff'], errors='coerce')

    sex_results = []
    for (pair, struct), g in df_sex.groupby(['pipeline_pair','structure']):
        g = g.dropna(subset=['volume_diff','sex'])
        n = g.groupby(['dataset','subject','session']).ngroups
        males = g[g['sex'].str.lower().str.startswith('m')]['volume_diff']
        females = g[g['sex'].str.lower().str.startswith('f')]['volume_diff']
        if len(males) < 2 or len(females) < 2:
            continue
        t, p = ttest_ind(males, females, equal_var=False)
        cohen_d = (males.mean() - females.mean()) / np.sqrt(((males.std() ** 2 + females.std() ** 2) / 2))
        sex_results.append({'pipeline_pair': pair, 'structure': struct, 't': t, 'p': p, 'cohen_d': cohen_d, 'n': n})

    sex_df = pd.DataFrame(sex_results).dropna(subset=['t', 'p'])
    if not sex_df.empty:
        sex_df['p_adj'] = np.minimum(sex_df['p'] * len(sex_df), 1.0)
        sex_df.to_csv(EXPERIMENT_STATE_ROOT / 'sex_effects_summary.csv', index=False)

        structure_order = get_structure_order()
        sex_df_ordered = sex_df[sex_df['structure'].isin(structure_order)].copy()
        sex_df_ordered['structure'] = pd.Categorical(
            sex_df_ordered['structure'], 
            categories=structure_order, 
            ordered=True
        )
        sex_df_ordered = sex_df_ordered.sort_values('structure')
        
        sex_pivot = sex_df_ordered.pivot(index='structure', columns='pipeline_pair', values='cohen_d')
        p_pivot_sex = sex_df_ordered.pivot(index='structure', columns='pipeline_pair', values='p_adj')
        
        sex_rounded = sex_pivot.round(2)
        annot_matrix_sex = sex_rounded.astype(str)
        annot_matrix_sex[p_pivot_sex >= 0.05] = ''
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sex_pivot, annot=annot_matrix_sex, fmt='', cmap='vlag', center=0,
                    cbar_kws={'label': "Cohen's d"})
        plt.title("Sex Effect (Cohen's d) on Volume Differences")
        plt.tight_layout()
        plt.savefig(EXPERIMENT_STATE_ROOT / 'sex_effects_heatmap_significant.png', dpi=300)
        plt.close()

if __name__ == '__main__':
    main()
