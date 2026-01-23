import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from scipy.stats import spearmanr, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EXPERIMENT_STATE_ROOT = Path(__file__).resolve().parents[1] / "experiment_state" / "figures"
EXPERIMENT_STATE_ROOT.mkdir(parents=True, exist_ok=True)
ROOT = Path('/tmp/neuroci_output_state') 

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_tabular_data(dataset_path: Path):
    """Load all TSV files inside a dataset's 'tabular' directory."""
    tabular_path = dataset_path / "tabular"
    dfs = {}
    if tabular_path.exists():
        for tsv in tabular_path.glob("*.tsv"):
            try:
                df = pd.read_csv(tsv, sep='\t')
                dfs[tsv.stem] = df
            except Exception:
                pass
    return dfs

def extract_demographics(dataset_name: str, dfs: dict):
    """Dataset-specific logic to extract age and sex information."""
    df_demo = None
    dataset_lower = dataset_name.lower()

    # PREVENT-AD logic
    if dataset_lower == 'preventad':
        # --- 1. Extract Sex (Subject-Level) ---
        df_sex = pd.DataFrame(columns=['subject', 'sex'])
        if 'demographics' in dfs:
            d = dfs['demographics'].copy()
            if 'participant_id' in d.columns and 'Sex' in d.columns:
                d['participant_id'] = d['participant_id'].astype(str).str.strip().apply(
                    lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                )
                df_sex = d[['participant_id', 'Sex']].rename(columns={'participant_id': 'subject', 'Sex': 'sex'})

        # --- 2. Extract Age (Session-Level) ---
        age_dfs = []

        # Source A: mci_status.tsv
        if 'mci_status' in dfs:
            mci = dfs['mci_status'].copy()
            if 'participant_id' in mci.columns:
                mci['subject'] = mci['participant_id'].astype(str).str.strip().apply(
                    lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                )
            
            age_col = None
            if 'Candidate_Age' in mci.columns: age_col = 'Candidate_Age'
            elif 'Age' in mci.columns: age_col = 'Age'
            elif 'age_months' in mci.columns: age_col = 'age_months'
            elif len(mci.columns) > 3 and mci.iloc[:, 3].dtype.kind in 'fi':
                 if 'Unnamed: 3' in mci.columns: age_col = 'Unnamed: 3'
            
            if age_col and 'subject' in mci.columns and 'visit_id' in mci.columns:
                mci['age'] = pd.to_numeric(mci[age_col], errors='coerce') / 12.0
                
                v1 = mci[['subject', 'visit_id', 'age']].copy()
                v1['session'] = v1['visit_id'].astype(str).str.strip().apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                v2 = mci[['subject', 'visit_id', 'age']].copy()
                v2['session'] = v2['visit_id'].astype(str).str.strip().apply(lambda x: x.replace('PREFU', 'FU')).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                v3 = mci[['subject', 'visit_id', 'age']].copy()
                v3['session'] = v3['visit_id'].astype(str).str.strip().apply(lambda x: x.replace('NAPFU', 'FU')).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)

                age_dfs.extend([v1[['subject', 'session', 'age']], v2[['subject', 'session', 'age']], v3[['subject', 'session', 'age']]])

        # Source B: mri_sessions-phase1.tsv
        if 'mri_sessions-phase1' in dfs:
            mri = dfs['mri_sessions-phase1'].copy()
            if 'participant_id' in mri.columns:
                mri['subject'] = mri['participant_id'].astype(str).str.strip().apply(
                    lambda x: f"sub-{x}" if not x.startswith('sub-') else x
                )
            
            if 'age' in mri.columns and 'subject' in mri.columns and 'session_id' in mri.columns:
                mri['age_years'] = pd.to_numeric(mri['age'], errors='coerce') / 12.0
                
                v1 = mri[['subject', 'session_id', 'age_years']].copy()
                v1['session'] = v1['session_id'].astype(str).str.strip().apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                v2 = mri[['subject', 'session_id', 'age_years']].copy()
                v2['session'] = v2['session_id'].astype(str).str.strip().apply(lambda x: x.replace('PREFU', 'FU')).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                v3 = mri[['subject', 'session_id', 'age_years']].copy()
                v3['session'] = v3['session_id'].astype(str).str.strip().apply(lambda x: x.replace('NAPFU', 'FU')).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                
                age_dfs.extend([v1[['subject', 'session', 'age_years']].rename(columns={'age_years':'age'}), 
                                v2[['subject', 'session', 'age_years']].rename(columns={'age_years':'age'}), 
                                v3[['subject', 'session', 'age_years']].rename(columns={'age_years':'age'})])

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
                    df_demo['subject'] = df_demo['subject'].astype(str).apply(lambda x: f"sub-{x}" if not x.startswith('sub-') else x)
                if 'session_id' in df_demo.columns:
                    df_demo = df_demo.rename(columns={'session_id': 'session'})
                    df_demo['session'] = df_demo['session'].astype(str).apply(lambda x: f"ses-{x}" if not x.startswith('ses-') else x)
                cols = ['subject']
                if 'session' in df_demo.columns: cols.append('session')
                if 'age' in df_demo.columns: cols.append('age')
                if 'sex' in df_demo.columns: cols.append('sex')
                df_demo = df_demo[cols]
    else:
        if 'participants' in dfs:
            df_demo = dfs['participants'].rename(columns={'participant_id': 'subject'})

    # Normalize Output Columns
    if df_demo is not None and not df_demo.empty:
        for col in ['subject', 'session', 'age', 'sex']:
            if col not in df_demo.columns:
                df_demo[col] = np.nan
                
        # Value Cleanup
        if 'age' in df_demo.columns:
            df_demo['age'] = pd.to_numeric(df_demo['age'].astype(str).str.split(',').str[0].replace('n/a', np.nan), errors='coerce')
        if 'sex' in df_demo.columns:
            s = df_demo['sex'].astype(str).str.split(',').str[0].str.upper()
            s = s.replace({'MALE': 'M', 'FEMALE': 'F', 'nan': np.nan, 'N/A': np.nan})
            df_demo['sex'] = s
            
        return df_demo[['subject', 'session', 'age', 'sex']]
    else:
        return pd.DataFrame(columns=['subject', 'session', 'age', 'sex'])

def shorten_pipeline_name(pipeline_name):
    mapping = {
        'freesurfer741ants243': 'FS741',
        'samseg8001ants243': 'Samseg8', 
        'freesurfer8001ants243': 'FS8001',
        'fslanat6071ants243': 'FSL6071'
    }
    return mapping.get(pipeline_name, pipeline_name)

def get_structure_order():
    base_structures = ['Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Hippocampus', 'Amygdala', 'Accumbens-area']
    ordered_structures = []
    for struct in base_structures:
        ordered_structures.extend([f'Left-{struct}', f'Right-{struct}'])
    ordered_structures.append('Brainstem')
    return ordered_structures

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

    print(f"Applying Strict Filtering... Initial: {len(df_tidy)}")
    
    # 0. Sanitize: Drop rows with volume <= 0
    if 'volume_mm3' in df_tidy.columns:
        df_tidy = df_tidy[df_tidy['volume_mm3'] > 0]

    # 1. Filter to required pipelines ONLY first
    df_tidy = df_tidy[df_tidy['pipeline'].isin(required_pipelines)].copy()

    # 2. Count pipelines per structure per session
    counts = (
        df_tidy.groupby(['dataset', 'subject', 'session', 'structure'])['pipeline']
        .nunique()
        .reset_index(name='n_pipelines')
    )

    # 3. Identify "Bad Sessions": Any session that has at least one structure with < 4 pipelines
    incomplete_structures = counts[counts['n_pipelines'] < len(required_pipelines)]
    bad_sessions = incomplete_structures[['dataset', 'subject', 'session']].drop_duplicates()
    
    if len(bad_sessions) > 0:
        # 4. Perform Anti-Join to drop these bad sessions entirely
        merged = df_tidy.merge(bad_sessions, 
                               on=['dataset', 'subject', 'session'], 
                               how='left', 
                               indicator=True)
        df_filtered = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
        print(f"Dropped {len(bad_sessions)} incomplete sessions.")
    else:
        df_filtered = df_tidy.copy()
    
    print(f"Filtered: {len(df_filtered)}")
    return df_filtered

def create_age_distribution_plot(df, output_dir):
    subject_ages = df[['dataset', 'subject', 'session', 'age']].drop_duplicates().dropna(subset=['age'])
    if subject_ages.empty:
        return
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.figure(figsize=(12, 8))
    
    datasets = subject_ages['dataset'].unique()
    palette = sns.color_palette("tab10", len(datasets))
    
    for i, dataset in enumerate(datasets):
        dataset_ages = subject_ages[subject_ages['dataset'] == dataset]['age']
        n_scans = len(dataset_ages)
        sns.histplot(data=dataset_ages, bins=20, element='step', fill=True, alpha=0.4, color=palette[i], label=f'{dataset} (n={n_scans})', stat='count')
    
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.title('Age Distribution by Dataset (Unique Scans)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'age_distribution_by_dataset.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_composite_figure(mean_diff_df, corr_df, sex_df, output_dir):
    """
    Create a composite figure with 3 vertically stacked heatmaps sharing the X axis.
    1. Mean Relative Volume Difference (All values shown, 2 decimals)
    2. Age Correlation (Only significant values shown, 2 decimals)
    3. Sex Effect (Only significant values shown, 2 decimals)
    """
    print("Generating Composite MICCAI Figure...")
    
    structure_order = get_structure_order()
    
    # --- Prepare Matrices ---
    
    # 1. Mean Diff (No Significance Test available in current logic, show all)
    md_sub = mean_diff_df[mean_diff_df['structure'].isin(structure_order)].copy()
    md_sub['structure'] = pd.Categorical(md_sub['structure'], categories=structure_order, ordered=True)
    md_sub = md_sub.sort_values('structure')
    md_pivot = md_sub.pivot(index='structure', columns='pipeline_pair', values='volume_diff')
    
    # 2. Age Correlation (Values AND P-values)
    c_sub = corr_df[corr_df['structure'].isin(structure_order)].copy()
    c_sub['structure'] = pd.Categorical(c_sub['structure'], categories=structure_order, ordered=True)
    c_sub = c_sub.sort_values('structure')
    corr_pivot = c_sub.pivot(index='structure', columns='pipeline_pair', values='r')
    p_corr_pivot = c_sub.pivot(index='structure', columns='pipeline_pair', values='p_adj')
    
    # 3. Sex Effect (Values AND P-values)
    s_sub = sex_df[sex_df['structure'].isin(structure_order)].copy()
    s_sub['structure'] = pd.Categorical(s_sub['structure'], categories=structure_order, ordered=True)
    s_sub = s_sub.sort_values('structure')
    sex_pivot = s_sub.pivot(index='structure', columns='pipeline_pair', values='cohen_d')
    p_sex_pivot = s_sub.pivot(index='structure', columns='pipeline_pair', values='p_adj')

    # Ensure consistent column ordering (sorted alphabetically for alignment)
    all_cols = sorted(list(set(md_pivot.columns) | set(corr_pivot.columns) | set(sex_pivot.columns)))
    
    # Reindex ALL matrices to ensure strict alignment
    md_pivot = md_pivot.reindex(columns=all_cols)
    corr_pivot = corr_pivot.reindex(columns=all_cols)
    p_corr_pivot = p_corr_pivot.reindex(columns=all_cols)
    sex_pivot = sex_pivot.reindex(columns=all_cols)
    p_sex_pivot = p_sex_pivot.reindex(columns=all_cols)

    # --- Create Custom Annotation Matrices ---
    # Helper to enforce ".2f" including trailing zeros (0.40 not 0.4)
    def strict_fmt(x):
        return '{:.2f}'.format(x) if pd.notnull(x) else ''

    # Annot 1: Mean Diff - Show ALL values
    annot_md = md_pivot.applymap(strict_fmt)

    # Annot 2: Age - Show ONLY significant (p < 0.05)
    annot_corr = corr_pivot.applymap(strict_fmt)
    mask_corr = (p_corr_pivot >= 0.05) | (p_corr_pivot.isna())
    annot_corr = annot_corr.mask(mask_corr, '')

    # Annot 3: Sex - Show ONLY significant (p < 0.05)
    annot_sex = sex_pivot.applymap(strict_fmt)
    mask_sex = (p_sex_pivot >= 0.05) | (p_sex_pivot.isna())
    annot_sex = annot_sex.mask(mask_sex, '')

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True, constrained_layout=True)
    
    # Plot 1: Mean Diff
    sns.heatmap(md_pivot, ax=axes[0], annot=annot_md, fmt='', cmap='viridis', 
                cbar_kws={'label': 'Mean Rel. Diff'})
    axes[0].set_title('Mean Relative Volume Difference by Structure')
    axes[0].set_xlabel('')
    
    # Plot 2: Age Correlation
    sns.heatmap(corr_pivot, ax=axes[1], annot=annot_corr, fmt='', cmap='coolwarm', center=0, 
                cbar_kws={'label': 'Spearman r'})
    axes[1].set_title('Spearman Correlation of Relative Volume Differences with Age')
    axes[1].set_xlabel('')
    
    # Plot 3: Sex Effect
    sns.heatmap(sex_pivot, ax=axes[2], annot=annot_sex, fmt='', cmap='vlag', center=0, 
                cbar_kws={'label': "Cohen's d"})
    axes[2].set_title("Sex Effect (Cohen's d) on Relative Volume Differences")
    axes[2].set_xlabel('Pipeline Pair')

    # Ensure Y-labels are on every plot
    for ax in axes:
        ax.set_ylabel('Structure')
        plt.setp(ax.get_yticklabels(), rotation=0)

    # Rotate X-labels on the bottom plot only
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha='right')

    plt.savefig(output_dir / 'composite_summary_figure.png', dpi=300)
    plt.close()

# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def main():
    print("Loading df_tidy.csv...")
    df_tidy_path = Path('/home/runner/work/NeuroCI/NeuroCI/experiment_state/figures/df_tidy.csv')
    df = pd.read_csv(df_tidy_path)

    # -------------------------------------------------------------------------
    # 1. APPLY GRAND INTERSECTION FILTER
    # -------------------------------------------------------------------------
    # Ensure volume is numeric and valid first
    if 'volume_mm3' in df.columns:
        df['volume_mm3'] = pd.to_numeric(df['volume_mm3'], errors='coerce')
        df = df.dropna(subset=['volume_mm3'])
    
    # CRITICAL: Filter complete cases (which includes the > 0 check)
    df = filter_complete_pipelines(df)

    # NOW filter to relevant structures for plotting
    required_structures = get_structure_order()
    df = df[df['structure'].isin(required_structures)].copy()

    # -------------------------------------------------------------------------
    # 2. MAP DEMOGRAPHICS
    # -------------------------------------------------------------------------
    print("Building Demographics Mapping...")
    age_mapping = {}
    sex_mapping = {}

    datasets_in_tidy = df['dataset'].unique()
    
    for dataset_name in datasets_in_tidy:
        dataset_path = ROOT / dataset_name
        if not dataset_path.exists():
            continue
            
        dfs = load_tabular_data(dataset_path)
        df_demo_dataset = extract_demographics(dataset_name, dfs)
        
        if df_demo_dataset is not None and not df_demo_dataset.empty:
            for _, row in df_demo_dataset.iterrows():
                subj = str(row["subject"]).strip()
                sess = str(row["session"]).strip() if pd.notna(row["session"]) else np.nan
                
                if pd.notna(sess) and sess != 'nan':
                    age_mapping[(dataset_name, subj, sess)] = row["age"]
                    sex_mapping[(dataset_name, subj, sess)] = row["sex"]
                    sex_mapping[(dataset_name, subj)] = row["sex"]
                else:
                    age_mapping[(dataset_name, subj)] = row["age"]
                    sex_mapping[(dataset_name, subj)] = row["sex"]

    # Apply Mapping
    print("Applying Age/Sex Mapping...")
    def get_age(row):
        val = age_mapping.get((row["dataset"], row["subject"], row["session"]), np.nan)
        if pd.isna(val):
            val = age_mapping.get((row["dataset"], row["subject"]), np.nan)
        return val
    
    def get_sex(row):
        val = sex_mapping.get((row["dataset"], row["subject"], row["session"]), np.nan)
        if pd.isna(val):
            val = sex_mapping.get((row["dataset"], row["subject"]), np.nan)
        return val

    df["age"] = df.apply(get_age, axis=1)
    df["sex"] = df.apply(get_sex, axis=1)
    
    print(f"Final Data with Demographics: {len(df)}")

    create_age_distribution_plot(df, EXPERIMENT_STATE_ROOT)

    # -------------------------------------------------------------------------
    # 3. ANALYSIS (Corrected for Head Size Confound)
    # -------------------------------------------------------------------------
    pairwise_diffs = []
    pipelines = df['pipeline'].unique()

    for (pipe1, pipe2) in itertools.combinations(pipelines, 2):
        df1 = df[df['pipeline'] == pipe1]
        df2 = df[df['pipeline'] == pipe2]
        merged = df1.merge(df2, on=['dataset', 'subject', 'session', 'structure'], suffixes=(f'_{pipe1}', f'_{pipe2}'))

        # --- CHANGED: Relative Difference Calculation (Normalized by Mean Volume) ---
        mean_vol = (merged['volume_mm3_' + pipe1] + merged['volume_mm3_' + pipe2]) / 2.0
        abs_diff = (merged['volume_mm3_' + pipe1] - merged['volume_mm3_' + pipe2]).abs()
        
        # Avoid division by zero, though complete case filter should prevent 0 volumes
        merged['volume_diff'] = abs_diff / mean_vol.replace(0, np.nan)
        # ---------------------------------------------------------------------------
        
        short_pipe1 = shorten_pipeline_name(pipe1)
        short_pipe2 = shorten_pipeline_name(pipe2)
        merged['pipeline_pair'] = f"{short_pipe1}_vs_{short_pipe2}"
        
        merged['age'] = merged['age_' + pipe1].combine_first(merged['age_' + pipe2])
        merged['sex'] = merged['sex_' + pipe1].combine_first(merged['sex_' + pipe2])

        pairwise_diffs.append(merged[['dataset', 'subject', 'session', 'structure', 'pipeline_pair', 'volume_diff', 'age', 'sex']])

    if not pairwise_diffs:
        print("No pairwise intersections found.")
        return

    df_diff = pd.concat(pairwise_diffs, ignore_index=True)

    # Initialize storage variables for the final composite plot
    final_corr_df = pd.DataFrame()
    final_sex_df = pd.DataFrame()
    final_mean_diff_df = pd.DataFrame()

    # Correlation
    df_age = df_diff[['dataset','subject','session','pipeline_pair','structure','volume_diff','age']].copy()
    df_age['volume_diff'] = pd.to_numeric(df_age['volume_diff'], errors='coerce')
    df_age['age'] = pd.to_numeric(df_age['age'], errors='coerce')
    
    corr_results = []
    for (pair, struct), g in df_age.groupby(['pipeline_pair','structure']):
        g = g.dropna(subset=['volume_diff','age'])
        n = g.groupby(['dataset','subject','session']).ngroups
        if n < 3: continue
        try:
            r, p = spearmanr(g['volume_diff'], g['age'])
        except ValueError:
            r, p = np.nan, np.nan
        corr_results.append({'pipeline_pair': pair, 'structure': struct, 'r': r, 'p': p, 'n': n})
    
    corr_df = pd.DataFrame(corr_results).dropna(subset=['r', 'p'])
    if not corr_df.empty:
        corr_df['p_adj'] = np.minimum(corr_df['p'] * len(corr_df), 1.0)
        corr_df.to_csv(EXPERIMENT_STATE_ROOT / 'age_correlation_summary_spearman.csv', index=False)
        final_corr_df = corr_df.copy() # Store for composite
        
        structure_order = get_structure_order()
        corr_df_ordered = corr_df[corr_df['structure'].isin(structure_order)].copy()
        corr_df_ordered['structure'] = pd.Categorical(corr_df_ordered['structure'], categories=structure_order, ordered=True)
        corr_df_ordered = corr_df_ordered.sort_values('structure')
        
        corr_pivot = corr_df_ordered.pivot(index='structure', columns='pipeline_pair', values='r')
        p_pivot = corr_df_ordered.pivot(index='structure', columns='pipeline_pair', values='p_adj')
        
        corr_rounded = corr_pivot.round(2)
        annot_matrix = corr_rounded.astype(str)
        annot_matrix[p_pivot >= 0.05] = ''
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_pivot, annot=annot_matrix, fmt='', cmap='coolwarm', center=0, cbar_kws={'label': 'Spearman r'})
        plt.title('Spearman Correlation of Relative Volume Differences with Age')
        plt.tight_layout()
        plt.savefig(EXPERIMENT_STATE_ROOT / 'corr_age_spearman_heatmap.png', dpi=300)
        plt.close()

    # Sex effects
    df_sex = df_diff[['dataset','subject','session','pipeline_pair','structure','volume_diff','sex']].copy()
    df_sex['volume_diff'] = pd.to_numeric(df_sex['volume_diff'], errors='coerce')

    sex_results = []
    for (pair, struct), g in df_sex.groupby(['pipeline_pair','structure']):
        g = g.dropna(subset=['volume_diff','sex'])
        n = g.groupby(['dataset','subject','session']).ngroups
        males = g[g['sex'].str.lower().str.startswith('m')]['volume_diff']
        females = g[g['sex'].str.lower().str.startswith('f')]['volume_diff']
        if len(males) < 2 or len(females) < 2: continue
        t, p = ttest_ind(males, females, equal_var=False)
        cohen_d = (males.mean() - females.mean()) / np.sqrt(((males.std() ** 2 + females.std() ** 2) / 2))
        sex_results.append({'pipeline_pair': pair, 'structure': struct, 't': t, 'p': p, 'cohen_d': cohen_d, 'n': n})

    sex_df = pd.DataFrame(sex_results).dropna(subset=['t', 'p'])
    if not sex_df.empty:
        sex_df['p_adj'] = np.minimum(sex_df['p'] * len(sex_df), 1.0)
        sex_df.to_csv(EXPERIMENT_STATE_ROOT / 'sex_effects_summary.csv', index=False)
        final_sex_df = sex_df.copy() # Store for composite
        
        structure_order = get_structure_order()
        sex_df_ordered = sex_df[sex_df['structure'].isin(structure_order)].copy()
        sex_df_ordered['structure'] = pd.Categorical(sex_df_ordered['structure'], categories=structure_order, ordered=True)
        sex_df_ordered = sex_df_ordered.sort_values('structure')
        
        sex_pivot = sex_df_ordered.pivot(index='structure', columns='pipeline_pair', values='cohen_d')
        p_pivot_sex = sex_df_ordered.pivot(index='structure', columns='pipeline_pair', values='p_adj')
        
        sex_rounded = sex_pivot.round(2)
        annot_matrix_sex = sex_rounded.astype(str)
        annot_matrix_sex[p_pivot_sex >= 0.05] = ''
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sex_pivot, annot=annot_matrix_sex, fmt='', cmap='vlag', center=0, cbar_kws={'label': "Cohen's d"})
        plt.title("Sex Effect (Cohen's d) on Relative Volume Differences")
        plt.tight_layout()
        plt.savefig(EXPERIMENT_STATE_ROOT / 'sex_effects_heatmap_significant.png', dpi=300)
        plt.close()

    # -------------------------------------------------------------------------
    # NEW: Mean Volume Difference Heatmap
    # -------------------------------------------------------------------------
    mean_diff_results = df_diff.groupby(['pipeline_pair', 'structure'])['volume_diff'].mean().reset_index()
    
    if not mean_diff_results.empty:
        mean_diff_results.to_csv(EXPERIMENT_STATE_ROOT / 'mean_vol_diff_summary.csv', index=False)
        final_mean_diff_df = mean_diff_results.copy() # Store for composite

        structure_order = get_structure_order()
        mean_diff_ordered = mean_diff_results[mean_diff_results['structure'].isin(structure_order)].copy()
        mean_diff_ordered['structure'] = pd.Categorical(mean_diff_ordered['structure'], categories=structure_order, ordered=True)
        mean_diff_ordered = mean_diff_ordered.sort_values('structure')

        mean_diff_pivot = mean_diff_ordered.pivot(index='structure', columns='pipeline_pair', values='volume_diff')
        
        # Round for annotation
        mean_diff_rounded = mean_diff_pivot.round(3)
        annot_matrix_mean = mean_diff_rounded.astype(str)

        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_diff_pivot, annot=annot_matrix_mean, fmt='', cmap='viridis', cbar_kws={'label': 'Mean Relative Volume Diff'})
        plt.title('Mean Relative Volume Difference by Structure')
        plt.tight_layout()
        plt.savefig(EXPERIMENT_STATE_ROOT / 'mean_vol_diff_heatmap.png', dpi=300)
        plt.close()

    # -------------------------------------------------------------------------
    # 4. GENERATE COMPOSITE FIGURE (MICCAI)
    # -------------------------------------------------------------------------
    if not final_mean_diff_df.empty and not final_corr_df.empty and not final_sex_df.empty:
        create_composite_figure(final_mean_diff_df, final_corr_df, final_sex_df, EXPERIMENT_STATE_ROOT)
    else:
        print("Skipping composite figure: Insufficient data in one of the 3 metrics.")

if __name__ == '__main__':
    main()
