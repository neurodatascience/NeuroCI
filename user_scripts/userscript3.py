import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from scipy.stats import pearsonr, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EXPERIMENT_STATE_ROOT = Path(__file__).resolve().parents[1] / "experiment_state" / "figures"
EXPERIMENT_STATE_ROOT.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_tabular_data(dataset_path: Path):
    """Load all TSV files inside a dataset's 'tabular' directory into a dictionary."""
    tabular_path = dataset_path / "tabular"
    dfs = {}
    for tsv in tabular_path.glob("*.tsv"):
        df = pd.read_csv(tsv, sep='\t')
        dfs[tsv.stem] = df
    return dfs

def extract_demographics(dataset_name, dfs):
    """Extract demographic information (age, sex) from dataset-specific tabular data."""
    df_demo = None
    if dataset_name.lower() == 'preventad':
        if 'participants' in dfs and 'ad8' in dfs:
            part = dfs['participants']
            ad8 = dfs['ad8']
            ad8 = ad8.rename(columns={'participant_id': 'subject', 'Candidate_Age': 'age_months'})
            ad8['age'] = ad8['age_months'] / 12.0
            df_demo = ad8.merge(part, left_on='subject', right_on='participant_id', how='left')
            df_demo = df_demo[['subject', 'visit_id', 'age', 'sex']]
    elif dataset_name.lower() == 'ds005752':
        df = dfs.get('participants', pd.DataFrame())
        if not df.empty:
            df_demo = df[['participant_id', 'age', 'sex']].rename(columns={'participant_id': 'subject'})
    elif dataset_name.lower() == 'ds003592':
        df = dfs.get('participants', pd.DataFrame())
        if not df.empty:
            df_demo = df[['participant_id', 'age', 'sex']].rename(columns={'participant_id': 'subject'})
    else:
        if 'participants' in dfs:
            df_demo = dfs['participants'].rename(columns={'participant_id': 'subject'})
            if 'age' not in df_demo.columns:
                df_demo['age'] = np.nan
            if 'sex' not in df_demo.columns:
                df_demo['sex'] = np.nan
    return df_demo

# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def main():
    df_tidy = pd.read_csv(Path('/home/runner/work/NeuroCI/NeuroCI/experiment_state/figures/df_tidy.csv'))
    df_all = []
    root = Path('/tmp/neuroci_output_state')

    # Merge demographic data into tidy volumes for all datasets
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dfs = load_tabular_data(dataset_dir)
        df_demo = extract_demographics(dataset_dir.name, dfs)
        if df_demo is None or df_demo.empty:
            continue
        df_dataset = df_tidy[df_tidy['dataset'] == dataset_dir.name].copy()
        df_dataset = df_dataset.merge(df_demo, how='left', on='subject')
        df_all.append(df_dataset)

    if not df_all:
        print("No datasets with usable demographic data found.")
        return

    df = pd.concat(df_all, ignore_index=True)

    # -------------------------------------------------------------------------
    # Compute pairwise pipeline differences (absolute values)
    # -------------------------------------------------------------------------
    pairwise_diffs = []
    pipelines = df['pipeline'].unique()

    for (pipe1, pipe2) in itertools.combinations(pipelines, 2):
        df1 = df[df['pipeline'] == pipe1]
        df2 = df[df['pipeline'] == pipe2]
        merged = df1.merge(df2, on=['dataset', 'subject', 'session', 'structure'], suffixes=(f'_{pipe1}', f'_{pipe2}'))

        # Compute absolute volume difference between pipelines
        merged['volume_diff'] = (merged['volume_mm3_' + pipe1] - merged['volume_mm3_' + pipe2]).abs()
        merged['pipeline_pair'] = f"{pipe1}_vs_{pipe2}"
        merged['age'] = merged['age_' + pipe1].combine_first(merged['age_' + pipe2])
        merged['sex'] = merged['sex_' + pipe1].combine_first(merged['sex_' + pipe2])

        pairwise_diffs.append(merged[['dataset', 'subject', 'session', 'structure', 'pipeline_pair', 'volume_diff', 'age', 'sex']])

    df_diff = pd.concat(pairwise_diffs, ignore_index=True)

    # -------------------------------------------------------------------------
    # Correlation with age per pipeline_pair × structure
    # -------------------------------------------------------------------------
    df_age = df_diff[['dataset','subject','session','pipeline_pair','structure','volume_diff','age']].copy()
    df_age['volume_diff'] = pd.to_numeric(df_age['volume_diff'], errors='coerce')
    df_age['age'] = pd.to_numeric(df_age['age'], errors='coerce')

    corr_results = []
    for (pair, struct), g in df_age.groupby(['pipeline_pair','structure']):
        g = g.dropna(subset=['volume_diff','age'])
        n = g.groupby(['dataset','subject','session']).ngroups  # unique scans
        if n < 3:
            continue
        r, p = pearsonr(g['volume_diff'], g['age'])
        corr_results.append({'pipeline_pair': pair, 'structure': struct, 'r': r, 'p': p, 'n': n})

    corr_df = pd.DataFrame(corr_results)
    corr_df['p_adj'] = np.minimum(corr_df['p'] * len(corr_df), 1.0)
    corr_pivot = corr_df.pivot(index='structure', columns='pipeline_pair', values='r')

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_pivot, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation of Volume Differences with Age (n varies per pipeline_pair×structure)')
    plt.tight_layout()
    plt.savefig(EXPERIMENT_STATE_ROOT / 'corr_age_heatmap.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Sex effects per pipeline_pair × structure
    # -------------------------------------------------------------------------
    df_sex = df_diff[['dataset','subject','session','pipeline_pair','structure','volume_diff','sex']].copy()
    df_sex['volume_diff'] = pd.to_numeric(df_sex['volume_diff'], errors='coerce')

    sex_results = []
    for (pair, struct), g in df_sex.groupby(['pipeline_pair','structure']):
        g = g.dropna(subset=['volume_diff','sex'])
        n = g.groupby(['dataset','subject','session']).ngroups  # unique scans
        males = g[g['sex'].str.lower().str.startswith('m')]['volume_diff']
        females = g[g['sex'].str.lower().str.startswith('f')]['volume_diff']
        if len(males) < 2 or len(females) < 2:
            continue
        t, p = ttest_ind(males, females, equal_var=False)
        cohen_d = (males.mean() - females.mean()) / np.sqrt(((males.std() ** 2 + females.std() ** 2) / 2))
        sex_results.append({'pipeline_pair': pair, 'structure': struct, 't': t, 'p': p, 'cohen_d': cohen_d, 'n': n})

    sex_df = pd.DataFrame(sex_results)
    sex_df['p_adj'] = np.minimum(sex_df['p'] * len(sex_df), 1.0)

    # Save summary CSV of t-tests
    sex_df.to_csv(EXPERIMENT_STATE_ROOT / 'sex_effects_summary.csv', index=False)

    sex_pivot = sex_df.pivot(index='structure', columns='pipeline_pair', values='cohen_d')
    plt.figure(figsize=(10, 6))
    sns.heatmap(sex_pivot, annot=True, cmap='vlag', center=0)
    plt.title('Sex Effect (Cohen\'s d) on Volume Differences (n varies per pipeline_pair×structure)')
    plt.tight_layout()
    plt.savefig(EXPERIMENT_STATE_ROOT / 'sex_effects_heatmap.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
