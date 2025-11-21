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
    
    # Handle comma-separated multiple values by taking the first one
    if df_demo is not None and not df_demo.empty:
        # For age: take first value and convert to float
        if 'age' in df_demo.columns:
            df_demo['age'] = df_demo['age'].astype(str).str.split(',').str[0]
            # Replace 'n/a' with NaN and convert to float
            df_demo['age'] = df_demo['age'].replace('n/a', np.nan)
            df_demo['age'] = pd.to_numeric(df_demo['age'], errors='coerce')
        
        # For sex: take first value
        if 'sex' in df_demo.columns:
            df_demo['sex'] = df_demo['sex'].astype(str).str.split(',').str[0]
            # Replace 'n/a' with NaN
            df_demo['sex'] = df_demo['sex'].replace('n/a', np.nan)
    
    return df_demo

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
    # Define base structures (without hemisphere prefix)
    base_structures = [
        'Thalamus', 'Caudate', 'Putamen', 'Pallidum', 
        'Hippocampus', 'Amygdala', 'Accumbens-area'
    ]
    
    # Create pairs: left then right for each structure
    ordered_structures = []
    for struct in base_structures:
        ordered_structures.extend([f'Left-{struct}', f'Right-{struct}'])
    
    # Add Brainstem at the end
    ordered_structures.append('Brainstem')
    
    return ordered_structures

def create_age_distribution_plot(df, output_dir):
    """Create a histogram showing age distributions for each dataset."""
    # Get unique subjects with their ages and datasets
    subject_ages = df[['dataset', 'subject', 'age']].drop_duplicates().dropna(subset=['age'])
    
    if subject_ages.empty:
        print("No age data available for plotting.")
        return
    
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create overlapping histograms for each dataset
    datasets = subject_ages['dataset'].unique()
    palette = sns.color_palette("tab10", len(datasets))
    
    for i, dataset in enumerate(datasets):
        dataset_ages = subject_ages[subject_ages['dataset'] == dataset]['age']
        n_subjects = len(dataset_ages)
        
        sns.histplot(
            data=dataset_ages,
            bins=20,
            element='step',
            fill=True,
            alpha=0.4,
            color=palette[i],
            label=f'{dataset} (n={n_subjects})',
            stat='count'
        )
    
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.title('Age Distribution by Dataset')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / 'age_distribution_by_dataset.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved age distribution plot: {output_path}")

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

        # Compute absolute volume difference between pipelines
        merged['volume_diff'] = (merged['volume_mm3_' + pipe1] - merged['volume_mm3_' + pipe2]).abs()
        
        # Use shortened pipeline names for the pair
        short_pipe1 = shorten_pipeline_name(pipe1)
        short_pipe2 = shorten_pipeline_name(pipe2)
        merged['pipeline_pair'] = f"{short_pipe1}_vs_{short_pipe2}"
        
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
    
    # Save summary CSV
    corr_df.to_csv(EXPERIMENT_STATE_ROOT / 'age_correlation_summary.csv', index=False)
    
    # Prepare pivot tables with ordered structures
    structure_order = get_structure_order()
    
    # Filter to only include structures in our desired order
    corr_df_ordered = corr_df[corr_df['structure'].isin(structure_order)].copy()
    
    # Convert structure to categorical with desired order
    corr_df_ordered['structure'] = pd.Categorical(
        corr_df_ordered['structure'], 
        categories=structure_order, 
        ordered=True
    )
    
    # Sort by structure order
    corr_df_ordered = corr_df_ordered.sort_values('structure')
    
    corr_pivot = corr_df_ordered.pivot(index='structure', columns='pipeline_pair', values='r')
    p_pivot = corr_df_ordered.pivot(index='structure', columns='pipeline_pair', values='p_adj')
    
    # Round r values for display
    corr_rounded = corr_pivot.round(2)
    
    # Mask annotations where p_adj >= 0.05
    annot_matrix = corr_rounded.astype(str)
    annot_matrix[p_pivot >= 0.05] = ''  # empty string for non-significant
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_pivot, annot=annot_matrix, fmt='', cmap='coolwarm', center=0,
                cbar_kws={'label': 'r'})
    plt.title('Correlation of Volume Differences with Age (significant r values only)')
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

    # Apply structure ordering to sex results
    sex_df_ordered = sex_df[sex_df['structure'].isin(structure_order)].copy()
    sex_df_ordered['structure'] = pd.Categorical(
        sex_df_ordered['structure'], 
        categories=structure_order, 
        ordered=True
    )
    sex_df_ordered = sex_df_ordered.sort_values('structure')
    
    sex_pivot = sex_df_ordered.pivot(index='structure', columns='pipeline_pair', values='cohen_d')
    plt.figure(figsize=(10, 8))
    sns.heatmap(sex_pivot, annot=True, cmap='vlag', center=0)
    plt.title('Sex Effect (Cohen\'s d) on Volume Differences (n varies per pipeline_pair×structure)')
    plt.tight_layout()
    plt.savefig(EXPERIMENT_STATE_ROOT / 'sex_effects_heatmap.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    main()
