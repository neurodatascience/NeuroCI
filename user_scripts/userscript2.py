import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def create_boxen_figures(df_tidy, output_dir):
    """Create boxen plots which show more percentiles"""
    
    # Set style for better visuals
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    for dataset in df_tidy['dataset'].unique():
        dataset_data = df_tidy[df_tidy['dataset'] == dataset]
        
        print(f"Creating boxen plots for {dataset} with {len(dataset_data)} rows...")
        
        g = sns.catplot(
            data=dataset_data,
            x='pipeline', 
            y='volume_mm3',
            col='structure',
            col_wrap=5,
            kind='boxen',  # Enhanced box plot
            height=3,
            aspect=1.2,
            sharey=False,
            palette='viridis'
        )
        
        g.set_titles("{col_name}")
        g.set_xticklabels(rotation=45)
        g.fig.suptitle(f'Pipeline Variability (Boxen) - {dataset}', y=1.02, fontsize=16)
        
        output_path = output_dir / f'boxen_comparison_{dataset}.png'
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
        create_boxen_figures(df_tidy, EXPERIMENT_STATE_ROOT)
        print("✓ Boxen plots generated successfully!")
        
    else:
        print(f"✗ File not found: {tidy_path}")
        print("Please run the data extraction script first to generate df_tidy.csv")
