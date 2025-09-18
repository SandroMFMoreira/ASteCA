import os
import json
import subprocess
import pandas as pd

from catalogues import get_catalogue

# Load data
df = get_catalogue(['Dias'])[0]
df = df[(df.r_sun < 2000) & (df.age < 2000)]
cluster_names = ['BH_99'] #df['cluster'].unique() # ['Alessi_2', 'Alessi_3', 'Alessi_24'] #

#cluster_names = pd.read_csv('Dias_parsec_UBVRI_cmd_ccd/classif_parsec_UBVRI_cmd_ccd.csv')
#cluster_names = cluster_names[cluster_names.age_classif == 'A'].cluster

# Create directories if they don't exist
config_dir = 'cluster_configs'
os.makedirs(config_dir, exist_ok=True)

# Global config
evolution_model = 'parsec'
l_adjust        = 'cmd_ccd'
phot_system     = 'UBVRI'
av_fixed        = False
results_dir = f'Dias_{evolution_model}_{phot_system}_cmd_ccd_av_fixed/Plots/'
os.makedirs(results_dir, exist_ok=True)
results_name = f"Dias_{evolution_model}_{phot_system}_cmd_ccd_av_fixed.csv"
overwrite = False  # <-- SET THIS TO True to force rerun


# If av_fixed, load the AV results once:
if av_fixed:
    av_df = pd.read_csv('Dias_parsec_UBVRI_cmd_ccd/Dias_parsec_UBVRI_cmd_ccd.csv')
    available_clusters = set(av_df['cluster'])
    cluster_names = [name for name in cluster_names if name in available_clusters]


def run_cluster_fit(cluster_name):
    output_pdf = os.path.join(results_dir, f"{cluster_name}.pdf")
    if not overwrite and os.path.exists(output_pdf):
        print(f"[SKIP] {cluster_name}: PDF already exists.")
        return

    # Build per-cluster config
    config = {
        'cluster_name':    cluster_name,
        'evolution_model': evolution_model,
        'l_adjust':        l_adjust,
        'phot_system':     phot_system,
        'av_fixed':        av_fixed,
        'results_dir':     results_dir,
    }

    if av_fixed:
        av_val = float(av_df.loc[av_df['cluster'] == cluster_name, 'av'].iloc[0])
        config['av_value'] = av_val

    # Write JSON config
    config_path = os.path.join(config_dir, f"{cluster_name}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Run worker
    ret = subprocess.run(
        ['python', '-u','age_inference.py', config_path],
        capture_output=True,
        text=True
    )
    if ret.returncode != 0:
        print(f"[ERROR] {cluster_name} failed:")
        print(ret.stdout)
        print(ret.stderr)
    else:
        print(f"[OK]    {cluster_name} completed.")


if __name__ == '__main__':
    for name in cluster_names:
        run_cluster_fit(name)

    # Merge results
    result_files = [
        os.path.join(f"./junk/results_{name}.csv")
        for name in cluster_names
        if os.path.exists(f"./junk/results_{name}.csv")
    ]

    if result_files:
        combined = pd.concat((pd.read_csv(f) for f in result_files), ignore_index=True)
        merged_path = os.path.join(f'{results_name}.csv')
        combined.to_csv(merged_path, index=False)
        print(f"Merged all results into {merged_path}")
    else:
        print("No result files were generated.")
