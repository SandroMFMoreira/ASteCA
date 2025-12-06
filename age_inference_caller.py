import os
import json
import subprocess
import pandas as pd
from catalogues import get_catalogue

# --- Load classification catalogue ---
df_classif = pd.read_csv('./Dias_parsec_UBVRI_cmd_ccd_kde/classif_parsec_Gaia_cmd_ccd.csv')

# Clusters with classification not equal to 'A'
clusters_non_A = set(df_classif.loc[(df_classif.age_classif !=  'A') & (df_classif.age_classif !=  'B'), 'cluster'].unique())

# # --- Load Hunt catalogue ---
# df = get_catalogue(['Hunt'])[0]
# df = df[(df.r_sun < 1000) & (df.age < 300)]

# --- Load Hunt catalogue ---
df = get_catalogue(['Dias'])[0]
df = df[(df.r_sun < 1500) & (df.age < 300)]

clusters_hunt = set(df['cluster'].unique())

# --- Clusters that appear in Hunt but NOT in any classification ---
clusters_with_classif = set(df_classif['cluster'].unique())
clusters_hunt_only = clusters_hunt - clusters_with_classif

# --- Final
cluster_names = sorted(clusters_non_A.union(clusters_hunt_only))


# Create directories if they don't exist
config_dir = 'cluster_configs'
os.makedirs(config_dir, exist_ok=True)

# Global config
evolution_model = 'parsec'
l_adjust = 'cmd_ccd'
phot_system = 'UBVRI'
av_fixed = False
results_name = f'Dias_{evolution_model}_{phot_system}_cmd_ccd_kde'
results_dir = results_name + '/Plots/'
os.makedirs(results_dir, exist_ok=True)
overwrite = True  # <-- SET THIS TO True to force rerun

# If av_fixed, load the AV results once:
if av_fixed:
    av_df = pd.read_csv('Sandro_parsec_UBVRI_cmd_ccd_kde/Sandro_parsec_UBVRI_cmd_ccd_kde.csv')
    available_clusters = set(av_df['cluster'])
    cluster_names = [name for name in cluster_names if name in available_clusters]


def run_cluster_fit(cluster_name):
    output_pdf = os.path.join(results_dir, f"{cluster_name}.pdf")
    if not overwrite and os.path.exists(output_pdf):
        print(f"[SKIP] {cluster_name}: PDF already exists.")
        return

    # Build per-cluster config
    config = {
        'cluster_name': cluster_name,
        'evolution_model': evolution_model,
        'l_adjust': l_adjust,
        'phot_system': phot_system,
        'av_fixed': av_fixed,
        'results_dir': results_dir,
    }

    if av_fixed:
        av_val = float(av_df.loc[av_df['cluster'] == cluster_name, 'av'].iloc[0])
        config['av_value'] = av_val

    # Write JSON config
    config_path = os.path.join(config_dir, f"{cluster_name}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # prepare log file path
    log_path = os.path.join(config_dir, f"{cluster_name}.log")

    # Run worker and stream output live, while also saving to a log file
    cmd = ['python', '-u', 'age_inference.py', config_path]
    # Use Popen so we can stream stdout/stderr line-by-line
    with open(log_path, 'w', encoding='utf-8') as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # line-buffered
        )
        # Stream stdout to both console and log file
        try:
            for line in proc.stdout:
                # Remove trailing newline only when printing, keep in log
                print(line, end='')        # live to console
                logf.write(line)           # save in log
            proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            proc.wait()
            print(f"[INTERRUPT] {cluster_name} killed by user.")
            return

    if proc.returncode != 0:
        print(f"[ERROR] {cluster_name} failed (see {log_path}): returncode={proc.returncode}")
    else:
        print(f"[OK]    {cluster_name} completed (log: {log_path}).")


if __name__ == '__main__':
    for name in cluster_names:
        run_cluster_fit(name)

    # Merge results
    result_files = [
        os.path.join(f"./{results_name}/junk/results_{name}.csv")
        for name in cluster_names
        if os.path.exists(f"./{results_name}/junk/results_{name}.csv")
    ]

    if result_files:
        combined = pd.concat((pd.read_csv(f) for f in result_files), ignore_index=True)
        merged_path = os.path.join(f'{results_name}.csv')
        combined.to_csv(merged_path, index=False)
        print(f"Merged all results into {merged_path}")
    else:
        print("No result files were generated.")
