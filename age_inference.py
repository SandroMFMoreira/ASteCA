import os
import sys
import tempfile
import json
import numpy as np
import pandas as pd
import asteca
import pyabc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

av_U=1.55814
av_B=1.32616
av_V=1.00096
av_R=0.80815
av_I=0.59893
av_eVI = av_V-av_I


av_G = 0.83627
av_BP = 1.08337
av_RP = 0.63439
av_eBR = av_BP-av_RP

lambdas = [3598.54, 4385.92, 5490.56, 6594.72, 8059.88, 12369.26, 16464.45, 22105.45, 6390.21, 5182.58, 7825.08]
bands = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'G', 'G_BP', 'G_RP']

# Create the DataFrame
eff_lambda_df = pd.DataFrame([lambdas], columns=bands)

def load_isochrones(evolution_model, phot_system):
    if evolution_model == 'baraffe':
        if phot_system == 'Gaia':
            iso_path = "./Isochrones/BHAC15_iso.txt"
            magnitude = "G"
            color1 = ("G_BP", "G_RP")
            color2 = ("G", "G_RP")
            mag_effl = eff_lambda_df["G"].values
            color_effl1 = (
                eff_lambda_df['G_BP'].values,
                eff_lambda_df['G_RP'].values
            )
            color_effl2 = (
                eff_lambda_df['G'].values,
                eff_lambda_df['G_RP'].values
            )

        elif phot_system == 'UBVRI':
            iso_path = "./Isochrones/BHAC15_iso_UBVRI.txt"
            magnitude = "Mv"
            color1 = ("Mv", "Mi")
            color2 = None
            mag_effl = eff_lambda_df["V"].values
            color_effl1 = (
                eff_lambda_df['V'].values,
                eff_lambda_df['I'].values
            )
            color_effl2 = None

        else:
            raise ValueError("phot_system must be 'Gaia' or 'UBVRI'")

    elif evolution_model == 'parsec':
        if phot_system == 'Gaia':
            iso_path = "./Isochrones/IsochronesParsecGaia.dat"
            magnitude = "Gmag"
            color1 = ("G_BPmag", "G_RPmag")
            color2 = ("Gmag", "G_RPmag")
            mag_effl = eff_lambda_df["G"].values
            color_effl1 = (
                eff_lambda_df["G_BP"].values,
                eff_lambda_df["G_RP"].values
            )
            color_effl2 = (
                eff_lambda_df["G"].values,
                eff_lambda_df["G_RP"].values
            )

        elif phot_system == 'UBVRI':
            iso_path = "./Isochrones/isocronas_UBVRI_idades_6.5-10.dat"
            magnitude = "Vmag"
            color1 = ("Vmag", "Imag")
            color2 = ("Umag", "Bmag")
            mag_effl = eff_lambda_df["V"].values
            color_effl1 = (
                eff_lambda_df["V"].values,
                eff_lambda_df["I"].values
            )
            color_effl2 = (
                eff_lambda_df["U"].values,
                eff_lambda_df["B"].values
            )

        else:
            raise ValueError("phot_system must be 'Gaia' or 'UBVRI'")

    else:
        raise ValueError("evolution_model must be 'baraffe' or 'parsec'")

    return asteca.isochrones(
        model=evolution_model,
        isochs_path=iso_path,
        magnitude=magnitude,
        color=color1,
        color2=color2,
        magnitude_effl=mag_effl,
        color_effl=color_effl1,
        color2_effl=color_effl2,
    )


def setup_synthetic_clusters(isochrones):
    return asteca.synthetic(isochrones, seed=4573304)


def process_cluster(cluster_name, cluster_df, synthcl, my_cluster, l_adjust, av_fixed, cluster_av, results_dir):
    # Create DataFrame with one row
    cluster_results = pd.DataFrame({
        'cluster': [cluster_name],
        'distance': [None],  # Placeholder, will be updated
        'age': [None],
        'av': [None]
    })

    parallax = np.nanmedian(cluster_df["plx"])

    # sigma_parallax = np.std(cluster_df["plx"])
    dm = round(-5 * np.log10(parallax) + 10, 3)

    # Av
    if av_fixed:
        fix_params = {"alpha": 0.09, "beta": 0.94, "Rv": 3.1, "DR": 0., "met": 0.0152, "dm": dm, 'Av': cluster_av}
    else:
        fix_params = {"alpha": 0.09, "beta": 0.94, "Rv": 3.1, "DR": 0., "met": 0.0152, "dm": dm}

    synthcl.calibrate(my_cluster, fix_params)

    # Likelihood
    likelihood = asteca.likelihood(my_cluster, compute_l=l_adjust)

    # Define parameter priors
    loga_min, loga_max = 6, 8.5  # ~1Myr - 320Myr
    av_min, av_max = 0, 2

    priors = pyabc.Distribution(
        {
            "loga": pyabc.RV("uniform", loga_min, loga_max - loga_min),
            "Av": pyabc.RV("uniform", av_min, av_max - av_min),
        }
    )

    def model(fit_params):
        """Generate synthetic cluster. pyABC expects a dictionary from this
        function, so we return a dictionary with a single element.
        """
        synth_clust = synthcl.generate(fit_params)
        synth_dict = {"data": synth_clust}
        return synth_dict

    def distance(synth_dict, _):
        """The likelihood returned works as a distance which means that the optimal
        value is 0.0.
        """
        return likelihood.get(synth_dict["data"])

    # Run ABC-SMC
    pop_size = 100
    abc = pyabc.ABCSMC(
        model,
        priors,
        distance,
        population_size=pop_size
    )

    # This is a temporary file required by pyABC
    db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), f"pyABC_{cluster_name}.db")
    abc.new(db_path)

    history = abc.run(minimum_epsilon=0.01, max_nr_populations=25)

    # Extract results
    df_results, weights = history.get_distribution()
    fit_params = {k: pyabc.weighted_statistics.weighted_median(df_results[k].values, weights) for k in df_results.keys()}

    # Extract last iteration and weights
    df, w = history.get_distribution()

    label = ''
    add_breaker = '\n'
    for k in df.keys():
        _median = pyabc.weighted_statistics.weighted_median(df[k].values, w)
        _std = pyabc.weighted_statistics.weighted_std(df[k].values, w)

        if k == 'loga':
            _std = (10 ** _median * np.log(10) * _std) /10**6
            _median = (10**_median)/10**6

            add_breaker = ''
            k = 'Age (Myr)'
            cluster_results.loc[0, 'age'] = _median

        elif k == 'dm':
            _std = 10 ** (_median / 5) * np.log(10) * _std
            _median = 10**((_median+5)/5)

            k = 'Distance (pc)'
            cluster_results.loc[0, 'distance'] = _median

        else:
            cluster_results.loc[0, 'av'] = _median

        label += "{:<5}: {:.1f} +/- {:.1f} {:<5}".format(k, round(_median, 1), round(_std, 1), add_breaker)

    pyabc.settings.set_figure_params("pyabc")  # for beautified plots

    # Create a PdfPages object to save all plots to one PDF
    pdf_filename = f"{results_dir}/{cluster_name}.pdf"
    with PdfPages(pdf_filename) as pdf:

        # Matrix of 1d and 2d histograms over all parameters
        pyabc.visualization.plot_histogram_matrix(history)
        pdf.savefig()
        plt.close()

        pyabc.visualization.plot_credible_intervals(history)
        pdf.savefig()
        plt.close()

        # cluster
        iso_final = asteca.plot.get_isochrone(synthcl, fit_params)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # CMD plot
        asteca.plot.cluster(my_cluster, axes[0], col_plot="cmd")
        axes[0].plot(iso_final[1, :], iso_final[0, :], color='black')

        # CCD plot (only the cluster, no isochrone if color2 is None)
        asteca.plot.cluster(my_cluster, axes[1], col_plot="ccd")
        if synthcl.isochs.color2 is not None:
            axes[1].plot(iso_final[1, :], iso_final[2, :], color='black')

        plt.suptitle(label)
        pdf.savefig()
        plt.close()

        # Synthetic
        if synthcl.isochs.color2 is not None:
            iso_final = asteca.plot.get_isochrone(synthcl, fit_params)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            asteca.plot.synthetic(synthcl, axes[0], fit_params, iso_final, col_plot='cmd')
            asteca.plot.synthetic(synthcl, axes[1], fit_params, iso_final, col_plot='ccd')

            plt.suptitle(label)
            pdf.savefig()
            plt.close()
        else:
            # Only plot the CMD for synthetic if no color2 is defined
            iso_final = asteca.plot.get_isochrone(synthcl, fit_params)
            fig, ax = plt.subplots(figsize=(5, 5))
            asteca.plot.synthetic(synthcl, ax, fit_params, iso_final, col_plot='cmd')
            plt.title(label)
            pdf.savefig()
            plt.close()

    print(f"Finished {cluster_name}, all plots saved to {pdf_filename}.")
    return cluster_results


def main():
    if len(sys.argv) < 2:
        print("Usage: python age_inference.py <config_file.json>")
        sys.exit(1)

    # Load the config that the caller has already validated
    with open(sys.argv[1], "r") as f:
        cfg = json.load(f)

    cluster_name = cfg["cluster_name"]
    evolution_model = cfg["evolution_model"]
    l_adjust = cfg["l_adjust"]
    phot_system = cfg["phot_system"]
    av_fixed = cfg["av_fixed"]
    results_dir = cfg["results_dir"]

    # Pull av_value straight from JSON
    av_value = cfg.get("av_value") if av_fixed else None

    print(f"Processing {cluster_name} | av_fixed={av_fixed}, av_value={av_value}")

    # Load isochrones and set up synthetic cluster generator
    isochrones = load_isochrones(evolution_model, phot_system)
    synthcl    = setup_synthetic_clusters(isochrones)

    # Read and filter cluster data
    df = pd.read_csv("dias_dr3_0.5.csv")
    cluster_df = df[df["cluster"] == cluster_name]
    cluster_df = cluster_df[cluster_df["pmemb"] > 0.5]

    if phot_system == 'Gaia':
        cluster_df.dropna(subset=["Gmag"], inplace=True)
    else:
        cluster_df.dropna(subset=["Vmag"], inplace=True)

    if cluster_df.empty:
        print(f"Skipping {cluster_name}: no members with pmemb > 0.5")
        return

    # Build the asteca.cluster object
    if phot_system == "Gaia":
        my_cluster = asteca.cluster(
            obs_df=cluster_df,
            magnitude="Gmag", e_mag="e_Gmag",
            color="BP_RP",  e_color="e_BP_RP",
            color2="G_RP", e_color2="e_G_RP",
        )
    else:  # UBVRI
        my_cluster = asteca.cluster(
            obs_df=cluster_df,
            magnitude="Vmag", e_mag="e_Vmag",
            color="Vmag-Imag",  e_color="e_Vmag-Imag",
            color2="Umag-Bmag", e_color2="e_Umag-Bmag",
        )

    print(my_cluster)

    # Delegate to the refactored process_cluster
    results = process_cluster(
        cluster_name=cluster_name,
        cluster_df=cluster_df,
        synthcl=synthcl,
        my_cluster=my_cluster,
        l_adjust=l_adjust,
        av_fixed=av_fixed,
        cluster_av=av_value,
        results_dir=results_dir
    )

    # Save if any results returned
    if results is not None and not results.empty:
        out_path = f"./junk/results_{cluster_name}.csv"
        results.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()


