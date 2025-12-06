import sys
import json
import numpy as np
import pandas as pd
import asteca
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import emcee
import corner  # pip install corner if you don't have it

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
    if (evolution_model == 'baraffe') or (evolution_model == 'mixed'):
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
            if evolution_model == 'baraffe':
                iso_path = "./Isochrones/BHAC15_iso_UBVRI.txt"
                color2 = None
                color_effl2 = None
            else:
                iso_path = "./Isochrones/mixed_UBVRI.txt"
                color2 = ("U", "B")
                color_effl2 = (
                    eff_lambda_df["U"].values,
                    eff_lambda_df["B"].values
                )
            magnitude = "Mv"
            color1 = ("Mv", "Mi")

            mag_effl = eff_lambda_df["V"].values
            color_effl1 = (
                eff_lambda_df['V'].values,
                eff_lambda_df['I'].values
            )


        else:
            raise ValueError("phot_system must be 'Gaia' or 'UBVRI'")

    elif evolution_model == 'parsec':
        if phot_system == 'Gaia':
            iso_path = "./Isochrones/combined_isochrones.dat"
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


def setup_synthetic_clusters(isochrones, ext_law):
    return asteca.synthetic(isochrones, seed=None, ext_law=ext_law)



def process_cluster(cluster_name, cluster_df, synthcl, my_cluster, l_adjust, av_fixed, cluster_av, results_dir):
    """
    Serial (non-parallel) emcee MCMC version.
    - Does NOT save the chain to disk.
    - Produces trace plots, corner plot, and cluster/synthetic plots saved into a PDF.
    - RETURNS only cluster_results (DataFrame) for compatibility with your existing code.
    """

    # Create DataFrame with one row
    cluster_results = pd.DataFrame({
        'cluster': [cluster_name],
        'distance': [None],  # Placeholder, will be updated
        'age': [None],
        'av': [None],
        'lk_dist': [None]
    })

    # Compute distance modulus from median parallax (as you had)
    parallax = np.nanmedian(cluster_df["plx"])
    dm = round(-5 * np.log10(parallax) + 10, 3)

    # fix_params and whether Av is free
    if av_fixed:
        fix_params = {"alpha": 0.09, "beta": 0.94, "Rv": 3.1, "DR": 0., "met": 0.0152, "dm": dm, "Av": cluster_av}
        vary_av = False
    else:
        fix_params = {"alpha": 0.09, "beta": 0.94, "Rv": 3.1, "DR": 0., "met": 0.0152, "dm": dm}
        vary_av = True

    # Calibrate synthetic cluster
    synthcl.calibrate(cluster=my_cluster, fix_params=fix_params)

    # Likelihood object (returns a distance; lower is better)
    likelihood = asteca.likelihood(my_cluster, compute_l=l_adjust, use_kde=True)

    # Convert 'distance' returned by likelihood object to log-likelihood.
    # Here we use loglike = -distance (unnormalized). Adjust mapping if desired.

    # MCMC parameters
    n_walkers = 100
    n_steps = 2500
    rng = np.random.default_rng(None)  # reproducible

    # Priors bounds
    loga_min, loga_max = 6.0, 10.0  # log10(age in yr)
    av_min, av_max = 0.0, 2.0

    # Set up initial walker positions uniformly in prior
    if vary_av:
        ndim = 2
        p0 = np.zeros((n_walkers, ndim))
        p0[:, 0] = rng.uniform(loga_min, loga_max, size=n_walkers)
        p0[:, 1] = rng.uniform(av_min, av_max, size=n_walkers)
        param_labels = ["loga", "Av"]
    else:
        ndim = 1
        p0 = np.zeros((n_walkers, ndim))
        p0[:, 0] = rng.uniform(loga_min, loga_max, size=n_walkers)
        param_labels = ["loga"]

    # set single-threaded BLAS/OMP (still OK to keep)
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # --- Serial log-probability closure that captures local objects ---
    def log_prob(theta):
        # log-prior
        if vary_av:
            if len(theta) != 2:
                return -np.inf
            loga, Av = theta
            if not (loga_min <= loga <= loga_max and av_min <= Av <= av_max):
                return -np.inf
        else:
            if len(theta) != 1:
                return -np.inf
            (loga,) = theta
            if not (loga_min <= loga <= loga_max):
                return -np.inf

        # Build fit params
        fit_params = {}
        if vary_av:
            loga, Av = theta
            fit_params["loga"] = loga
            fit_params["Av"] = Av
        else:
            (loga,) = theta
            fit_params["loga"] = loga

        # Full params for synth generation
        fit_params_full = dict(fix_params)
        fit_params_full.update(fit_params)

        try:
            synth_sample = synthcl.generate(fit_params_full)
            # asteca.likelihood.get returns a distance metric; lower is better
            dist = likelihood.get(synth_sample)
            # uniform prior => prior term is 0 (within bounds)
            lp = 0.0
            return lp - dist
        except Exception:
            return -np.inf

    # Create serial sampler (no pool)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)

    start_time = time.time()
    sampler.run_mcmc(p0, n_steps, progress=True)
    elapsed = time.time() - start_time

    # Post-processing: discard burn-in (15% of steps)
    burnin = max(1, int(0.15 * n_steps))
    flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=1)  # shape (n_samples, ndim)

    # Fallback if flattened sample ended up empty for some reason
    if flat_samples.size == 0:
        flat_samples = sampler.get_chain(flat=True)

    # Try to recover flattened log-probabilities for each sample (several emcee variants)
    flat_logprob = None
    try:
        # preferred API (emcee v3+)
        flat_logprob = sampler.get_log_prob(discard=burnin, flat=True)
    except Exception:
        try:
            # fallback: maybe flat only
            flat_logprob = sampler.get_log_prob(flat=True)
        except Exception:
            try:
                # some older code stores lnprobability attribute per chain
                flat_logprob = sampler.lnprobability.reshape(-1)
            except Exception:
                # If we can't get log-probs, fall back to median behavior
                flat_logprob = None

    # If we have log-probs, pick the sample with maximum log-probability (best-fit)
    if flat_logprob is not None and np.size(flat_logprob) == flat_samples.shape[0]:
        best_idx = np.argmax(flat_logprob)
        best_params = flat_samples[best_idx]
    else:
        # fallback: use the median sample if log-probs are unavailable
        best_params = np.median(flat_samples, axis=0)

    # Compute medians and stds in parameter space (stds used for uncertainty estimates)
    med = np.median(flat_samples, axis=0)
    std = np.std(flat_samples, axis=0)

    best_loga = med[0]
    std_loga = std[0]

    if vary_av:
        best_Av = med[1]
        std_Av = std[1]

        # Set best_params array in the same ordering as samples for downstream code
        best_params = np.array([best_loga, best_Av] + [0] * (flat_samples.shape[1] - 2))
    else:
        # Only age was varied; use fixed Av from fix_params (rounded for reporting)
        best_Av = np.round(fix_params.get('Av', np.nan), 2) if fix_params.get('Av', None) is not None else np.nan

        best_params = np.array([best_loga] + [0] * (flat_samples.shape[1] - 1))

    # --- Map sampled params to physical outputs (use median-based best_params for the "best synth")
    age_myr = (10 ** best_loga) / 1e6
    age_std_myr = (10 ** best_loga * np.log(10) * std_loga) / 1e6

    # Store results rounded as requested
    cluster_results.loc[0, 'age'] = np.round(age_myr, 3)
    cluster_results.loc[0, 'av'] = (np.round(best_Av, 2) if (best_Av is not None and not np.isnan(best_Av)) else np.nan)

    # Fill distance (we used fixed dm). Convert dm -> pc
    try:
        if 'dm' in fix_params and fix_params['dm'] is not None:
            distance_pc = 10 ** ((fix_params['dm'] + 5) / 5)
            cluster_results.loc[0, 'distance'] = distance_pc
    except Exception:
        cluster_results.loc[0, 'distance'] = np.nan

    # Compute lk_dist (distance metric) at best-fit (median) parameters
    try:
        fit_best = dict(fix_params)
        fit_best["loga"] = best_loga
        if vary_av:
            fit_best["Av"] = best_Av

        synthcl.calibrate(my_cluster, fit_best, n_points=10_000)
        synth_best = synthcl.generate(fit_best)
        best_dist, contamination_mask = likelihood.get(synth_best, rmv_contam=True)
        cluster_results.loc[0, 'lk_dist'] = best_dist
    except Exception:
        cluster_results.loc[0, 'lk_dist'] = np.nan

    # Create PDF with diagnostics including corner plot
    os.makedirs(results_dir, exist_ok=True)
    pdf_filename = os.path.join(results_dir, f"{cluster_name}.pdf")
    with PdfPages(pdf_filename) as pdf:
        # 1) Trace plots for each parameter (walkers over steps)
        # sampler.get_chain() -> shape (n_steps, n_walkers, ndim)
        chain = sampler.get_chain()
        # ensure chain shape is (n_walkers, n_steps, ndim) for plotting ease
        if chain.shape[0] == n_steps and chain.shape[1] == n_walkers:
            chain_swapped = np.swapaxes(chain, 0, 1)  # -> (n_walkers, n_steps, ndim)
        else:
            chain_swapped = chain  # assume already (n_walkers, n_steps, ndim)

        fig, axes = plt.subplots(ndim, 1, figsize=(9, 2.5 * ndim), squeeze=False)
        axes = axes.flatten()
        for i in range(ndim):
            for w in range(chain_swapped.shape[0]):
                axes[i].plot(chain_swapped[w, :, i], alpha=0.6, lw=0.8)
            axes[i].set_ylabel(param_labels[i])
            axes[i].axvline(burnin, color='k', linestyle='--', lw=0.8)
        axes[-1].set_xlabel("step")
        plt.suptitle(f"{cluster_name} — MCMC traces (elapsed {elapsed:.1f}s)")
        pdf.savefig()
        plt.close()

        # 2) Corner plot of the posterior (from flat_samples)
        try:
            fig_corner = corner.corner(flat_samples, labels=param_labels, show_titles=True,
                                       title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 10})
            plt.suptitle(f"{cluster_name} — Corner plot (posterior)", y=1.02)
            pdf.savefig(bbox_inches="tight")
            plt.close()
        except Exception:
            # fallback plotting if corner fails
            if ndim == 1:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                ax.hist(flat_samples[:, 0], bins=40)
                ax.set_xlabel(param_labels[0])
                ax.set_ylabel("counts")
                plt.suptitle(f"{cluster_name} — Posterior (fallback)")
                pdf.savefig()
                plt.close()
            else:
                fig, axes = plt.subplots(ndim, ndim, figsize=(3 * ndim, 3 * ndim))
                for i in range(ndim):
                    for j in range(ndim):
                        ax = axes[i, j]
                        if i == j:
                            ax.hist(flat_samples[:, i], bins=40)
                        else:
                            ax.scatter(flat_samples[:, j], flat_samples[:, i], s=1, alpha=0.1)
                plt.suptitle(f"{cluster_name} — Joint posterior samples (fallback)")
                pdf.savefig()
                plt.close()

        # 3) Cluster and synthetic plots similar to your original function
        try:
            fit_params_for_iso = dict(fix_params)
            fit_params_for_iso["loga"] = best_loga
            if vary_av:
                fit_params_for_iso["Av"] = best_Av
            iso_final = asteca.plot.get_isochrone(synthcl, fit_params_for_iso)

            # CMD and CCD if available
            if synthcl.isochs.color2 is not None:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                asteca.plot.cluster(my_cluster, axes[0], col_plot="cmd", contamination_mask=contamination_mask)
                axes[0].plot(iso_final[1, :], iso_final[0, :], color='black')
                asteca.plot.cluster(my_cluster, axes[1], col_plot="ccd")
                axes[1].plot(iso_final[1, :], iso_final[2, :], color='black')
                plt.suptitle(f"{cluster_name} — Observed cluster\nAge(Myr)={cluster_results.loc[0,'age']:.2f}, Av={cluster_results.loc[0,'av']}")
                pdf.savefig()
                plt.close()

                # Synthetic
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                asteca.plot.synthetic(synthcl, axes[0], fit_params_for_iso, iso_final, col_plot='cmd')
                asteca.plot.synthetic(synthcl, axes[1], fit_params_for_iso, iso_final, col_plot='ccd')
                plt.suptitle(f"{cluster_name} — Synthetic (posterior median)")
                pdf.savefig()
                plt.close()

                print('ola')
            else:
                # Only CMD
                fig, ax = plt.subplots(figsize=(6, 6))
                asteca.plot.cluster(my_cluster, ax, col_plot="cmd")
                ax.plot(iso_final[1, :], iso_final[0, :], color='black')
                plt.title(f"{cluster_name} — Observed CMD")
                pdf.savefig()
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 6))
                asteca.plot.synthetic(synthcl, ax, fit_params_for_iso, iso_final, col_plot='cmd')
                plt.title(f"{cluster_name} — Synthetic CMD (posterior median)")
                pdf.savefig()
                plt.close()
        except Exception:
            # If plotting or isochrone retrieval fails, skip
            pass

    print(f"Finished {cluster_name}: PDF diagnostics to {pdf_filename} (chain not saved to disk).")

    # Return only the DataFrame (keeps compatibility with your main())
    return cluster_results

# Define log-prior (uniform within bounds)
def log_prior(theta, vary_av, loga_min=6.0, loga_max=10.0, av_min=0, av_max=2):
    if vary_av:
        loga, Av = theta
        if (loga_min <= loga <= loga_max) and (av_min <= Av <= av_max):
            return 0.0
        return -np.inf
    else:
        (loga,) = theta
        if loga_min <= loga <= loga_max:
            return 0.0
        return -np.inf


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

    if phot_system == 'Gaia':
        ext_law = 'GAIADR3'
    else:
        ext_law = 'CCMO'

    # Load isochrones and set up synthetic cluster generator
    isochrones = load_isochrones(evolution_model, phot_system)
    synthcl    = setup_synthetic_clusters(isochrones, ext_law=ext_law)

    # Read and filter cluster data
    df = pd.read_csv("dias_dr3_0.5.csv")
    #df = pd.read_csv("/home/sandro/PycharmProjects/OCs-ScaleHeight-Evolution_rep/LargeFiles/Hunt/members_Hunt_dr3_50.csv")
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
            ra="ra",
            dec="dec",
            obs_df=cluster_df,
            magnitude="Gmag", e_mag="e_Gmag",
            color="BP_RP",  e_color="e_BP_RP",
            color2="G_RP", e_color2="e_G_RP",
        )
    else:  # UBVRI
        my_cluster = asteca.cluster(
            ra="ra",
            dec="dec",
            obs_df=cluster_df,
            magnitude="Vmag", e_mag="e_Vmag",
            color="Vmag-Imag",  e_color="e_Vmag-Imag",
            color2=None if evolution_model == 'baraffe' else "Umag-Bmag",
            e_color2=None if evolution_model == 'baraffe' else "e_Umag-Bmag",
        )

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
