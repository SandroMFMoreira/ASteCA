import os
import sys
import json
import pandas as pd
import asteca
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import emcee
import corner  # pip install corner if you don't have it
from emcee import EnsembleSampler, moves

import numpy as np

from age_inference_caller import evolution_model

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
    emcee MCMC version (20 walkers x 100 steps).
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

    if evolution_model == 'baraffe':
        model = evolution_model
    else:
        model = None

    # Likelihood object (returns a distance; lower is better)
    likelihood = asteca.likelihood(my_cluster, compute_l=l_adjust, use_kde=True, model=model)

    # Convert 'distance' returned by likelihood object to log-likelihood.
    # Here we use loglike = -distance (unnormalized). Adjust mapping if desired.

    # MCMC parameters
    n_walkers = 50
    n_steps = 5000

    # Priors bounds
    loga_min, loga_max = 6.0, 8.5  # log10(age in yr)
    av_min, av_max = 0.0, 1.0

    # Set up initial walker positions random.uniformly in prior
    if vary_av:
        ndim = 2
        p0 = np.zeros((n_walkers, ndim))
        p0[:, 0] = np.random.uniform(loga_min, loga_max, size=n_walkers)
        p0[:, 1] = np.random.uniform(av_min, av_max, size=n_walkers)
        param_labels = ["loga", "Av"]
    else:
        ndim = 1
        p0 = np.zeros((n_walkers, ndim))
        p0[:, 0] = np.random.uniform(loga_min, loga_max, size=n_walkers)
        param_labels = ["loga"]

    procs = 32 # min(n_walkers, n_cpus)

    ctx = mp.get_context("fork")
    # Use a pool for parallel likelihood evaluation
    with ctx.Pool(processes=procs, initializer=init_worker,
                  initargs=(synthcl, fix_params, vary_av, likelihood)) as pool:

        sampler = EnsembleSampler(
            n_walkers,
            ndim,
            log_probability_worker,
            pool=pool
        )

        start_time = time.time()
        sampler.run_mcmc(p0, n_steps, progress=True)
        elapsed = time.time() - start_time

    # --- For plotting ---
    burnin = max(1500, int(0.2 * n_steps))
    flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=10)

    # After run_mcmc
    try:
        # Estimate IAT on the post burn-in portion for stationarity
        tau = emcee.autocorr.integrated_time(
            sampler.get_chain(discard=burnin), quiet=True
        )
        print("IAT (post burn-in):", tau)
    except emcee.autocorr.AutocorrError as e:
        print("IAT unstable post burn-in:", e)
        # Extend the run if needed
        # sampler.run_mcmc(None, extra_steps, progress=True)

    # Fallback if flattened sample ended up empty
    if flat_samples.size == 0:
        flat_samples = sampler.get_chain(flat=True)

    # Try to recover log-probabilities for plotting
    try:
        flat_logprob = sampler.get_log_prob(discard=burnin, flat=True)
    except Exception:
        try:
            flat_logprob = sampler.get_log_prob(flat=True)
        except Exception:
            flat_logprob = getattr(sampler, "lnprobability", None)
            if flat_logprob is not None:
                flat_logprob = flat_logprob.reshape(-1)

    # --- For saving (full chain, no discard/thin) ---
    full_samples = sampler.get_chain(flat=True)
    try:
        full_logprob = sampler.get_log_prob(flat=True)
    except Exception:
        full_logprob = getattr(sampler, "lnprobability", None)
        if full_logprob is not None:
            full_logprob = full_logprob.reshape(-1)

    if full_logprob is not None and full_logprob.size == len(full_samples):
        chain_with_logp = np.column_stack((full_samples, full_logprob))
        header = " ".join([f"param_{i}" for i in range(full_samples.shape[1])] + ["log_prob"])
    else:
        chain_with_logp = full_samples
        header = " ".join([f"param_{i}" for i in range(full_samples.shape[1])])

    np.savetxt("chain.txt", chain_with_logp, header=header)

    # If we have log-probs, pick the sample with maximum log-probability (best-fit)
    if flat_logprob is not None and np.size(flat_logprob) == flat_samples.shape[0]:
        best_idx = np.argmax(flat_logprob)
        best_params = flat_samples[best_idx]
    else:
        # fallback: use the median sample if log-probs are unavailable
        best_params = np.median(flat_samples, axis=0)

    # Compute stds in parameter space (still useful for uncertainty estimates)
    std = np.std(flat_samples, axis=0)

    # Use best-likelihood sample for best_loga and best_Av
    best_loga = best_params[0]

    if vary_av:
        best_Av = best_params[1]
    else:
        best_Av = np.round(fix_params.get('Av', np.nan), 2) if fix_params.get('Av', None) is not None else np.nan


    # --- Map sampled params to physical outputs (use median-based best_params for the "best synth")
    age_myr = (10 ** best_loga) / 1e6

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

        synthcl.calibrate(my_cluster, fix_params, n_points=200)
        synth_best = synthcl.generate(fit_best)
        best_dist, contamination_mask = likelihood.get(synth_best, rmv_contam=True)
        cluster_results.loc[0, 'lk_dist'] = best_dist
    except Exception:
        cluster_results.loc[0, 'lk_dist'] = np.nan

    # # Uncertainties for each fundamental parameter
    # model_std = {"met": 0.001, "loga": age_std_myr, "dm": 0.001, "Av": std_Av}
    #
    # # Keep only the keys from fix_params that are also in model_std
    # model = {k: fix_params[k] for k in fix_params if k in model_std}
    #
    # model['loga'] = median_loga
    # model['Av'] = median_Av

    # # Call the method
    # synthcl.get_models(model, model_std)
    #
    # masses_dict = synthcl.cluster_masses()
    #
    # # Print the median mass values and their STDDEVs
    # for k, arr in masses_dict.items():
    #     print("{:<8}: {:.0f}+/-{:.0f}".format(k, np.median(arr), np.std(arr)))

    # Fill distance (we used fixed dm). Convert dm -> pc

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
                plt.suptitle(f"{cluster_name} — Observed cluster\nAge(Myr)={cluster_results.loc[0,'age']:.2f}, Av={cluster_results.loc[0,'av']}, lk_dist={cluster_results.loc[0,'lk_dist']:.2f}" )
                pdf.savefig()
                plt.close()

                # Synthetic
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                asteca.plot.synthetic(synthcl, axes[0], fit_params_for_iso, iso_final, col_plot='cmd')
                asteca.plot.synthetic(synthcl, axes[1], fit_params_for_iso, iso_final, col_plot='ccd')
                plt.suptitle(f"{cluster_name} — Synthetic (posterior median)")
                pdf.savefig()
                plt.close()
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
        except Exception as e:
            # If plotting or isochrone retrieval fails, skip
            print(e)

    print(f"Finished {cluster_name}: PDF diagnostics to {pdf_filename} (chain not saved to disk).")

    # Return only the DataFrame (keeps compatibility with your main())
    return cluster_results

# Define log-prior (uniform within bounds)
def log_prior(theta, vary_av, loga_min=6.0, loga_max=10.0, av_min=0, av_max=3):
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


def init_worker(synthcl_, fix_params_, vary_av_, likelihood_):
    global WORK_SYNTHCL, WORK_FIX_PARAMS, WORK_VARY_AV, WORK_LIKELIHOOD
    WORK_SYNTHCL = synthcl_
    WORK_FIX_PARAMS = fix_params_
    WORK_VARY_AV = vary_av_
    WORK_LIKELIHOOD = likelihood_


def log_probability_worker(theta):
    seed = np.random.randint(0, 2 ** 32 - 1)
    np.random.seed(seed)

    # call the existing module-level log_probability but using worker globals
    try:
        # reuse the same function body but refer to the globals
        lp = log_prior(theta, WORK_VARY_AV)
        if not np.isfinite(lp):
            return -np.inf

        fit_params = {}
        if WORK_VARY_AV:
            loga, Av = theta
            fit_params["loga"] = loga
            fit_params["Av"] = Av
        else:
            (loga,) = theta
            fit_params["loga"] = loga

        fit_params_full = dict(WORK_FIX_PARAMS)
        fit_params_full.update(fit_params)

        WORK_SYNTHCL.rng = np.random.default_rng(seed)

        start_time = time.time()
        synth_sample = WORK_SYNTHCL.generate(fit_params_full)
        # print('generate', time.time() - start_time)
        dist = WORK_LIKELIHOOD.get(synth_sample)
        # print('dist', time.time() - start_time)
    except Exception:
        return -np.inf
    return lp + dist

import hashlib

def theta_seed(theta, seed_base=12345):
    """
    Deterministic 32-bit-ish seed derived from theta array + seed_base.
    """
    s = ",".join([f"{float(x):.12g}" for x in np.atleast_1d(theta)])
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h[:8], 16) ^ int(seed_base & 0xFFFFFFFF)

def evaluate_one(theta):
    """
    Evaluate log-likelihood-like score for a single theta.
    Uses WORK_SYNTHCL, WORK_FIX_PARAMS, WORK_VARY_AV, WORK_LIKELIHOOD set by init_worker.
    Returns a scalar float (higher = better). Returns -np.inf on failure.
    This must be module-level so it can be pickled by multiprocessing.
    """
    try:
        # Validate theta numeric and finite
        theta = np.asarray(theta, dtype=float)
        if not np.all(np.isfinite(theta)):
            return -np.inf

        # Deterministic seed per theta -> reproducible synth generation
        seed = theta_seed(theta, seed_base=12345)
        WORK_SYNTHCL.rng = np.random.default_rng(seed)

        # Build dict with fixed params + theta values
        # param order expected: ["loga", "Av"] or ["loga"]
        fit = dict(WORK_FIX_PARAMS)
        if WORK_VARY_AV:
            # theta length must be 2
            if theta.size != 2:
                return -np.inf
            fit["loga"] = float(theta[0])
            fit["Av"] = float(theta[1])
        else:
            # theta length must be 1
            if theta.size != 1:
                return -np.inf
            fit["loga"] = float(theta[0])

        if float(theta[0]) < 1000:
            penalty = False
        else:
            penalty = True

        # Optionally allow averaging inside WORK_SYNTHCL.rng usage if external logic sets n_avg
        # We'll expect the calling code to handle averaging by re-setting WORK_SYNTHCL.rng appropriately.
        synth = WORK_SYNTHCL.generate(fit)
        # WORK_LIKELIHOOD.get returns a "distance" (lower is better)
        d = WORK_LIKELIHOOD.get(synth, penalty=penalty)
        # Convert to log-like (higher better)
        return float(d)
    except Exception:
        return -np.inf

def sample_multivariate_within_bounds(rng, mean, cov, n_samples, lower, upper, cov_shrink_limit=1e-6):
    """
    Draw up to n_samples from multivariate normal(mean, cov) but only accept samples inside [lower, upper].
    If proposals consistently fall outside bounds, shrink cov until some accepted or fallback to uniform.
    Returns array shape (n_samples, ndim).
    """
    ndim = mean.size
    samples = np.empty((n_samples, ndim))
    drawn = 0
    cov_matrix = np.array(cov, copy=True)
    tries = 0
    while drawn < n_samples:
        try:
            needed = n_samples - drawn
            draws = rng.multivariate_normal(mean, cov_matrix, size=needed)
        except Exception:
            # fallback to diagonal proposal
            draws = mean + rng.normal(scale=np.sqrt(np.diag(cov_matrix) + 1e-12), size=(needed, ndim))

        # accept those within bounds
        accept = np.all((draws >= lower) & (draws <= upper), axis=1)
        n_accept = np.sum(accept)
        if n_accept > 0:
            take = min(n_accept, n_samples - drawn)
            samples[drawn:drawn+take, :] = draws[accept][:take]
            drawn += take

        # if none accepted, shrink covariance and retry
        if n_accept == 0:
            cov_matrix = cov_matrix * 0.5
            tries += 1
            if np.all(np.diag(cov_matrix) < cov_shrink_limit) or tries > 20:
                # fallback: fill remaining with uniform draws
                for j in range(drawn, n_samples):
                    samples[j, :] = rng.uniform(lower, upper)
                drawn = n_samples
                break
    return samples

# --- The main CEM function (module-level) ---
def process_cluster_cem(cluster_name, cluster_df, synthcl, my_cluster,
                        l_adjust, av_fixed, cluster_av, results_dir,
                        pop_size=200, elite_frac=0.25, n_iter=50,
                        cov_reg=1e-4, n_avg=2, seed_base=12345):
    """
    Cross-Entropy Method version of process_cluster.
    Returns cluster_results DataFrame (same layout as your emcee version).
    All functions used by multiprocessing are module-level to be picklable.
    """

    # --- local imports used in plotting etc. (ok at function scope) ---
    from matplotlib.backends.backend_pdf import PdfPages

    # --- bookkeeping and results DF ----
    cluster_results = pd.DataFrame({
        'cluster': [cluster_name],
        'distance': [None],
        'age': [None],
        'av': [None],
        'lk_dist': [None]
    })

    # distance modulus from median parallax (same as before)
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

    # Respect your evolution_model global if present
    model = None
    try:
        if evolution_model == 'baraffe':
            model = evolution_model
    except Exception:
        model = None

    likelihood = asteca.likelihood(my_cluster, compute_l=l_adjust, use_kde=True, model=model)

    # Priors bounds (same as before)
    loga_min, loga_max = 6.0, 10.0
    av_min, av_max = 0.0, 3.0

    # Dimensionality and bounds arrays
    if vary_av:
        ndim = 2
        param_labels = ["loga", "Av"]
        lower = np.array([loga_min, av_min])
        upper = np.array([loga_max, av_max])
    else:
        ndim = 1
        param_labels = ["loga"]
        lower = np.array([loga_min])
        upper = np.array([loga_max])

    # initial population (uniform in prior)
    rng = np.random.default_rng(seed_base)
    pop = np.empty((pop_size, ndim))
    for i in range(ndim):
        pop[:, i] = rng.uniform(lower[i], upper[i], size=pop_size)

    # history containers
    means_hist = []
    best_hist = []
    best_params_hist = []
    pop_history_final = None

    # multiprocessing pool using your init_worker initializer
    procs = min(pop_size, max(1, mp.cpu_count() - 1))
    ctx = mp.get_context("fork")
    init_args = (synthcl, fix_params, vary_av, likelihood)

    start_time = time.time()
    with ctx.Pool(processes=procs, initializer=init_worker, initargs=init_args) as pool:
        for it in range(n_iter):
            # Evaluate population in parallel: pool.map expects sequence of picklable objects (numpy arrays are OK)
            thetas = [pop[i, :].astype(float) for i in range(pop.shape[0])]
            loglikes = pool.map(evaluate_one, thetas)  # evaluate_one is top-level

            # convert to numpy array
            loglikes = np.array(loglikes, dtype=float)

            # if all invalid, stop
            if not np.any(np.isfinite(loglikes)):
                print(f"[CEM] iter {it+1}: all evaluations invalid. Stopping early.")
                break

            # rank and select elites
            n_elite = max(2, int(np.ceil(elite_frac * pop_size)))
            ranks = np.argsort(loglikes)[::-1]
            elites_idx = ranks[:n_elite]
            elites = pop[elites_idx, :]

            # compute mean and covariance
            mean_elite = np.mean(elites, axis=0)
            cov_elite = np.cov(elites, rowvar=False)
            if ndim == 1:
                cov_matrix = np.atleast_2d(cov_elite)
            else:
                cov_matrix = cov_elite

            # regularize
            cov_matrix = cov_matrix + cov_reg * np.eye(ndim)

            # save history
            means_hist.append(mean_elite.copy())
            best_hist.append(loglikes[ranks[0]])
            best_params_hist.append(pop[ranks[0], :].copy())

            # propose new population by sampling multivariate normal within bounds
            rng_local = rng  # use same RNG object
            new_pop = sample_multivariate_within_bounds(rng_local, mean_elite, cov_matrix, pop_size, lower, upper)
            pop = new_pop

            elapsed = time.time() - start_time
            # (f"[CEM] iter {it+1}/{n_iter}  best_loglike={best_hist[-1]:.3f}  mean_params={mean_elite}  elapsed={elapsed:.1f}s")

        pop_history_final = pop.copy()

    total_elapsed = time.time() - start_time

    # final evaluation of the final population (single-threaded)
    final_scores = np.array([evaluate_one((pop_history_final[i, :], n_avg)) for i in range(pop_history_final.shape[0])])
    if np.any(np.isfinite(final_scores)):
        final_best_idx = int(np.nanargmax(final_scores))
        best_params = pop_history_final[final_best_idx, :]
    else:
        best_params = means_hist[-1] if len(means_hist) > 0 else pop_history_final[0, :]

    # uncertainties from final elites
    idxs_sorted_final = np.argsort(final_scores)[::-1]
    n_elite_final = max(2, int(np.ceil(elite_frac * pop_size)))
    elites_final = pop_history_final[idxs_sorted_final[:n_elite_final], :]
    final_mean = np.mean(elites_final, axis=0)
    final_cov = np.cov(elites_final, rowvar=False)
    final_cov = np.atleast_2d(final_cov)
    param_std = np.sqrt(np.abs(np.diag(final_cov)))

    # map outputs (age, Av)
    best_loga = float(best_params[0])
    if vary_av:
        best_Av = float(best_params[1])
    else:
        best_Av = np.round(fix_params.get('Av', np.nan), 2) if fix_params.get('Av', None) is not None else np.nan

    age_myr = (10 ** best_loga) / 1e6
    cluster_results.loc[0, 'age'] = np.round(age_myr, 3)
    cluster_results.loc[0, 'av'] = (np.round(best_Av, 2) if (best_Av is not None and not np.isnan(best_Av)) else np.nan)

    # Fill distance (we used fixed dm). Convert dm -> pc
    try:
        if 'dm' in fix_params and fix_params['dm'] is not None:
            distance_pc = 10 ** ((fix_params['dm'] + 5) / 5)
            cluster_results.loc[0, 'distance'] = distance_pc
    except Exception:
        cluster_results.loc[0, 'distance'] = np.nan

    # Evaluate lk_dist at best-fit params
    try:
        fit_best = dict(fix_params)
        fit_best["loga"] = best_loga
        if vary_av:
            fit_best["Av"] = best_Av

        synthcl.calibrate(my_cluster, fix_params, n_points = 300)
        synth_best = synthcl.generate(fit_best)
        best_dist, contamination_mask = likelihood.get(synth_best, rmv_contam=True)
        cluster_results.loc[0, 'lk_dist'] = best_dist
    except Exception:
        cluster_results.loc[0, 'lk_dist'] = np.nan

    # ---- Produce PDF diagnostics ----
    os.makedirs(results_dir, exist_ok=True)
    pdf_filename = os.path.join(results_dir, f"{cluster_name}.pdf")
    with PdfPages(pdf_filename) as pdf:
        # trace of parameter means and best score per iteration
        if len(means_hist) > 0:
            means_hist_arr = np.array(means_hist)
            best_hist_arr = np.array(best_hist)
            fig, axes = plt.subplots(ndim + 1, 1, figsize=(8, 2.5 * (ndim + 1)))
            for i in range(ndim):
                axes[i].plot(means_hist_arr[:, i], marker='o', lw=1, label='mean_elite')
                axes[i].set_ylabel(param_labels[i])
                axes[i].axhline(best_params[i], color='k', ls='--', lw=0.8, label='final_best')
                axes[i].legend()
            axes[-1].plot(best_hist_arr, marker='o', lw=1)
            axes[-1].set_ylabel("best_score")
            axes[-1].set_xlabel("CEM iteration")
            plt.suptitle(f"{cluster_name} — CEM diagnostics (elapsed {total_elapsed:.1f}s)")
            pdf.savefig()
            plt.close()

        # corner of final population
        try:
            fig_corner = corner.corner(pop_history_final, labels=param_labels, show_titles=True,
                                       title_fmt=".3f", quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 10})
            plt.suptitle(f"{cluster_name} — Final population (CEM)", y=1.02)
            pdf.savefig(bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        # cluster + synthetic plots
        try:
            fit_params_for_iso = dict(fix_params)
            fit_params_for_iso["loga"] = best_loga
            if vary_av:
                fit_params_for_iso["Av"] = best_Av
            iso_final = asteca.plot.get_isochrone(synthcl, fit_params_for_iso)

            if synthcl.isochs.color2 is not None:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                asteca.plot.cluster(my_cluster, axes[0], col_plot="cmd", contamination_mask=contamination_mask)
                axes[0].plot(iso_final[1, :], iso_final[0, :], color='black')
                asteca.plot.cluster(my_cluster, axes[1], col_plot="ccd")
                axes[1].plot(iso_final[1, :], iso_final[2, :], color='black')
                plt.suptitle(
                    f"{cluster_name} — Observed cluster\nAge(Myr)={cluster_results.loc[0, 'age']:.2f}, Av={cluster_results.loc[0, 'av']}, lk_dist={cluster_results.loc[0, 'lk_dist']:.2f}")
                pdf.savefig()
                plt.close()

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                asteca.plot.synthetic(synthcl, axes[0], fit_params_for_iso, iso_final, col_plot='cmd')
                asteca.plot.synthetic(synthcl, axes[1], fit_params_for_iso, iso_final, col_plot='ccd')
                plt.suptitle(f"{cluster_name} — Synthetic (CEM best)")
                pdf.savefig()
                plt.close()
            else:
                fig, ax = plt.subplots(figsize=(6, 6))
                asteca.plot.cluster(my_cluster, ax, col_plot="cmd")
                ax.plot(iso_final[1, :], iso_final[0, :], color='black')
                plt.title(f"{cluster_name} — Observed CMD")
                pdf.savefig()
                plt.close()

                fig, ax = plt.subplots(figsize=(6, 6))
                asteca.plot.synthetic(synthcl, ax, fit_params_for_iso, iso_final, col_plot='cmd')
                plt.title(f"{cluster_name} — Synthetic CMD (CEM best)")
                pdf.savefig()
                plt.close()
        except Exception as e:
            print("CEM plotting error:", e)

    # Print best & uncertainties
    std_report = param_std if param_std is not None else np.full(ndim, np.nan)
    print(f"[CEM] best_params = {best_params}")
    print(f"[CEM] param stds  = {std_report}")

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

    if phot_system == 'Gaia':
        ext_law = 'GAIADR3'
    else:
        ext_law = 'CCMO'

    # Load isochrones and set up synthetic cluster generator
    isochrones = load_isochrones(evolution_model, phot_system)
    synthcl    = setup_synthetic_clusters(isochrones, ext_law=ext_law)

    # Read and filter cluster data
    df = pd.read_csv("dias_dr3_0.5.csv")
    # df = pd.read_csv("/home/sandro/PycharmProjects/OCs-ScaleHeight-Evolution_rep/LargeFiles/Hunt/members_Hunt_dr3_50.csv")
    cluster_df = df[df["cluster"] == cluster_name]
    cluster_df = cluster_df[cluster_df["pmemb"] > 0.5]

    # base_path = f"/home/sandro/PycharmProjects/pyUPMASK/output_dr3_enriched/{cluster_name}"

    # # Try astrometry_3_15 first, then fallback to astrometry_3_10
    # file_15 = os.path.join(base_path, "astrometry_3_15_dr3_enriched.csv")
    # file_10 = os.path.join(base_path, "astrometry_3_10_dr3_enriched.csv")
#
    # if os.path.exists(file_15):
    #     cluster_df = pd.read_csv(file_15)
    # elif os.path.exists(file_10):
    #     cluster_df = pd.read_csv(file_10)
    # else:
    #     print(f"No astrometry file available for cluster {cluster_name}. Skipping.")
    #     cluster_df = None

    # # If no file was loaded, stop here
    # if cluster_df is not None:
#
    #     # Apply membership probability cut
    #     cluster_df = cluster_df[cluster_df["probs_final"] > 0.5]
#
    #     # Ignore cluster if too many members
    #     if len(cluster_df) > 3000:
    #         print(f"Cluster {cluster_name} has {len(cluster_df)} members > 3000. Skipping.")
    #         cluster_df = None
#
    # cluster_df.rename(columns={
    #     "parallax": "plx",
    #     "phot_g_mean_mag": "Gmag",
    #     "phot_bp_mean_mag": "BPmag",
    #     "phot_rp_mean_mag": "RPmag",
    #     "bp_rp": "BP_RP",
    #     "g_rp": "G_RP",
    #     "phot_g_mean_mag_error": "e_Gmag",
    #     "bp_rp_error": "e_BP_RP",
    #     "g_rp_error": "e_G_RP"
    # }, inplace=True)
#
    # cluster_df['e_Vmag'] = 0.001* cluster_df['Vmag']
    # cluster_df['Vmag-Imag'] = cluster_df['Vmag'] - cluster_df['Imag']
    # cluster_df['Umag-Bmag'] = cluster_df['Umag'] - cluster_df['Bmag']
    # cluster_df['e_Vmag-Imag'] = 0.001* np.abs((cluster_df['Vmag'] - cluster_df['Imag']))
    # cluster_df['e_Umag-Bmag'] = 0.001* np.abs((cluster_df['Umag'] - cluster_df['Bmag']))

    if phot_system == 'Gaia':
        cluster_df.dropna(subset=["Gmag"], inplace=True)
    else:
        cluster_df.dropna(subset=["Vmag"], inplace=True)

    if cluster_df.empty:
        print(f"Skipping {cluster_name}: no members with pmemb > 0.5")
        return

    cluster_df = cluster_df[cluster_df.Gmag <= 17]

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
    results = process_cluster_cem(
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
        out_dir = f"./{results_dir}/junk"
        os.makedirs(out_dir, exist_ok=True)  # <-- create directory if missing

        out_path = f"{out_dir}/results_{cluster_name}.csv"
        results.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

import multiprocessing as mp

if __name__ == "__main__":
    main()


