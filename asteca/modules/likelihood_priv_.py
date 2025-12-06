from astropy.stats import calculate_bin_edges
from fast_histogram import histogram2d
from scipy.special import loggamma
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Sequence, Dict, Any, Union
from sklearn.neighbors import NearestNeighbors
import numpy as np
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree as CKDTree


def nn_loglik(obs_pts, obs_errs, syn_pts, weights=None, k=5, eps=1e-5, return_per_point=False):
    """
    Nearest-neighbour based log-likelihood using scipy.spatial.cKDTree.
    Returns either the per-point vector or the summed scalar.
    Enforces a minimum distance threshold (default=0.01).
    """
    obs_pts = np.asarray(obs_pts, dtype=float)
    obs_errs = np.asarray(obs_errs, dtype=float)
    syn_pts = np.asarray(syn_pts, dtype=float)

    if obs_pts.ndim != 2 or syn_pts.ndim != 2:
        raise ValueError("obs_pts and syn_pts must be 2D arrays (N, ndim)")
    if obs_pts.shape[1] != syn_pts.shape[1]:
        raise ValueError("obs_pts and syn_pts must have same number of dimensions")

    tree = CKDTree(syn_pts)
    dists, idx = tree.query(obs_pts, k=k)

    # Ensure shapes (N_obs, k)
    if k == 1:
        dists = dists.reshape(-1, 1)
        idx = idx.reshape(-1, 1)
    else:
        dists = np.asarray(dists)
        idx = np.asarray(idx)

    dists = np.maximum(dists, 1e-2)

    avg_dists = dists.mean(axis=1)     # (N_obs,)
    nn_idx = idx[:, 0]                 # (N_obs,)
    sigma = np.linalg.norm(obs_errs, axis=1)

    logL_i = -0.5 * (avg_dists / sigma) ** 2 - 0.5 * np.log(2 * np.pi * sigma ** 2)

    if weights is not None:
        w = np.asarray(weights)
        if w.shape[0] != syn_pts.shape[0]:
            raise ValueError("weights must have same length as syn_pts")
        w_matched = w[nn_idx] + eps
        logL_i += np.log(w_matched)

    return logL_i if return_per_point else float(np.sum(logL_i))


def knn_likelihood(
    obs_data,
    synth_clust,
    compute_l="cmd",
    is_imf_weighted=False,
    rmv_contam=False,
    plot=False,
    k=5,
    penalty=True,           # toggle bluest-point penalty
    penalty_halfwidth=0.3   # half-width of color interval around bluest point
):
    """
    Compute KNN-based log-likelihood in CMD and/or CCD, return total and per-point vectors.
    If plot=True, show side-by-side CMD and CCD with observed points color-coded by per-point likelihood.

    Returns:
        If rmv_contam is False:
            (-total_logL, logL_vec_cmd, logL_vec_ccd, masks)
        If rmv_contam is True:
            (-total_logL, contamination_mask, logL_vec_cmd, logL_vec_ccd, masks)

        where:
            logL_vec_cmd: np.ndarray or None (length == sum(mask_obs_cmd))
            logL_vec_ccd: np.ndarray or None (length == sum(mask_obs_ccd))
            masks: dict with 'obs_cmd', 'syn_cmd', 'obs_ccd', 'syn_ccd' boolean masks
    """

    if synth_clust is None or not np.asarray(synth_clust).any():
        # Keep return signature consistent
        if rmv_contam:
            return np.inf, np.zeros(len(obs_data['obs_mag']), dtype=bool), None, None, {}
        else:
            return np.inf, None, None, {}

    # Observed data
    obs_mag = np.asarray(obs_data['obs_mag'])
    obs_colors = [np.asarray(c) for c in obs_data['obs_colors']]

    obs_e_mag = np.asarray(obs_data['obs_e_mag'])
    obs_e_colors = [np.asarray(e) for e in obs_data['obs_e_colors']]

    n_colors_obs = len(obs_colors)

    # Synthetic data
    synth_arrs = [np.asarray(x)for x in synth_clust]
    mag_syn = synth_arrs[0]
    colors_syn = synth_arrs[1:1 + n_colors_obs]
    mass_probs_syn = synth_arrs[-1] if (is_imf_weighted and len(synth_arrs) == 1 + n_colors_obs + 1) else None

    total_logL = 0.0
    logL_vec_cmd, logL_vec_ccd = None, None

    masks = {'obs_cmd': None, 'syn_cmd': None, 'obs_ccd': None, 'syn_ccd': None}

    # CMD likelihood (mag, color1)
    if compute_l in {"cmd", "cmd_ccd"} and n_colors_obs >= 1:
        mask_syn_cmd = ~np.isnan(mag_syn) & ~np.isnan(colors_syn[0])
        mask_obs_cmd = (~np.isnan(obs_mag) & ~np.isnan(obs_colors[0]) &
                        ~np.isnan(obs_e_mag) & ~np.isnan(obs_e_colors[0]))
        masks['syn_cmd'] = mask_syn_cmd
        masks['obs_cmd'] = mask_obs_cmd

        if mask_syn_cmd.any() and mask_obs_cmd.any():
            # Build arrays
            pts_syn_cmd_raw = np.column_stack([mag_syn[mask_syn_cmd], colors_syn[0][mask_syn_cmd]])
            pts_obs_cmd_raw = np.column_stack([obs_mag[mask_obs_cmd], obs_colors[0][mask_obs_cmd]])

            # Standard scale both observed and synthetic together
            scaler_cmd = StandardScaler()
            scaler_cmd.fit(np.vstack([pts_obs_cmd_raw, pts_syn_cmd_raw]))
            pts_obs_cmd = scaler_cmd.transform(pts_obs_cmd_raw)
            pts_syn_cmd = scaler_cmd.transform(pts_syn_cmd_raw)

            # Fixed error of 0.01 in scaled space
            pts_err_cmd = np.full_like(pts_obs_cmd, 0.05)

            weights_cmd = mass_probs_syn[mask_syn_cmd] if mass_probs_syn is not None else None

            logL_vec_cmd = nn_loglik(pts_obs_cmd, pts_err_cmd, pts_syn_cmd, weights_cmd,
                                     k=k, return_per_point=True)

            total_logL += np.sum(logL_vec_cmd)

            # Bluest-point penalty (now in scaled CMD space)
            if penalty:
                # --- observed bluest point selection ---
                obs_c1_full = obs_colors[0]
                valid_obs_c1 = np.where(mask_obs_cmd)[0]
                idx_obs_blue = valid_obs_c1[np.argmin(obs_c1_full[valid_obs_c1])]
                color_blue = obs_c1_full[idx_obs_blue]
                cmin, cmax = color_blue - penalty_halfwidth, color_blue + penalty_halfwidth

                # observed indices within the interval
                obs_in_region = valid_obs_c1[(obs_c1_full[valid_obs_c1] >= cmin) &
                                             (obs_c1_full[valid_obs_c1] <= cmax)]
                if obs_in_region.size > 0:
                    idx_obs_bright = obs_in_region[np.argmin(obs_mag[obs_in_region])]
                    # raw observed point (color, mag)
                    obs_point_cmd_raw = np.array([obs_colors[0][idx_obs_bright],
                                                  obs_mag[idx_obs_bright]])
                    obs_err_cmd_raw = np.array([obs_e_colors[0][idx_obs_bright],
                                                obs_e_mag[idx_obs_bright]])

                    # scale observed point
                    obs_point_cmd = scaler_cmd.transform(obs_point_cmd_raw.reshape(1, -1))[0]

                    # scale observed error, then apply floor
                    obs_err_cmd_scaled = scaler_cmd.transform(obs_err_cmd_raw.reshape(1, -1))[0]
                    sigma_penalty = np.linalg.norm(np.maximum(obs_err_cmd_scaled, 0.15))

                    # --- synthetic candidates within same color interval ---
                    syn_c1_full = colors_syn[0]
                    syn_in_region = np.where(mask_syn_cmd &
                                             (syn_c1_full >= cmin) & (syn_c1_full <= cmax))[0]

                    if syn_in_region.size > 0:
                        n_obs = len(valid_obs_c1)  # number of observed stars
                        n_draws = 10
                        penalties = []

                        for _ in range(n_draws):
                            # random subsample of synthetic stars
                            if syn_in_region.size >= n_obs:
                                syn_sample = np.random.choice(syn_in_region, size=n_obs, replace=False)
                            else:
                                syn_sample = np.random.choice(syn_in_region, size=n_obs, replace=True)

                            # pick brightest synthetic star in this subsample
                            idx_syn_bright = syn_sample[np.argmin(mag_syn[syn_sample])]
                            syn_point_cmd_raw = np.array([colors_syn[0][idx_syn_bright],
                                                          mag_syn[idx_syn_bright]])

                            # scale synthetic point
                            syn_point_cmd = scaler_cmd.transform(syn_point_cmd_raw.reshape(1, -1))[0]

                            # compute penalty in scaled space with error floor
                            penalty_dist = np.linalg.norm(syn_point_cmd - obs_point_cmd)
                            logL_penalty = -0.5 * (penalty_dist / sigma_penalty) ** 2 \
                                           - 0.5 * np.log(2 * np.pi * sigma_penalty ** 2)
                            penalties.append(logL_penalty)

                        # average penalty across draws
                        avg_penalty = np.mean(penalties)
                        total_logL += avg_penalty

    # CCD likelihood (color1, color2)
    if compute_l in {"ccd", "cmd_ccd"} and n_colors_obs >= 2:
        mask_syn_ccd = ~np.isnan(colors_syn[0]) & ~np.isnan(colors_syn[1])
        mask_obs_ccd = (~np.isnan(obs_colors[0]) & ~np.isnan(obs_colors[1]) &
                        ~np.isnan(obs_e_colors[0]) & ~np.isnan(obs_e_colors[1]))
        masks['syn_ccd'] = mask_syn_ccd
        masks['obs_ccd'] = mask_obs_ccd

        if mask_syn_ccd.any() and mask_obs_ccd.any():
            pts_syn_ccd_raw = np.column_stack([colors_syn[0][mask_syn_ccd], colors_syn[1][mask_syn_ccd]])
            pts_obs_ccd_raw = np.column_stack([obs_colors[0][mask_obs_ccd], obs_colors[1][mask_obs_ccd]])

            # Standard scale both observed and synthetic together
            scaler_ccd = StandardScaler()
            scaler_ccd.fit(np.vstack([pts_obs_ccd_raw, pts_syn_ccd_raw]))
            pts_obs_ccd = scaler_ccd.transform(pts_obs_ccd_raw)
            pts_syn_ccd = scaler_ccd.transform(pts_syn_ccd_raw)

            # Fixed error of 0.01 in scaled space
            pts_err_ccd = np.full_like(pts_obs_ccd, 0.05)

            weights_ccd = mass_probs_syn[mask_syn_ccd] if mass_probs_syn is not None else None

            logL_vec_ccd = nn_loglik(pts_obs_ccd, pts_err_ccd, pts_syn_ccd, weights_ccd,
                                     k=k, return_per_point=True)

            total_logL += np.sum(logL_vec_ccd)

    # Optional plotting: only plot masked observed points to match vector lengths
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # CMD
        if logL_vec_cmd is not None:
            # Use only masked observed points
            obs_color1_cmd = obs_colors[0][masks['obs_cmd']]
            obs_mag_cmd = obs_mag[masks['obs_cmd']]
            syn_color1_cmd = colors_syn[0][masks['syn_cmd']]
            syn_mag_cmd = mag_syn[masks['syn_cmd']]

            sc1 = axes[0].scatter(obs_color1_cmd, obs_mag_cmd, c=logL_vec_cmd,
                                  cmap='viridis', s=40, edgecolor='k', label='Observed')
            axes[0].scatter(syn_color1_cmd, syn_mag_cmd, c='red', alpha=0.5, s=20, label='Synthetic')
            axes[0].invert_yaxis()
            axes[0].set_xlabel("Color 1")
            axes[0].set_ylabel("Magnitude")
            axes[0].set_title("CMD")
            axes[0].legend()
            fig.colorbar(sc1, ax=axes[0], label='Log-likelihood per point')
        else:
            axes[0].set_visible(False)

        # CCD
        if logL_vec_ccd is not None:
            obs_c1_ccd = obs_colors[0][masks['obs_ccd']]
            obs_c2_ccd = obs_colors[1][masks['obs_ccd']]
            syn_c1_ccd = colors_syn[0][masks['syn_ccd']]
            syn_c2_ccd = colors_syn[1][masks['syn_ccd']]

            sc2 = axes[1].scatter(obs_c1_ccd, obs_c2_ccd, c=logL_vec_ccd,
                                  cmap='plasma', s=40, edgecolor='k', label='Observed')
            axes[1].scatter(syn_c1_ccd, syn_c2_ccd, c='red', alpha=0.5, s=20, label='Synthetic')
            axes[1].set_xlabel("Color 1")
            axes[1].set_ylabel("Color 2")
            axes[1].set_title("CCD")
            axes[1].legend()
            fig.colorbar(sc2, ax=axes[1], label='Log-likelihood per point')
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        plt.show()

    # Return values — keep consistent signatures
    if rmv_contam:
        contamination_mask = np.zeros(len(obs_mag), dtype=bool)
        return -total_logL, contamination_mask
    else:
        return -total_logL



def plot_kde_overlay(kde_gauss, kde_exp, scaler, grid_ref_pts, overlay_pts,
                     xlabel, ylabel, title, is_cmd=True):
    """
    Plot a 2D KDE density heatmap with optional overlay points,
    using the average of Gaussian and Exponential KDEs.

    Parameters
    ----------
    kde_gauss : fitted sklearn.neighbors.KernelDensity
        Gaussian KDE model trained on scaled synthetic points.
    kde_exp : fitted sklearn.neighbors.KernelDensity
        Exponential KDE model trained on scaled synthetic points.
    scaler : sklearn.preprocessing.StandardScaler or None
        Scaler used to normalize CMD/CCD points. If None, no scaling applied.
    grid_ref_pts : ndarray, shape (n_samples, 2)
        Reference points (synthetic CMD/CCD) to set plotting grid limits.
    overlay_pts : ndarray, shape (m_samples, 2)
        Points to overlay (observed CMD/CCD).
    xlabel, ylabel : str
        Axis labels.
    title : str
        Plot title.
    is_cmd : bool
        If True, treat axes as CMD (mag vs. color) and invert y-axis.
    """
    if grid_ref_pts is None or grid_ref_pts.size == 0:
        return

    # Grid limits in original CMD/CCD units
    x_min, x_max = -4, 4
    y_min, y_max = 4, -4

    X, Y = np.meshgrid(np.linspace(x_min, x_max, 200),
                       np.linspace(y_min, y_max, 200))
    grid_plot = np.vstack([X.ravel(), Y.ravel()]).T

    # For CMD, training order was (mag, color), so swap
    if is_cmd:
        grid_kde = grid_plot[:, [1, 0]]
    else:
        grid_kde = grid_plot

    # Evaluate both KDEs and average
    Z_gauss = np.exp(kde_gauss.score_samples(grid_kde))
    Z_exp   = np.exp(kde_exp.score_samples(grid_kde))
    Z = 0.5 * (Z_gauss + Z_exp)
    Z = Z.reshape(X.shape)
    Z[Z <= 1e-10] = 1e-10

    # Plot density heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(Z, extent=[x_min, x_max, y_max, y_min],
               origin='upper', cmap='viridis', aspect='auto',
               norm=LogNorm(vmin=Z.min(), vmax=Z.max()))
    if overlay_pts is not None and overlay_pts.size > 0:
        plt.scatter(overlay_pts[:, 0], overlay_pts[:, 1],
                    c='red', s=10, label='Overlay Points')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label='KDE Density (avg of Gauss+Exp)')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()


def kde_distance(obs_data: dict, synth_clust: np.ndarray,
                 compute_l: str = "cmd", bandwidth: float = 0.05,
                 plot: bool = True, rmv_contam: bool = False):

    if synth_clust is None or not np.asarray(synth_clust).any():
        return (1e9, np.array([], dtype=bool)) if rmv_contam else 1e9

    if not (isinstance(obs_data, dict) and 'obs_mag' in obs_data and 'obs_colors' in obs_data):
        raise ValueError("obs_data must be a dict with keys 'obs_mag' and 'obs_colors'")

    # Original data
    obs_mag = np.asarray(obs_data['obs_mag'])
    obs_colors = [np.asarray(c) for c in obs_data['obs_colors']]

    obs_e_mag = np.asarray(obs_data['obs_e_mag'])
    obs_e_colors = [np.asarray(e) for e in obs_data['obs_e_colors']]

    # Sampled data using Gaussian errors
    obs_mag  = np.random.normal(loc=obs_mag, scale=obs_e_mag)
    obs_colors = [np.random.normal(loc=c, scale=e) for c, e in zip(obs_colors, obs_e_colors)]

    # print(max(abs(obs_mag_mod - obs_mag)))
    # print(max(abs(obs_colors[0] - obs_colors_mod[0])))
    # print(max(abs(obs_colors[1] - obs_colors_mod[1])))

    n_colors_obs = len(obs_colors)

    synth_arrs = [np.asarray(x) for x in synth_clust]
    mag_syn = synth_arrs[0]
    colors_syn = synth_arrs[1:1 + n_colors_obs]

    combined_score = 0.0
    max_dist_cmd = 0.0  # default in case penalty isn't triggered

    # --- CMD block ---
    if compute_l in {"cmd", "cmd_ccd"} and n_colors_obs >= 1:
        mask_syn_cmd = ~np.isnan(mag_syn) & ~np.isnan(colors_syn[0])
        mask_obs_cmd = ~np.isnan(obs_mag) & ~np.isnan(obs_colors[0])
        if mask_syn_cmd.sum() > 0 and mask_obs_cmd.sum() > 0:
            pts_syn_cmd = np.vstack([mag_syn[mask_syn_cmd], colors_syn[0][mask_syn_cmd]]).T
            pts_obs_cmd = np.vstack([obs_mag[mask_obs_cmd], obs_colors[0][mask_obs_cmd]]).T

            scaler_cmd = StandardScaler().fit(pts_obs_cmd)
            obs_scaled_cmd = scaler_cmd.transform(pts_obs_cmd)
            syn_scaled_cmd = scaler_cmd.transform(pts_syn_cmd)

            kde_gauss = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(syn_scaled_cmd)
            kde_exp = KernelDensity(bandwidth=1.5 * bandwidth, kernel="exponential").fit(syn_scaled_cmd)

            log_floor = np.log(1e-6)

            log_scores_gauss = kde_gauss.score_samples(obs_scaled_cmd)
            log_scores_exp = kde_exp.score_samples(obs_scaled_cmd)

            log_scores_gauss = np.maximum(log_scores_gauss, log_floor)
            log_scores_exp = np.maximum(log_scores_exp, log_floor)

            log_scores_cmd = (log_scores_gauss + log_scores_exp) # log_scores_gauss # 0.5 *

            combined_score += np.sum(log_scores_cmd)

            # Penalty around bluest point
            idx_obs_blue = np.argmin(obs_colors[0])
            color_blue = obs_colors[0][idx_obs_blue]
            color_interval = (color_blue - 1, color_blue + 1)

            obs_in_region = np.where((obs_colors[0] >= color_interval[0]) &
                                     (obs_colors[0] <= color_interval[1]))[0]
            if obs_in_region.size > 0:
                # brightest observation = lowest magnitude in region
                idx_obs_bright = obs_in_region[np.argmin(obs_mag[obs_in_region])]
                obs_point_cmd = np.array([obs_colors[0][idx_obs_bright],
                                          obs_mag[idx_obs_bright]])

                syn_in_region = np.where((colors_syn[0] >= color_interval[0]) &
                                         (colors_syn[0] <= color_interval[1]))[0]

                if syn_in_region.size > 0:
                    # brightest synthetic = lowest magnitude in region
                    idx_syn_bright = syn_in_region[np.argmin(mag_syn[syn_in_region])]
                    syn_point_cmd = np.array([colors_syn[0][idx_syn_bright],
                                              mag_syn[idx_syn_bright]])

                    # distance between brightest obs and brightest syn
                    max_dist_cmd = np.linalg.norm(syn_point_cmd - obs_point_cmd)

            if plot:
                fig, ax = plt.subplots(figsize=(6, 5))
                sc = ax.scatter(obs_scaled_cmd[:, 1], obs_scaled_cmd[:, 0], c=log_scores_cmd,
                                cmap='viridis', s=20, label="Observed")
                ax.scatter(syn_scaled_cmd[:, 1], syn_scaled_cmd[:, 0], c='gray', s=10,
                           alpha=0.5, label="Synthetic KDE support")
                ax.invert_yaxis()
                ax.set_xlabel("Color (scaled)")
                ax.set_ylabel("Magnitude (scaled)")
                ax.set_title(f"CMD KDE: bandwidth={bandwidth}")
                ax.legend()
                plt.colorbar(sc, label="Log KDE score")
                plt.show()

                plot_kde_overlay(
                    kde_gauss, kde_exp, scaler_cmd,
                    syn_scaled_cmd,  # reference grid from synthetic CMD points
                    obs_scaled_cmd,  # overlay observed CMD points
                    xlabel="Color", ylabel="Magnitude",
                    title=f"CMD KDE overlay (avg Gauss+Exp, bw={bandwidth})",
                    is_cmd=True
                )

    # --- CCD block ---
    if compute_l in {"ccd", "cmd_ccd"} and n_colors_obs >= 2:
        mask_syn_ccd = ~np.isnan(colors_syn[0]) & ~np.isnan(colors_syn[1])
        mask_obs_ccd = ~np.isnan(obs_colors[0]) & ~np.isnan(obs_colors[1])
        if mask_syn_ccd.sum() > 0 and mask_obs_ccd.sum() > 0:
            pts_syn_ccd = np.vstack([colors_syn[0][mask_syn_ccd], colors_syn[1][mask_syn_ccd]]).T
            pts_obs_ccd = np.vstack([obs_colors[0][mask_obs_ccd], obs_colors[1][mask_obs_ccd]]).T

            scaler_ccd = StandardScaler().fit(pts_obs_ccd)
            obs_scaled_ccd = scaler_ccd.transform(pts_obs_ccd)
            syn_scaled_ccd = scaler_ccd.transform(pts_syn_ccd)

            kde_gauss = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(syn_scaled_ccd)
            kde_exp = KernelDensity(bandwidth=1.5 * bandwidth, kernel="exponential").fit(syn_scaled_ccd)

            log_scores_gauss = kde_gauss.score_samples(obs_scaled_ccd)
            log_scores_exp = kde_exp.score_samples(obs_scaled_ccd)

            log_floor = np.log(1e-6)  # ≈ -9.21034
            log_scores_gauss = np.maximum(log_scores_gauss, log_floor)
            log_scores_exp = np.maximum(log_scores_exp, log_floor)

            log_scores_ccd = 0.5 * (log_scores_gauss + log_scores_exp) # log_scores_gauss #

            combined_score += np.sum(log_scores_ccd)

            if plot:
                fig, ax = plt.subplots(figsize=(6, 5))
                sc = ax.scatter(pts_obs_ccd[:, 0], pts_obs_ccd[:, 1], c=log_scores_ccd,
                                cmap='plasma', s=20, label="Observed")
                ax.scatter(pts_syn_ccd[:, 0], pts_syn_ccd[:, 1], c='gray', s=10,
                           alpha=0.5, label="Synthetic KDE support")
                ax.set_xlabel("Color 1")
                ax.set_ylabel("Color 2")
                ax.set_title(f"CCD KDE: bandwidth={bandwidth}")
                ax.legend()
                plt.colorbar(sc, label="Log KDE score")
                plt.show()

                plot_kde_overlay(
                    kde_gauss, kde_exp, scaler_ccd,
                    pts_syn_ccd,  # synthetic CCD points
                    pts_obs_ccd,  # observed CCD points
                    xlabel="Color 1", ylabel="Color 2",
                    title=f"CCD KDE overlay (avg Gauss+Exp, bw={bandwidth})",
                    is_cmd=False
                )

    combined_score += - max_dist_cmd * (len(log_scores_cmd))

    if rmv_contam:
        contamination_mask = np.zeros(len(obs_mag), dtype=bool)
        return -combined_score, contamination_mask
    else:
        return -combined_score


def lkl_data(
    bin_method: str,
    mag_v: np.ndarray,
    colors_v: list[np.ndarray],
    compute_l: str = "cmd",  # Default flag
) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Prepare data for likelihood calculation.

    This function calculates the Hess diagram of the observed cluster and
    prepares the data for the likelihood calculation.

    :param bin_method: Method to use for binning the data.
    :type bin_method: str
    :param mag_v: Array of magnitudes.
    :type mag_v: np.ndarray
    :param colors_v: List of arrays of colors.
    :type colors_v: list[np.ndarray]
    :param compute_l: Flag to specify likelihood calculation type.
        - "cmd" -> Compute CMD histogram.
        - "ccd" -> Compute CCD histogram (requires at least two colors).
        - "cmd_ccd" -> Compute both CMD and CCD histograms (requires two colors).
    :type compute_l: str

    :return: Bin ranges, number of bins, indexes of bins with stars, and flattened histogram.
    :rtype: tuple[list, list, np.ndarray, np.ndarray]

    :raises ValueError: If `compute_l` is "ccd" or "cmd_ccd" but fewer than two colors are provided.
    :raises ValueError: If `compute_l` is not one of ["cmd", "ccd", "cmd_ccd"].
    """

    # Validate compute_l flag
    valid_flags = {"cmd", "ccd", "cmd_ccd"}
    if compute_l not in valid_flags:
        raise ValueError(f"Invalid value for 'compute_l': '{compute_l}'. Must be one of {valid_flags}.")

    # Validate color input when needed
    if compute_l in {"ccd", "cmd_ccd"} and len(colors_v) < 2:
        raise ValueError("Cannot compute CCD histogram: At least two colors are required, but fewer were provided.")

    # Obtain bin edges for each dimension, defining a grid.
    ranges, Nbins = bin_edges_f(bin_method, mag_v, colors_v)

    # Obtain histogram for observed cluster.
    hess_diag = []

    # Compute CMD histogram if "cmd" or "cmd_ccd"
    if compute_l in {"cmd", "cmd_ccd"}:
        # Fast 2D histogram
        hess_diag.append(
            histogram2d(
                mag_v,
                colors_v[0],
                range=[
                    [ranges[0][0], ranges[0][1]],
                    [ranges[1][0], ranges[1][1]],
                ],
                bins=[Nbins[0], Nbins[1]],
            )
        )

    # Compute CCD histogram if "ccd" or "cmd_ccd"
    if compute_l in {"ccd", "cmd_ccd"}:
        hess_diag.append(
            histogram2d(
                colors_v[0],
                colors_v[1],
                range=[
                    [ranges[1][0], ranges[1][1]],
                    [ranges[2][0], ranges[2][1]],
                ],
                bins=[Nbins[1], Nbins[2]],
            )
        )

    # Flatten array
    cl_histo_f = []
    for diag in hess_diag:
        cl_histo_f += list(np.array(diag).ravel())
    cl_histo_f = np.array(cl_histo_f)

    # Index of bins where stars were observed
    cl_z_idx = cl_histo_f != 0

    # Remove all bins where n_i=0 (no observed stars)
    cl_histo_f_z = cl_histo_f[cl_z_idx]

    return ranges, Nbins, cl_z_idx, cl_histo_f_z


def bin_edges_f(
    bin_method: str, mag: np.ndarray, colors: list[np.ndarray]
) -> tuple[list, list]:
    """Calculate bin edges for the Hess diagram.

    This function calculates the bin edges for the Hess diagram, using
    different methods.

    :param bin_method: Method to use for binning the data.
    :type bin_method: str
    :param mag: Array of magnitudes.
    :type mag: np.ndarray
    :param colors: List of arrays of colors.
    :type colors: list[np.ndarray]

    :return: Bin ranges and number of bins.
    :rtype: tuple[list, list]
    """

    bin_edges = []

    if bin_method == "fixed":
        N_mag, N_col = 25, 20
        # Magnitude
        mag_min, mag_max = np.nanmin(mag), np.nanmax(mag)
        bin_edges.append(np.linspace(mag_min, mag_max, N_mag))
        # Colors
        for col in colors:
            col_min, col_max = np.nanmin(col), np.nanmax(col)
            bin_edges.append(np.linspace(col_min, col_max, N_col))
    else:
        bin_edges.append(calculate_bin_edges(mag[~np.isnan(mag)], bins=bin_method))  # pyright: ignore
        for col in colors:
            bin_edges.append(
                calculate_bin_edges(col[~np.isnan(col)], bins=bin_method)  # pyright: ignore
            )

    # Extract ranges and number of bins for each dimension (magnitude and colors),
    # used by histogram2d
    ranges, Nbins = [], []
    for be in bin_edges:
        ranges.append([be[0], be[-1]])
        Nbins.append(len(be))

    return ranges, Nbins


def plot_bin_weights_imshow(
    unweighted_hist_cmd, weighted_hist_cmd,
    x_range_cmd, y_range_cmd, nbx_cmd, nby_cmd,
    tremmel, max_lk,
    unweighted_hist_ccd=None, weighted_hist_ccd=None,
    x_range_ccd=None, y_range_ccd=None, nbx_ccd=None, nby_ccd=None,
    cmap="viridis"
):
    """
    Plot CMD (mandatory) and optionally CCD (if unweighted_hist_ccd and weighted_hist_ccd are provided).

    Parameters
    ----------
    unweighted_hist_cmd, weighted_hist_cmd : 1D arrays
        Flattened histograms for CMD with shape (nbx_cmd * nby_cmd,).
    x_range_cmd : [min, max]
        magnitude range (ranges[0])
    y_range_cmd : [min, max]
        color range (ranges[1])
    nbx_cmd, nby_cmd : int
        bins for CMD (Nbins[0], Nbins[1])

    tremmel, max_lk : scalar
        values shown in the figure title/annotation.

    unweighted_hist_ccd, weighted_hist_ccd : optional 1D arrays
        Flattened histograms for CCD with shape (nbx_ccd * nby_ccd,).
    x_range_ccd, y_range_ccd : optional [min, max]
        ranges for CCD axes (color1, color2) -> typically ranges[1], ranges[2]
    nbx_ccd, nby_ccd : optional ints
        bins for CCD (Nbins[1], Nbins[2])

    Returns
    -------
    None (shows the figure)
    """

    # reshape CMD arrays
    H_orig_cmd = np.asarray(unweighted_hist_cmd).reshape((nbx_cmd, nby_cmd))
    H_weight_cmd = np.asarray(weighted_hist_cmd).reshape((nbx_cmd, nby_cmd))

    # extents: x=color, y=mag (so extent = [x_min, x_max, y_min, y_max])
    mag_range = x_range_cmd
    color_range = y_range_cmd
    extent_cmd = [color_range[0], color_range[1], mag_range[0], mag_range[1]]

    # decide grid layout depending on whether CCD is provided
    has_ccd = (unweighted_hist_ccd is not None) and (weighted_hist_ccd is not None) \
              and (x_range_ccd is not None) and (y_range_ccd is not None) \
              and (nbx_ccd is not None) and (nby_ccd is not None)

    if has_ccd:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        ax_cmd = axes[0]
        ax_ccd = axes[1]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax_cmd = axes
        ax_ccd = None

    # ------- CMD row (always present) -------
    im0 = ax_cmd[0].imshow(H_orig_cmd, origin="lower", extent=extent_cmd, aspect="auto", cmap=cmap)
    ax_cmd[0].set_title("CMD: Original counts")
    ax_cmd[0].set_xlabel("Color")
    ax_cmd[0].set_ylabel("Magnitude")
    fig.colorbar(im0, ax=ax_cmd[0], fraction=0.046)

    im1 = ax_cmd[1].imshow(H_weight_cmd, origin="lower", extent=extent_cmd, aspect="auto", cmap=cmap)
    ax_cmd[1].set_title("CMD: Weighted counts")
    ax_cmd[1].set_xlabel("Color")
    ax_cmd[1].set_ylabel("Magnitude")
    fig.colorbar(im1, ax=ax_cmd[1], fraction=0.046)

    ratio_cmd = np.zeros_like(H_orig_cmd, dtype=float)
    mask_cmd = H_orig_cmd > 0
    ratio_cmd[mask_cmd] = H_weight_cmd[mask_cmd] / H_orig_cmd[mask_cmd]
    im2 = ax_cmd[2].imshow(ratio_cmd, origin="lower", extent=extent_cmd, aspect="auto", cmap="coolwarm")
    ax_cmd[2].set_title("CMD: Weight factor (weighted / original)")
    ax_cmd[2].set_xlabel("Color")
    ax_cmd[2].set_ylabel("Magnitude")
    fig.colorbar(im2, ax=ax_cmd[2], fraction=0.046)

    # draw bin edges for CMD
    color_edges_cmd = np.linspace(color_range[0], color_range[1], nby_cmd + 1)
    mag_edges_cmd = np.linspace(mag_range[0], mag_range[1], nbx_cmd + 1)
    for a in ax_cmd:
        for x in color_edges_cmd:
            a.axvline(x, lw=0.6, alpha=0.5)
        for y in mag_edges_cmd:
            a.axhline(y, lw=0.6, alpha=0.5)
        # CMD convention: brighter up -> invert y-axis
        a.invert_yaxis()

    # ------- CCD row (optional) -------
    if has_ccd:
        H_orig_ccd = np.asarray(unweighted_hist_ccd).reshape((nbx_ccd, nby_ccd))
        H_weight_ccd = np.asarray(weighted_hist_ccd).reshape((nbx_ccd, nby_ccd))
        extent_ccd = [x_range_ccd[0], x_range_ccd[1], y_range_ccd[0], y_range_ccd[1]]  # x=color1, y=color2

        im3 = ax_ccd[0].imshow(H_orig_ccd, origin="lower", extent=extent_ccd, aspect="auto", cmap=cmap)
        ax_ccd[0].set_title("CCD: Original counts")
        ax_ccd[0].set_xlabel("Color 1")
        ax_ccd[0].set_ylabel("Color 2")
        fig.colorbar(im3, ax=ax_ccd[0], fraction=0.046)

        im4 = ax_ccd[1].imshow(H_weight_ccd, origin="lower", extent=extent_ccd, aspect="auto", cmap=cmap)
        ax_ccd[1].set_title("CCD: Weighted counts")
        ax_ccd[1].set_xlabel("Color 1")
        ax_ccd[1].set_ylabel("Color 2")
        fig.colorbar(im4, ax=ax_ccd[1], fraction=0.046)

        ratio_ccd = np.zeros_like(H_orig_ccd, dtype=float)
        mask_ccd = H_orig_ccd > 0
        ratio_ccd[mask_ccd] = H_weight_ccd[mask_ccd] / H_orig_ccd[mask_ccd]
        im5 = ax_ccd[2].imshow(ratio_ccd, origin="lower", extent=extent_ccd, aspect="auto", cmap="coolwarm")
        ax_ccd[2].set_title("CCD: Weight factor (weighted / original)")
        ax_ccd[2].set_xlabel("Color 1")
        ax_ccd[2].set_ylabel("Color 2")
        fig.colorbar(im5, ax=ax_ccd[2], fraction=0.046)

        # draw bin edges for CCD (note nbx_ccd-> rows, nby_ccd-> cols)
        color1_edges = np.linspace(x_range_ccd[0], x_range_ccd[1], nby_ccd + 1)
        color2_edges = np.linspace(y_range_ccd[0], y_range_ccd[1], nbx_ccd + 1)
        for a in ax_ccd:
            for x in color1_edges:
                a.axvline(x, lw=0.6, alpha=0.5)
            for y in color2_edges:
                a.axhline(y, lw=0.6, alpha=0.5)

            a.invert_yaxis()
            # do not invert CCD axes

    # global title / annotation
    suptitle = f"Tremmel LKL: {tremmel:.4g}   Max LKL (run): {max_lk:.4g}"
    plt.suptitle(suptitle, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def tremmel(
    ranges: list,
    Nbins: list,
    cl_z_idx: np.ndarray,
    cl_histo_f_z: np.ndarray,
    max_lkl: float,
    synth_clust: np.ndarray,
    compute_l: str = "cmd",
    is_imf_weighted: bool = False,
) -> float:
    r"""Poisson likelihood ratio as defined in Tremmel et al (2013), Eq 10 with
    v_{i,j}=1. This returns the log likelihood.

    .. math::

        p(d|\theta) = \prod_i^N \frac{\Gamma(n_i+m_i+\frac{1}{2})}
        {2^{n_i+m_i+\frac{1}{2}} n_i!\Gamma(m_i+\frac{1}{2}))}

    .. math::

        \log(p) = \sum_i^N \left[\log\Gamma(n_i+m_i+\frac{1}{2})
        - (m_i+n_i+\frac{1}{2})\log2 -\log n_i!
        - \log \Gamma(m_i+\frac{1}{2}) \right]

    Minus logarithm:

    .. math::

        \log(p) = \sum_i^N \left[\log\Gamma(n_i+m_i+\frac{1}{2})-
        \log \Gamma(m_i+\frac{1}{2}) \right]
        - 0.693  (M+N+\frac{1}{2}) - \sum_i^N \log n_i!

    .. math::

        \log(p) = SumLogGamma(n_i, m_i) -0.693 (N+\frac{1}{2}) -
        \sum_i^N \log n_i! - 0.693\,M

    .. math::

        \log(p) = f(n_i) + SumLogGamma(n_i, m_i) - 0.693\,M

    .. math::

        \log(p)\approx SumLogGamma(n_i, m_i) - 0.693\,M

    :param ranges: Per-dimension ranges.
    :type ranges: list
    :param Nbins: Per-dimension total number of bins
    :type Nbins: list
    :param cl_z_idx: Index of bins where the number of stars is not 0
    :type cl_z_idx: np.ndarray
    :param cl_histo_f_z: Flattened observed Hess diagram with the empty bins removed
    :type cl_histo_f_z: np.ndarray
    :param max_lkl: Maximum likelihood value, used for normalization
    :type max_lkl: float
    :param synth_clust: Synthetic cluster data.
    :param compute_l: Specifies which likelihood to compute:
                      - "cmd" for Color-Magnitude Diagram
                      - "ccd" for Color-Color Diagram
                      - "cmd_ccd" for both
    :type synth_clust: np.ndarray

    :return: Log likelihood value.
    :rtype: float
    """
    if synth_clust is None or not np.asarray(synth_clust).any():
        return -1.0e09

    # Parse synthetic cluster input depending on whether IMF weights exist
    if is_imf_weighted:
        mag = synth_clust[0]
        colors = synth_clust[1: len(synth_clust) - 1]
        mass_probs = synth_clust[-1]
    else:
        mag = synth_clust[0]
        colors = synth_clust[1:]

    if compute_l not in ["cmd", "ccd", "cmd_ccd"]:
        raise ValueError(f"Invalid compute_l value: {compute_l}.")

    if compute_l in ["ccd", "cmd_ccd"] and len(colors) < 2:
        raise ValueError(f"Cannot compute '{compute_l}': At least two colors are required for CCD.")

    # Helper: compute mean probability and counts per 2D bin
    def mean_prob_per_bin(x, y, probs, x_range, y_range, nbx, nby):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        probs = np.asarray(probs).ravel()
        nbx = int(nbx)
        nby = int(nby)
        x_edges = np.linspace(x_range[0], x_range[1], nbx + 1)
        y_edges = np.linspace(y_range[0], y_range[1], nby + 1)
        ix = np.digitize(x, x_edges) - 1
        iy = np.digitize(y, y_edges) - 1
        mask = (ix >= 0) & (ix < nbx) & (iy >= 0) & (iy < nby)

        sum_probs = np.zeros((nbx, nby), dtype=float)
        counts = np.zeros((nbx, nby), dtype=float)

        if mask.any():
            np.add.at(sum_probs, (ix[mask], iy[mask]), probs[mask])
            np.add.at(counts, (ix[mask], iy[mask]), 1.0)

        meanp = np.zeros_like(sum_probs)
        nz = counts > 0
        # keep your sqrt(prob/count) definition
        meanp[nz] = np.sqrt(sum_probs[nz] / counts[nz])

        return meanp, counts

    syn_histo_f_parts = []
    unweighted_hist_parts = []

    # Build synthetic histograms: either IMF-weighted (use mass_probs) or plain counts
    if is_imf_weighted:
        # CMD: weighted using mass_probs (and also store counts)
        if compute_l in ["cmd", "cmd_ccd"]:
            meanp_cmd, counts_cmd = mean_prob_per_bin(
                mag, colors[0], mass_probs,
                x_range=[ranges[0][0], ranges[0][1]],
                y_range=[ranges[1][0], ranges[1][1]],
                nbx=Nbins[0], nby=Nbins[1]
            )
            weighted_cmd = np.zeros_like(counts_cmd, dtype=float)
            nz = meanp_cmd > 0
            weighted_cmd[nz] = counts_cmd[nz] / meanp_cmd[nz]
            syn_histo_f_parts += list(weighted_cmd.ravel())
            unweighted_hist_parts += list(counts_cmd.ravel())

        # CCD: weighted using mass_probs
        if compute_l in ["ccd", "cmd_ccd"]:
            meanp_ccd, counts_ccd = mean_prob_per_bin(
                colors[0], colors[1], mass_probs,
                x_range=[ranges[1][0], ranges[1][1]],
                y_range=[ranges[2][0], ranges[2][1]],
                nbx=Nbins[1], nby=Nbins[2]
            )
            weighted_ccd = np.zeros_like(counts_ccd, dtype=float)
            nz_ccd = meanp_ccd > 0
            weighted_ccd[nz_ccd] = counts_ccd[nz_ccd] / meanp_ccd[nz_ccd]
            syn_histo_f_parts += list(weighted_ccd.ravel())
            unweighted_hist_parts += list(counts_ccd.ravel())

    else:
        # Not IMF-weighted: use plain counts from histogram2d for both syn and unweighted
        if compute_l in ["cmd", "cmd_ccd"]:
            hess_diag_cmd, _, _ = np.histogram2d(
                mag, colors[0],
                bins=[Nbins[0], Nbins[1]],
                range=[[ranges[0][0], ranges[0][1]], [ranges[1][0], ranges[1][1]]]
            )
            syn_histo_f_parts += list(hess_diag_cmd.ravel())
            unweighted_hist_parts += list(hess_diag_cmd.ravel())

        if compute_l in ["ccd", "cmd_ccd"]:
            hess_diag_ccd, _, _ = np.histogram2d(
                colors[0], colors[1],
                bins=[Nbins[1], Nbins[2]],
                range=[[ranges[1][0], ranges[1][1]], [ranges[2][0], ranges[2][1]]]
            )
            syn_histo_f_parts += list(hess_diag_ccd.ravel())
            unweighted_hist_parts += list(hess_diag_ccd.ravel())

    syn_histo_f = np.array(syn_histo_f_parts, dtype=float)
    unweighted_hist = np.array(unweighted_hist_parts, dtype=float)

    # select non-empty bins (same indexing as observations)
    syn_histo_f_z = syn_histo_f[cl_z_idx]
    unweighted_hist_z = unweighted_hist[cl_z_idx]

    # Build weight factors: syn/unweighted where unweighted>0, else 1
    weight_factors = np.ones_like(unweighted_hist, dtype=float)
    nonzero_mask = unweighted_hist > 0
    weight_factors[nonzero_mask] = syn_histo_f[nonzero_mask] / unweighted_hist[nonzero_mask]

    # Apply weights to observed histogram (only z-indexed bins)
    cl_histo_f_z_weighted = cl_histo_f_z * weight_factors[cl_z_idx]

    # Tremmel likelihood using weighted observations and weighted synthetic model
    tremmel_lkl_weighted = np.sum(
        loggamma(cl_histo_f_z_weighted + syn_histo_f_z + 0.5) - loggamma(syn_histo_f_z + 0.5)
    )

    # Per-run maximum likelihood: compare weighted observations to themselves
    tremmel_lkl_max_run = np.sum(
        loggamma(cl_histo_f_z_weighted + cl_histo_f_z_weighted + 0.5) - loggamma(cl_histo_f_z_weighted + 0.5)
    )

    if tremmel_lkl_max_run <= 0:
        if max_lkl > 0:
            max_lkl_use = max_lkl
        else:
            raise ValueError("Computed per-run max_lkl is non-positive and provided max_lkl is not valid (>0).")
    else:
        max_lkl_use = tremmel_lkl_max_run

    # # lengths
    # len_cmd = Nbins[0] * Nbins[1]  # CMD: mag x color
    # len_ccd = Nbins[1] * Nbins[2]  # CCD: color1 x color2
    #
    # # assume unweighted_hist is full concatenation [cmd.ravel(), ccd.ravel()]
    # cmd_unweighted = unweighted_hist[:len_cmd]
    # ccd_unweighted = unweighted_hist[len_cmd: len_cmd + len_ccd]
    #
    # # same for weighted (you used weighted = unweighted * weight_factors)
    # full_weighted = unweighted_hist * weight_factors
    # cmd_weighted = full_weighted[:len_cmd]
    # ccd_weighted = full_weighted[len_cmd: len_cmd + len_ccd]
    #
    # # ranges:
    # mag_range = [ranges[0][0], ranges[0][1]]
    # color_range = [ranges[1][0], ranges[1][1]]
    # color1_range = [ranges[1][0], ranges[1][1]]
    # color2_range = [ranges[2][0], ranges[2][1]]
    #
    # plot_bin_weights_imshow(
    #     cmd_unweighted, cmd_weighted,
    #     x_range_cmd=mag_range, y_range_cmd=color_range, nbx_cmd=Nbins[0], nby_cmd=Nbins[1],
    #     tremmel=tremmel_lkl_weighted, max_lk=tremmel_lkl_max_run,
    #     unweighted_hist_ccd=ccd_unweighted, weighted_hist_ccd=ccd_weighted,
    #     x_range_ccd=color1_range, y_range_ccd=color2_range, nbx_ccd=Nbins[1], nby_ccd=Nbins[2]
    # )

    return 1 - tremmel_lkl_weighted / max_lkl_use



# def visual(cluster_dict, synth_clust):
#     # If synthetic  cluster is empty, assign a small likelihood value.
#     if not synth_clust.any():
#         return -1.0e09

#     mag_o, colors_o = cluster_dict["mag"], cluster_dict["colors"]
#     mag_s, colors_s = synth_clust[0], synth_clust[1:]

#     N_mag, N_col = 15, 10
#     mag = list(mag_o) + list(mag_s)
#     col = list(colors_o[0]) + list(colors_s[0])
#     mag_min, mag_max = np.nanmin(mag), np.nanmax(mag)
#     bin_edges = [np.linspace(mag_min, mag_max, N_mag)]
#     col_min, col_max = np.nanmin(col), np.nanmax(col)
#     bin_edges.append(np.linspace(col_min, col_max, N_col))

#     # Obtain histogram for observed cluster.
#     cl_histo_f = []
#     for i, col_o in enumerate(colors_o):
#         hess_diag = np.histogram2d(mag_o, col_o, bins=bin_edges)[0]
#         # Flatten array
#         cl_histo_f += list(hess_diag.ravel())
#     cl_histo_f = np.array(cl_histo_f)
#     # Down sample histogram
#     # msk = cl_histo_f > 5
#     # cl_histo_f[msk] = 5

#     syn_histo_f = []
#     for i, col_s in enumerate(colors_s):
#         hess_diag = np.histogram2d(mag_s, col_s, bins=bin_edges)[0]
#         # Flatten array
#         syn_histo_f += list(hess_diag.ravel())
#     syn_histo_f = np.array(syn_histo_f)
#     # Down sample histogram
#     # msk = syn_histo_f > 5
#     # syn_histo_f[msk] = 5

#     # return -sum(abs(cluster_dict["hist_down_samp"]-syn_histo_f))

#     # create a mask for where each data set is non-zero
#     m1 = cl_histo_f != 0
#     m2 = syn_histo_f != 0
#     m1_area = m1.sum()
#     m2_area = m2.sum()
#     tot_area = m1_area + m2_area
#     # use a logical and to create a combined map where both datasets are non-zero
#     ovrlp_area = np.logical_and(m1, m2).sum()

#     # # calculate the overlapping density, where 0.5 is the bin width
#     # ol_density = np.abs((cl_histo_f - syn_histo_f) * 0.5)[ol]
#     # # calculate the total overlap percent
#     # h_overlap = ol_density.sum() * 100

#     h_overlap = ovrlp_area / tot_area

#     # ol_density = np.abs((cl_histo_f - cl_histo_f) * 0.5)[ol]
#     # h_overlap_0 = ol_density.sum() * 100
#     # print(h_overlap_0)
#     # breakpoint()

#     # # cl_histo_f = cluster_dict["hist_down_samp"]
#     # mi_cnst = np.clip(cl_histo_f, a_min=0, a_max=1)
#     # # Final chi.
#     # mig_chi = np.sum((cl_histo_f + mi_cnst - syn_histo_f)**2 / (cl_histo_f + 1.))
#     # mig_chi_0 = np.sum((cl_histo_f + mi_cnst)**2 / (cl_histo_f + 1.))
#     # mig_chi_opt = np.sum((cl_histo_f + mi_cnst - cl_histo_f)**2 / (cl_histo_f + 1.))
#     # print(mig_chi_opt, mig_chi_0, mig_chi)

#     # import matplotlib.pyplot as plt
#     # # plt.subplot(121)
#     # y_edges, x_edges = bin_edges #cluster_dict['bin_edges']
#     # for xe in x_edges:
#     #     plt.axvline(xe, c="grey", ls=":")
#     # for ye in y_edges:
#     #     plt.axhline(ye, c="grey", ls=":")
#     # plt.title(h_overlap)
#     # plt.scatter(colors_o[0], mag_o, alpha=.5)
#     # plt.scatter(colors_s[0], mag_s, alpha=.5)
#     # plt.gca().invert_yaxis()

#     # # plt.subplot(122)
#     # # plt.title(h_overlap)
#     # # plt.bar(np.arange(len(cl_histo_f)), cl_histo_f, label='obs', alpha=.5)
#     # # plt.bar(np.arange(len(syn_histo_f)), syn_histo_f, label='syn', alpha=.5)
#     # # plt.legend()
#     # plt.show()

#     return h_overlap


# def mean_dist(cluster_dict, synth_clust):
#     # If synthetic cluster is empty, assign a small likelihood value.
#     if not synth_clust.any():
#         return -1.0e09

#     # mag_o, colors_o = cluster_dict['mag'], cluster_dict['colors']
#     mag0, colors0 = cluster_dict["mag0"], cluster_dict["colors0"]
#     mag_s, colors_s = synth_clust[0], synth_clust[1:]

#     dist = (np.median(mag0) - np.median(mag_s)) ** 2 + (
#         np.median(colors0) - np.median(colors_s)
#     ) ** 2
#     return -dist

#     if len(mag_s) < 5:
#         return -1.0e09
#     import ndtest

#     P_val = ndtest.ks2d2s(mag0, colors0, mag_s, colors_s[0])

#     # import matplotlib.pyplot as plt
#     # plt.title(P_val)
#     # plt.scatter(colors_o0, mag_o, alpha=.5)
#     # plt.scatter(colors_s[0], mag_s, alpha=.5)
#     # plt.gca().invert_yaxis()
#     # plt.show()

#     return P_val


def bins_distance(
    mag_v: np.ndarray, colors_v: list[np.ndarray], synth_clust: np.ndarray
) -> float:
    """Sum of distances to corresponding bins in the Hess diagram. Only applied
    on the first two dimensions (magnitude +  first color)

    :param mag_v: Array of magnitudes.
    :type mag_v: np.ndarray
    :param colors_v: List of arrays of colors.
    :type colors_v: list[np.ndarray]
    :param synth_clust: Synthetic cluster data.
    :type synth_clust: np.ndarray

    :return: Sum of distances.
    :rtype: float
    """
    if not synth_clust.any():
        return 1.0e09

    # Fixed percentiles for magnitude and color
    mpercs = (0.5, 10, 20, 30, 40, 50, 60, 70, 80, 90)
    cpercs = (0.5, 10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95)

    # Evaluate the magnitude and color in the defined percentiles
    perc_mag_o = np.nanpercentile(mag_v, mpercs)
    perc_colors_o = np.nanpercentile(colors_v[0], cpercs)

    # Create a 2-dimensional array of shape: (2, len(mpercs) * len(cpercs))
    pts_o = []
    for pm in perc_mag_o:
        for pc in perc_colors_o:
            pts_o.append([pm, pc])
    pts_o = np.array(pts_o).T

    # Same for the synthetic cluster
    mag_s, colors_s = synth_clust[0], synth_clust[1]
    perc_mag_s = np.nanpercentile(mag_s, mpercs)
    perc_colors_s = np.nanpercentile(colors_s, cpercs)
    pts_s = []
    for pm in perc_mag_s:
        for pc in perc_colors_s:
            pts_s.append([pm, pc])
    pts_s = np.array(pts_s).T

    # Distance (non root squared) between the two arrays
    dist = (pts_s[0] - pts_o[0]) ** 2 + (pts_s[1] - pts_o[1]) ** 2
    # More weight to smaller magnitudes and color (top left of Hess diagram)
    weights = np.linspace(1, 0.05, len(mpercs) * len(cpercs))
    lkl = sum(dist * weights)

    # import matplotlib.pyplot as plt
    # plt.title(f"lkl={lkl:.3f}")
    # plt.scatter(colors_v[0], mag_v, alpha=0.25, c="r", label='obs')
    # plt.scatter(colors_s, mag_s, alpha=0.25, c="b", label='synth')
    # # This shows the positions where the observed Hess diagram is defined by the
    # # percentiles
    # plt.scatter(pts_o[1], pts_o[0], c="r", marker="*")
    # # Same for the synthetic Hess diagram
    # plt.scatter(pts_s[1], pts_s[0], c="b", marker="x")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.show()

    return lkl


def chi_square(
    ranges: list,
    Nbins: list,
    cl_z_idx: np.ndarray,
    cl_histo_f_z: np.ndarray,
    synth_clust: np.ndarray,
) -> float:
    """Calculate the chi-square value.

    This function calculates the chi-square value between the observed and
    synthetic clusters.

    :param ranges: Per-dimension ranges.
    :type ranges: list
    :param Nbins: Per-dimension total number of bins
    :type Nbins: list
    :param cl_z_idx: Index of bins where the number of stars is not 0
    :type cl_z_idx: np.ndarray
    :param cl_histo_f_z: Flattened observed Hess diagram with the empty bins removed
    :type cl_histo_f_z: np.ndarray
    :param synth_clust: Synthetic cluster data.
    :type synth_clust: np.ndarray

    :return: Chi-square value.
    :rtype: float
    """
    # If synthetic cluster is empty, assign a small likelihood value.
    if not synth_clust.any():
        return -1.0e09

    # Obtain histogram for the synthetic cluster.
    mag, colors = synth_clust[0], synth_clust[1:]
    syn_histo_f = []
    for i, col in enumerate(colors):
        hess_diag = histogram2d(
            mag,
            col,
            range=[
                [ranges[0][0], ranges[0][1]],
                [ranges[i + 1][0], ranges[i + 1][1]],
            ],
            bins=[Nbins[0], Nbins[i + 1]],
        )
        # Flatten array
        syn_histo_f += list(hess_diag.ravel())
    syn_histo_f = np.array(syn_histo_f)

    # Remove all bins where n_i = 0 (no observed stars).
    syn_histo_f_z = syn_histo_f[cl_z_idx]

    chisq = ((cl_histo_f_z - syn_histo_f_z) ** 2).sum()
    return chisq
