import numpy as np
from astropy.stats import calculate_bin_edges
from fast_histogram import histogram2d
from scipy.special import loggamma
import matplotlib.pyplot as plt


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
