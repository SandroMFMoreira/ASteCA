import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .cluster import Cluster
from .synthetic import Synthetic


def radec(cluster: Cluster, ax: Axes, ) -> Axes:
    """Generate a (RA, DEC) plot.

    :param cluster: :py:class:`Cluster <asteca.cluster.Cluster>` object with the
        loaded data for the observed cluster
    :type cluster: Cluster
    :param ax: Matplotlib axis where to draw the plot
    :type ax: Axes

    :return: Matplotlib axis object
    :rtype: Axes
    """
    ra = cluster.ra_v
    dec = cluster.dec_v
    mag = cluster.mag_v

    msk = ~np.isnan(mag)
    ra = ra[msk]
    dec = dec[msk]
    mag = mag[msk]

    # Mag to flux
    sizes = 10 ** (mag / -2.5)
    # Normalize to 1
    sizes /= sizes.max()
    # Set min, max
    sizes = 1 + 75 * sizes

    plt.scatter(ra, dec, s=sizes, c="k", alpha=0.7)
    plt.xlabel("RA")
    plt.ylabel("DEC")
    plt.gca().invert_xaxis()

    return ax


def cluster(
        cluster: Cluster,
        ax: Axes,
        col_plot: str = "cmd",  # Default to 'cmd'
        color_idx: int = 0,
        binar_probs: np.ndarray | None = None,
        prob_binar_cut: float = 0.5,
) -> Axes:
    """Generate a color-magnitude plot or color-color diagram based on the specified parameters.

    :param cluster: :py:class:`Cluster <asteca.cluster.Cluster>` object with the
        loaded data for the observed cluster
    :type cluster: Cluster
    :param ax: Matplotlib axis where to draw the plot
    :type ax: Axes
    :param col_plot: Type of plot to generate. Options are:
        - "cmd" (default): Color-magnitude diagram.
        - "ccd": Color-color diagram.
    :type col_plot: str
    :param color_idx: Index of the color to plot. If ``0`` (default), plot the
        first color. If ``1`` plot the second color. Defaults to ``0``
    :type color_idx: int
    :param binar_probs: Array with probabilities of being a binary system for each
        observed star; defaults to ``None``
    :type binar_probs: np.ndarray | None
    :param prob_binar_cut: Probabilities value that separates single systems from
        binary systems; defaults to ``0.5``
    :type prob_binar_cut: float

    :raises ValueError: If ``color_idx`` is not ``0`` or ``1``

    :return: Matplotlib axis object
    :rtype: Axes
    """

    if color_idx > 1:
        raise ValueError(f"Wrong 'color_idx' value ({color_idx}), should be one of: [0, 1]")

    # Plot Color-Magnitude Diagram (CMD)
    def plot_cmd():
        scatter_args = {
            'c': 'green',
            'alpha': 0.5,
            'label': f"Observed, N={len(cluster.mag_v)}"
        }

        if binar_probs is None:
            ax.scatter(cluster.colors_v[color_idx], cluster.mag_v, **scatter_args)
        else:
            # Prepare the data for binary classification
            msk_binar = binar_probs > prob_binar_cut
            ax.scatter(cluster.colors_v[color_idx][~msk_binar], cluster.mag_v[~msk_binar], c="grey", marker="o",
                       alpha=0.5,
                       label=f"Observed (single), N={len(cluster.mag_v[~msk_binar])}")
            ax.scatter(cluster.colors_v[color_idx][msk_binar], cluster.mag_v[msk_binar], c=binar_probs[msk_binar],
                       marker="s",
                       alpha=0.5, label=f"Observed (binary), N={len(cluster.mag_v[msk_binar])}")

        ax.set_ylim(max(cluster.mag_v) + 0.5, min(cluster.mag_v) - 1)
        ax.set_xlabel(cluster.color if cluster.color else "Color1")
        ax.set_ylabel(cluster.magnitude)
        ax.legend()

    # Plot Color-Color Diagram (CCD)
    def plot_ccd():
        x = cluster.colors_v[color_idx]
        y = cluster.colors_v[color_idx * 1 - 1]  # Handles color index swap

        # Set axis labels
        if color_idx == 0:
            ax.set_xlabel(cluster.color if cluster.color else "Color1")
            ax.set_ylabel(cluster.color2 if cluster.color2 else "Color2")
        else:
            ax.set_xlabel(cluster.color2 if cluster.color2 else "Color2")
            ax.set_ylabel(cluster.color if cluster.color else "Color1")

        if binar_probs is None:
            # Plot all stars as a single group
            ax.scatter(x, y, c="blue", alpha=0.5, label="Observed")
        else:
            # Classify single vs. binary systems
            msk_binar = binar_probs > prob_binar_cut
            ax.scatter(x[~msk_binar], y[~msk_binar], c="grey", marker="o", alpha=0.5,
                       label=f"Observed (single), N={len(x[~msk_binar])}")
            ax.scatter(x[msk_binar], y[msk_binar], c=binar_probs[msk_binar], marker="s", alpha=0.5,
                       label=f"Observed (binary), N={len(x[msk_binar])}")

        ax.legend()
        ax.invert_yaxis()

    # Handle plot types
    if col_plot == "cmd":
        plot_cmd()  # Plot only CMD
        return ax

    elif col_plot == "ccd":
        if len(cluster.colors_v) < 2:
            raise ValueError("Cannot generate CCD plot: At least two colors are required, but only one was provided.")
        plot_ccd()  # Plot only CCD
        return ax

    else:
        raise ValueError(f"Invalid value for 'col_plot': '{col_plot}'. Must be 'cmd' or 'ccd'.")


def synthetic(
    synth: Synthetic,
    ax: Axes,
    fit_params: dict,
    isoch_arr: np.ndarray | None = None,
    col_plot: str = "cmd",  # Default to CMD
    color_idx: int = 0,
) -> Axes:
    """Generate a color-magnitude plot or color-color diagram for a synthetic cluster.

    The synthetic cluster is generated using the fundamental parameter values
    given in the ``fit_params`` dictionary.

    :param synth: :py:class:`Synthetic <asteca.synthetic.Synthetic>` object with the
        data required to generate synthetic clusters
    :type synth: Synthetic
    :param ax: Matplotlib axis where to draw the plot
    :type ax: Axes
    :param fit_params: Dictionary with the values for the fundamental parameters
        that were **not** included in the ``fix_params`` dictionary when the
        :py:class:`Synthetic` object was calibrated
        (:py:meth:`calibrate` method).
    :type fit_params: dict
    :param isoch_arr: Array with the isochrone data to plot; defaults to ``None``
    :type isoch_arr: np.ndarray | None
    :param col_plot: Type of plot to generate. Options are:
        - "cmd" (default): Color-Magnitude Diagram.
        - "ccd": Color-Color Diagram.
    :type col_plot: str
    :param color_idx: Index of the color to plot. If ``0`` (default), plot the
        first color. If ``1`` plot the second color. Defaults to ``0``
    :type color_idx: int

    :raises ValueError: If ``color_idx`` is not ``0`` or ``1`` or if ``col_plot`` is invalid.

    :return: Matplotlib axis object
    :rtype: Axes
    """

    if color_idx > 1:
        raise ValueError(
            f"Wrong 'color_idx' value ({color_idx}), should be one of: [0, 1]"
        )
    if col_plot not in ["cmd", "ccd"]:
        raise ValueError(f"Invalid 'col_plot' value ({col_plot}), must be 'cmd' or 'ccd'.")

    # Generate synthetic cluster
    synth_clust = synth.generate(fit_params, full_arr_flag=True)

    if synth.binar_flag:
        binar_idx = ~np.isnan(synth_clust[-1])
    else:
        binar_idx = np.full(synth_clust.shape[1], False)

    # CMD (Color-Magnitude Diagram)
    if col_plot == "cmd":
        x_synth = synth_clust[1 + color_idx]  # Color index 0 or 1
        y_synth = synth_clust[0]  # Magnitude

        # Plot isochrone if provided
        if isoch_arr is not None:
            ax.plot(isoch_arr[color_idx + 1], isoch_arr[0], c="k")

        ax.set_ylabel(synth.isochs.magnitude)
        c1, c2 = synth.isochs.color
        if color_idx == 1:
            if synth.isochs.color2 is None:
                raise ValueError("No second color available")
            c1, c2 = synth.isochs.color2
        ax.set_xlabel(f"{c1}-{c2}")

    # CCD (Color-Color Diagram)
    elif col_plot == "ccd":
        if synth.isochs.color2 is None:
            raise ValueError("Cannot generate CCD plot: A second color is required but not available.")

        x_synth = synth_clust[1]  # First color
        y_synth = synth_clust[2]  # Second color

        # Set axis labels
        if color_idx == 0:
            cx1, cx2 = synth.isochs.color
            cy1, cy2 = synth.isochs.color2
            ax.set_xlabel(f"{cx1}-{cx2}")
            ax.set_ylabel(f"{cy1}-{cy2}")
        else:
            cx1, cx2 = synth.isochs.color2
            cy1, cy2 = synth.isochs.color
            ax.set_xlabel(f"{cx1}-{cx2}")
            ax.set_ylabel(f"{cy1}-{cy2}")

        # Plot isochrone if provided
        if isoch_arr is not None:
            c1 = isoch_arr[color_idx + 1]
            c2 = isoch_arr[color_idx*-1 + 2]
            ax.plot(c1, c2, c="k")

    # Single synthetic systems
    ax.scatter(
        x_synth[~binar_idx],
        y_synth[~binar_idx],
        marker="^",
        c="#519ddb",
        alpha=0.5,
        label=f"Synthetic (single), N={len(x_synth[~binar_idx])}",
    )

    # Binary synthetic systems
    ax.scatter(
        x_synth[binar_idx],
        y_synth[binar_idx],
        marker="v",
        c="#F34C4C",
        alpha=0.5,
        label=f"Synthetic (binary), N={len(x_synth[binar_idx])}",
    )

    ax.legend()

    ax.invert_yaxis()

    return ax


def get_isochrone(
    synth: Synthetic,
    fit_params: dict,
    color_idx: int = 0,
) -> np.ndarray:
    """Generate an isochrone for plotting.

    The isochrone is generated using the fundamental parameter values
    given in the ``fit_params`` dictionary.

    :param synth: :py:class:`Synthetic <asteca.synthetic.Synthetic>` object with the
        data required to generate synthetic clusters
    :type synth: Synthetic
    :param fit_params: Dictionary with the values for the fundamental parameters
        that were **not** included in the ``fix_params`` dictionary when the
        :py:class:`Synthetic` object was calibrated (:py:meth:`calibrate` method).
    :type fit_params: dict
    :param color_idx: Index of the color to plot. If ``0`` (default), plot the
        first color. If ``1`` plot the second color. Defaults to ``0``
    :type color_idx: int

    :raises ValueError: If either parameter (met, age) is outside of allowed range

    :return: Array with the isochrone data to plot
    :rtype: np.ndarray
    """
    # Generate displaced isochrone
    fit_params_copy = dict(fit_params)

    # Check isochrones ranges
    for par in ("met", "loga"):
        try:
            pmin, pmax = min(synth.met_age_dict[par]), max(synth.met_age_dict[par])
            if fit_params_copy[par] < pmin or fit_params_copy[par] > pmax:
                raise ValueError(f"Parameter '{par}' out of range: [{pmin} - {pmax}]")
        except KeyError:
            pass

    # Generate physical synthetic cluster to extract the max mass
    isochrone_full = synth.generate(fit_params_copy, full_arr_flag=True)
    # Extract max mass
    max_mass = isochrone_full[synth.m_ini_idx].max()

    # Generate displaced isochrone
    fit_params_copy["DR"] = 0.0
    isochrone = synth.generate(fit_params_copy, plot_flag=True)

    # Apply max mass filter to isochrone
    msk = isochrone[synth.m_ini_idx] < max_mass

    # Generate proper array for plotting
    isochrone = np.array(isochrone[:3])[:, msk]

    return isochrone
