
import numpy as np
# from scipy.ndimage.filters import gaussian_filter
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# from matplotlib.colors import LinearSegmentedColormap
import matplotlib.offsetbox as offsetbox


def pl_cl_fl_regions(
    gs, plot_style, x_name, y_name, coord, x_min, x_max, y_min, y_max,
        asp_ratio, field_regions_rjct_c, cl_region_rjct_c, flag_no_fl_regs_c):
    """
    Cluster and field regions defined.
    """
    ax = plt.subplot(gs[0:2, 0:2])
    ax.set_aspect(aspect=asp_ratio)
    # Set plot limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    # If RA is used, invert axis.
    if coord == 'deg':
        ax.invert_xaxis()
    # Set axis labels
    plt.xlabel('{} ({})'.format(x_name, coord))
    plt.ylabel('{} ({})'.format(y_name, coord))
    if plot_style == 'asteca':
        ax.grid(which='both')

    # Plot cluster region.
    if len(cl_region_rjct_c) > 0:
        plt.scatter(
            list(zip(*cl_region_rjct_c))[1], list(zip(*cl_region_rjct_c))[2],
            marker='x', c='teal', s=15, lw=.5, edgecolors='none')

    N_flrg = 0
    if not flag_no_fl_regs_c:
        # Stars inside the field regions with rejected errors.
        for i, reg in enumerate(field_regions_rjct_c):
            if reg:
                fl_reg = list(zip(*reg))
                N_flrg += len(fl_reg[0])
                plt.scatter(fl_reg[1], fl_reg[2], marker='x',
                            c='teal', s=15, lw=.5, edgecolors='none')

    ax.set_title(r"$N_{{rjct}}$={} (phot compl)".format(
        len(cl_region_rjct_c) + N_flrg))


def pl_lum_func(gs, plot_style, y_ax, flag_no_fl_regs, lum_func):
    """
    LF of stars in cluster region and outside.
    """
    x_cl, y_cl, x_fl, y_fl, x_all, y_all = lum_func
    ax = plt.subplot(gs[0:2, 2:4])
    ax.set_title("LF after error removal (compl)")
    # Only draw units on axis (ie: 1, 2, 3)
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    if plot_style == 'asteca':
        ax.grid()
    # Set axis labels
    plt.xlabel('$' + y_ax + '$')
    plt.ylabel(r'$N^{\star}/A_{cl}$')

    # All frame.
    plt.step(x_all, y_all, where='post', color='k', lw=2.5, linestyle=':',
             label='Frame (compl)', zorder=6)
    # Cluster region LF (contaminated).
    plt.step(x_cl, y_cl, where='post', color='r', lw=1.,
             label=r'$LF_{cl+fl} \,(r \leq r_{cl})$', zorder=2)
    # Check if field regions were defined.
    if flag_no_fl_regs is not True:
        # Average field regions LF.
        plt.step(x_fl, y_fl, where='post', color='b', lw=1.,
                 label=r'$LF_{fl} \,(\star_{field})$', zorder=3)
        # Cluster region LF - average field regions LF.
        plt.step(x_cl, y_cl - y_fl, where='post', color='g', lw=1.7,
                 label=r'$LF_{cl}$', zorder=4)
        max_y = max(max(y_cl), max(y_fl), max(y_all))
    else:
        max_y = max(max(y_cl), max(y_all))
    # Set plot limits
    x_min, x_max = x_cl[-1] + .3, x_cl[1] - .3
    plt.xlim(x_min, x_max)
    plt.ylim(0., max_y + 0.05 * max_y)

    # Legends.
    leg = plt.legend(fancybox=True, loc='upper right', numpoints=1)
    # Set the alpha value of the legend.
    leg.get_frame().set_alpha(0.7)


def pl_data_rm_perc(
    gs, plot_style, y_ax, phot_analy_compl, phot_data_compl, err_rm_data,
        combined_compl):
    """
    """
    ax = plt.subplot(gs[0:2, 4:6])
    ax.set_title("Percentage of stars kept after each process")
    if plot_style == 'asteca':
        ax.grid()
    # Set axis labels
    plt.xlabel('$' + y_ax + '$')
    plt.ylabel('perc')

    edges, perc_vals = phot_analy_compl
    perc_vals_min = [min(perc_vals)]
    txt = "Photometric analysis completeness"
    plt.step(edges[:-1], perc_vals, where='post', lw=2., linestyle='--',
             label=txt)

    edges, perc_vals, perc_rmvd = phot_data_compl
    perc_vals_min.append(min(perc_vals))
    txt = "Photometric data completeness ({:.1f}% rm)".format(perc_rmvd)
    plt.step(
        edges[:-1], perc_vals, where='post', lw=2., linestyle='--', label=txt)

    edges, perc_vals, perc_rmvd = err_rm_data
    perc_vals_min.append(min(perc_vals))
    txt = "Error removal ({:.1f}% rm)".format(perc_rmvd)
    plt.step(
        edges[:-1], perc_vals, where='post', lw=2., color='teal',
        linestyle='--', label=txt)

    edges, perc_vals, perc_rmvd = combined_compl
    # Remove the extra '1.' value at the beginning (used by the completeness
    # removal function)
    perc_vals = perc_vals[1:]
    # Reverse.
    perc_vals = 1. - perc_vals
    perc_vals_min.append(min(perc_vals))
    txt = "Combined function ({:.1f}% rm)".format(perc_rmvd)
    plt.step(
        edges[:-1], perc_vals, where='post', lw=2., color='r', linestyle='--',
        label=txt)

    # Legends.
    leg = plt.legend(fancybox=True, numpoints=1, loc='lower right')
    # Set the alpha value of the legend.
    leg.get_frame().set_alpha(0.7)

    plt.gca().invert_xaxis()
    plt.ylim(min(.9, min(perc_vals_min)) - .05, 1.05)


def clCMD(
    ax, plot_style, x_min_cmd, x_max_cmd, y_min_cmd, y_max_cmd, x_ax, y_ax,
        xr, yr, xa, ya, n_memb, cl_sz_pt, err_bar, col_idx):
    # Set plot limits
    plt.xlim(x_min_cmd, x_max_cmd)
    plt.ylim(y_min_cmd, y_max_cmd)
    # Set axis labels
    plt.xlabel('$' + x_ax + '$')
    plt.ylabel('$' + y_ax + '$')
    if plot_style == 'asteca':
        ax.grid()
    # Add text box.
    text = r'$N_{{memb}} \approx {}$'.format(n_memb)
    ob = offsetbox.AnchoredText(text, pad=0.2, loc=1)
    ob.patch.set(alpha=0.7)
    ax.add_artist(ob)
    # Plot stars in CMD.
    if yr:
        # Only attempt to plot if any star is stored in the list.
        plt.scatter(xr, yr, marker='x', c='teal', s=12, lw=.5, zorder=2)
    plt.scatter(
        xa, ya, marker='o', c='r', s=cl_sz_pt, lw=0.3, edgecolor='k', zorder=3)
    # If list is not empty, plot error bars at several values.
    x_val, mag_y, xy_err = err_bar
    if x_val:
        plt.errorbar(
            x_val, mag_y, yerr=xy_err[0], xerr=xy_err[col_idx], fmt='k.',
            lw=0.8, ms=0., zorder=4)


def pl_cl_diag(
    gs, plot_style, x_ax0, y_ax, x_min_cmd0, x_max_cmd0, y_min_cmd0,
    y_max_cmd0, x_ax1, x_min_cmd1, x_max_cmd1, y_min_cmd1, y_max_cmd1,
    err_bar_cl0, err_bar_cl1, cl_region_rjct_c, cl_region_c, n_memb,
        cl_sz_pt):
    """
    Cluster's stars CMD diagram (stars inside cluster's radius)
    """
    ax = plt.subplot(gs[2:4, 0:2])
    ax.set_title(
        r"$N_{{accpt}}={}$ , $N_{{rjct}}={}$"
        r" ($r \leq r_{{cl}}$ compl)".format(
            len(cl_region_c), len(cl_region_rjct_c)))
    xr, yr = [], []
    if len(cl_region_rjct_c) > 0:
        xr = list(zip(*list(zip(*cl_region_rjct_c))[5]))[0]
        yr = list(zip(*list(zip(*cl_region_rjct_c))[3]))[0]
    xa = list(zip(*list(zip(*cl_region_c))[5]))[0]
    ya = list(zip(*list(zip(*cl_region_c))[3]))[0]
    # CMD for first color
    col_idx = 1
    clCMD(
        ax, plot_style, x_min_cmd0, x_max_cmd0, y_min_cmd0, y_max_cmd0, x_ax0,
        y_ax, xr, yr, xa, ya, n_memb, cl_sz_pt, err_bar_cl0, col_idx)

    if x_ax1 != '':
        ax = plt.subplot(gs[4:6, 0:2])
        xr = []
        if len(cl_region_rjct_c) > 0:
            xr = list(zip(*list(zip(*cl_region_rjct_c))[5]))[1]
        xa = list(zip(*list(zip(*cl_region_c))[5]))[1]
        # CMD for second color
        col_idx = 2
        clCMD(
            ax, plot_style, x_min_cmd1, x_max_cmd1, y_min_cmd1, y_max_cmd1,
            x_ax1, y_ax, xr, yr, xa, ya, n_memb, cl_sz_pt, err_bar_cl1,
            col_idx)


def hessKDE(
    ax, plot_style, x_ax, y_ax, x_max_cmd, x_min_cmd, y_min_cmd, y_max_cmd,
        cl_col, cl_mag, fr_col, fr_mag):

    # This bandwidth seems to produce nice results.
    bw, Nb = .2, 100

    ax.set_title("Cluster - Field (normalized)")
    plt.xlabel('$' + x_ax + '$')
    plt.ylabel('$' + y_ax + '$')

    xx, yy = np.mgrid[x_min_cmd:x_max_cmd:complex(Nb),
                      y_max_cmd:y_min_cmd:complex(Nb)]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # # Test the impact of the bw value in the final estimate.
    # scott_f = len(cl_col)**(-1. / (2 + 4))
    # for bws in (scott_f / 2., scott_f, scott_f * 2.):
    #     kernel1 = gaussian_kde(np.vstack([cl_col, cl_mag]), bw_method=bws)
    #     kernel2 = gaussian_kde(np.vstack([fr_col, fr_mag]), bw_method=bws)
    #     f1 = np.reshape(kernel1(positions).T, xx.shape)
    #     f2 = np.reshape(kernel2(positions).T, xx.shape)
    #     diff = f1 - f2
    #     diff = np.clip(diff, 1e-9, np.inf)
    #     cell = ((x_max_cmd - x_min_cmd) * (y_min_cmd - y_max_cmd)) / Nb**2
    #     print(np.sum(diff * cell))

    # Cluster data
    values1 = np.vstack([cl_col, cl_mag])
    kernel1 = gaussian_kde(values1, bw_method=bw)
    f1 = np.reshape(kernel1(positions).T, xx.shape)

    # Field regions data
    values2 = np.vstack([fr_col, fr_mag])
    kernel2 = gaussian_kde(values2, bw_method=bw)
    f2 = np.reshape(kernel2(positions).T, xx.shape)

    # Cluster - field regions
    diff = f1 - f2
    # Clip negative values.
    diff = np.clip(diff, 1e-9, np.inf)

    # Area of the 2D cell.
    cell = ((x_max_cmd - x_min_cmd) * (y_min_cmd - y_max_cmd)) / Nb**2
    # Integral of the cluster-field KDE. This value strongly depends on
    # the selected 'bw' value, so it is not really a stable indicator of
    # field contamination in the cluster region.
    integ = np.sum(diff * cell)
    # Add text box.
    text = r'$\int \Delta KDE_{{[cl-fr]}} \approx {:.2f}$'.format(integ)
    ob = offsetbox.AnchoredText(text, pad=0.2, loc=1)
    ob.patch.set(alpha=0.7)
    ax.add_artist(ob)

    ax.contourf(xx, yy, diff, cmap='Blues')
    ax.contour(xx, yy, diff, colors='k', linewidths=.5)
    # ax.clabel(CS, inline=1, fontsize=10)

    if plot_style == 'asteca':
        ax.grid()
    plt.gca().invert_yaxis()


def pl_hess_cmd(
    gs, plot_style, x_ax0, x_ax1, y_ax, x_max_cmd0, x_min_cmd0, y_min_cmd0,
    y_max_cmd0, x_max_cmd1, x_min_cmd1, y_min_cmd1, y_max_cmd1, stars_f_acpt,
        cl_region_c):
    """
    Hess diagram for CMD of field vs cluster region.
    """
    if stars_f_acpt[0]:
        ax = plt.subplot(gs[2:4, 2:4])
        cl_col = list(zip(*list(zip(*cl_region_c))[5]))[0]
        cl_mag = list(zip(*list(zip(*cl_region_c))[3]))[0]
        fr_col, fr_mag = stars_f_acpt[1], stars_f_acpt[0]

        hessKDE(
            ax, plot_style, x_ax0, y_ax, x_max_cmd0, x_min_cmd0, y_min_cmd0,
            y_max_cmd0, cl_col, cl_mag, fr_col, fr_mag)

        if stars_f_acpt[2]:
            cl_col = list(zip(*list(zip(*cl_region_c))[5]))[1]
            fr_col = stars_f_acpt[2]
            ax = plt.subplot(gs[4:6, 2:4])
            hessKDE(
                ax, plot_style, x_ax1, y_ax, x_max_cmd1, x_min_cmd1,
                y_min_cmd1, y_max_cmd1, cl_col, cl_mag, fr_col, fr_mag)


def flCMD(
    ax, plot_style, x_min_cmd, x_max_cmd, y_min_cmd, y_max_cmd, x_ax, y_ax,
    N_fr, x_fr_rject, y_fr_rject, x_fr_accpt, y_fr_accpt, f_sz_pt, err_bar,
        col_idx):
    """
    Field stars CMD diagram.
    """
    # Set plot limits
    plt.xlim(x_min_cmd, x_max_cmd)
    plt.ylim(y_min_cmd, y_max_cmd)
    # Set axis labels
    plt.xlabel('$' + x_ax + '$')
    plt.ylabel('$' + y_ax + '$')
    if plot_style == 'asteca':
        ax.grid()
    # Plot accepted/rejected stars within the field regions defined.
    if x_fr_rject:
        plt.scatter(x_fr_rject, y_fr_rject, marker='x',
                    c='teal', s=15, lw=.5, zorder=2)
    if x_fr_accpt:
        plt.scatter(x_fr_accpt, y_fr_accpt, marker='o', c='b',
                    s=f_sz_pt, lw=0.3, edgecolor='k', zorder=3)
        n_field = int(len(x_fr_accpt) / float(N_fr))
        # Add text box.
        text = r'$N_{{field}} \approx {}$'.format(n_field)
        ob = offsetbox.AnchoredText(text, pad=0.2, loc=1)
        ob.patch.set(alpha=0.7)
        ax.add_artist(ob)
    # If list is not empty, plot error bars at several values.
    x_val, mag_y, xy_err = err_bar
    if x_val:
        plt.errorbar(
            x_val, mag_y, yerr=xy_err[0], xerr=xy_err[col_idx], fmt='k.',
            lw=0.8, ms=0., zorder=4)


def pl_fl_diag(
    gs, plot_style, x_ax0, y_ax, x_min_cmd0, x_max_cmd0, y_min_cmd0,
    y_max_cmd0, x_ax1, x_min_cmd1, x_max_cmd1, y_min_cmd1, y_max_cmd1,
    field_regions_c, stars_f_rjct, stars_f_acpt, f_sz_pt, err_bar_fl0,
        err_bar_fl1):
    """
    Field stars CMD diagram.
    """
    ax = plt.subplot(gs[2:4, 4:6])

    N_fr = len(field_regions_c)
    x_fr_rject, y_fr_rject = stars_f_rjct[1], stars_f_rjct[0]
    x_fr_accpt, y_fr_accpt = stars_f_acpt[1], stars_f_acpt[0]

    ax.set_title(r"$N_{{accpt}}={}$ , $N_{{rjct}}={}$ (fields compl)".format(
        len(x_fr_accpt), len(x_fr_rject)))
    # CMD for first color
    col_idx = 1
    flCMD(
        ax, plot_style, x_min_cmd0, x_max_cmd0, y_min_cmd0, y_max_cmd0, x_ax0,
        y_ax, N_fr, x_fr_rject, y_fr_rject, x_fr_accpt, y_fr_accpt, f_sz_pt,
        err_bar_fl0, col_idx)

    if x_ax1 != '':
        ax = plt.subplot(gs[4:6, 4:6])
        x_fr_rject, x_fr_accpt = stars_f_rjct[2], stars_f_acpt[2]
        # CMD for second color
        col_idx = 2
        flCMD(
            ax, plot_style, x_min_cmd1, x_max_cmd1, y_min_cmd1, y_max_cmd1,
            x_ax1, y_ax, N_fr, x_fr_rject, y_fr_rject, x_fr_accpt, y_fr_accpt,
            f_sz_pt, err_bar_fl1, col_idx)


def plot(N, *args):
    """
    Handle each plot separately.
    """

    plt_map = {
        0: [pl_cl_fl_regions, 'cluster + field regions rejected stars'],
        1: [pl_cl_diag, 'cluster region photometric diagram'],
        2: [pl_hess_cmd, 'Hess CMD'],
        3: [pl_fl_diag, 'field regions photometric diagram'],
        4: [pl_lum_func, 'luminosity function'],
        5: [pl_data_rm_perc, 'error removal percentage']
    }

    fxn = plt_map.get(N, None)[0]
    if fxn is None:
        raise ValueError("  ERROR: there is no plot {}.".format(N))

    try:
        fxn(*args)
    except Exception:
        import traceback
        print(traceback.format_exc())
        print("  WARNING: error when plotting {}".format(plt_map.get(N)[1]))
