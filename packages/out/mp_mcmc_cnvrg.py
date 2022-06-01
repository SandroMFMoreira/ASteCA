
import numpy as np
import matplotlib.pyplot as plt


def pl_betas(gs, Tmax, N_steps, betas_pt):
    """
    Evolution of Temp swaps AFs.
    """
    ax = plt.subplot(gs[0:2, 0:2])
    Nt = len(betas_pt) - 1
    ax.set_title(r"$T_{{max}}={},\,N_{{temps}}={}$".format(Tmax, Nt + 1))
    for i, y in enumerate(betas_pt):
        if i == 0:
            lbl_ls = ("Cold", '--', 1.5, 4)
        elif i == Nt:
            lbl_ls = ("Hot", ':', 1.5, 4)
        else:
            lbl_ls = (None, '-', .5, 1)
        ax.plot(
            N_steps, y[:len(N_steps)], label=lbl_ls[0], ls=lbl_ls[1],
            lw=lbl_ls[2], zorder=lbl_ls[3])
    plt.xlabel("steps")
    plt.ylabel(r"$\beta\,(1/T)$")
    ax.legend(loc=0)


def pl_Tswaps(gs, N_steps, tswaps_afs):
    """
    Evolution of Temp swaps AFs.
    """
    ax = plt.subplot(gs[0:2, 2:4])
    Nt = len(tswaps_afs) - 1
    for i, y in enumerate(tswaps_afs):
        if i == 0:
            lbl_ls = ("Cold", '--', 1.5, 4)
        elif i == Nt:
            lbl_ls = ("Hot", ':', 1.5, 4)
        else:
            lbl_ls = (None, '-', .5, 1)
        ax.plot(
            N_steps, y, label=lbl_ls[0], ls=lbl_ls[1], lw=lbl_ls[2],
            zorder=lbl_ls[3])
    plt.xlabel("steps")
    plt.ylabel("Tswaps AF")
    ax.legend(loc=0)


def pl_MAF(gs, N_steps, maf_steps):
    """
    Evolution of MAF values.
    """
    ax = plt.subplot(gs[0:2, 4:6])
    ax.set_title(
        r"$MAF_{{[T=1]}}={:.3f}$".format(maf_steps[0][-1]))
    # x, y = list(zip(*maf_steps))
    # x, y_replicas = maf_steps
    Nt = len(maf_steps) - 1
    for i, y in enumerate(maf_steps):
        if i == 0:
            lbl_ls = ("Cold", '--', 1.5, 4)
        elif i == Nt:
            lbl_ls = ("Hot", ':', 1.5, 4)
        else:
            lbl_ls = (None, '-', .5, 1)
        ax.plot(
            N_steps, y, label=lbl_ls[0], ls=lbl_ls[1], lw=lbl_ls[2],
            zorder=lbl_ls[3])
    ax.legend(loc=0)  # , handlelength=0.)

    # DEPRECATED May 2020
    # elif algor == 'emcee':
    #     ax.set_title(r"$MAF={:.3f}$".format(maf_steps[-1]))
    #     ax.plot(N_steps, maf_steps, lw=1.5)

    plt.xlabel("steps")
    plt.ylabel("MAF")
    plt.axhline(y=.25, color='grey', ls=':', lw=1.2, zorder=4)
    plt.axhline(y=.5, color='grey', ls=':', lw=1.2, zorder=4)


def pl_MAP_lkl(gs, N_steps, lkl_mean_steps, lkl_steps):
    """
    Evolution of MAP likelihood values.
    """
    ax = plt.subplot(gs[2:4, 0:2])
    plt.title("Cold chains")
    ax.plot(N_steps, lkl_steps, label="Best")
    ymin, ymax = ax.get_ylim()

    ax.plot(N_steps, lkl_mean_steps, label="Mean")
    plt.xlabel("steps")
    plt.ylabel("Likelihood")
    ax.legend(loc=0)  # , handlelength=0.)
    plt.ylim(ymin, ymax)


def pl_tau_histo(gs, all_taus):
    """
    """
    ax = plt.subplot(gs[2:4, 2:4])
    plt.title(r"$\tau$ for all chains and parameters (N={})".format(
        len(all_taus)))
    plt.hist(np.nan_to_num(all_taus), bins=20, color='#64B1D7')
    plt.axvline(
        np.nanmean(all_taus), color="r",
        label=r"$\hat{{\tau}}_{{c\,,\,p}}$={:.0f}".format(
            np.nanmean(all_taus)))
    plt.xlabel(r"$\tau$ (post burn-in)")
    plt.ylabel(r"$N$")
    ax.legend(loc=0)


def pl_tau(gs, N_steps, tau_autocorr):
    """
    Tau vs steps plot.
    """
    ax = plt.subplot(gs[2:4, 4:6])
    plt.title("Mean across chains and parameters")
    plt.xlabel("steps")
    plt.ylabel(r"$\hat{\tau}_{{c\,|\,p}}$")

    N_steps, taus = tau_autocorr
    plt.plot(N_steps, N_steps / 100., "--g", label="N/100")
    plt.plot(N_steps, N_steps / 500., "--b", label="N/500")
    plt.plot(N_steps, N_steps / 1000., "--r", label="N/1000")
    plt.plot(N_steps, taus)
    # plt.xlim(0, max(N_steps))
    try:
        plt.ylim(
            0, np.nanmax(taus) + 0.1 * (np.nanmax(taus) - np.nanmin(taus)))
    except ValueError:
        print("  WARNING: no mean autocorrelation values to plot")
    ax.legend(loc=0)


def pl_lags(gs, varIdxs, acorr_function, par_list):
    """
    lags plot.
    """
    ax = plt.subplot(gs[4:6, 0:2])
    plt.title("Autocorrelation function")
    plt.xlabel("Lag")
    plt.ylabel("ACF")

    for i, par_name in enumerate(par_list):
        if i in varIdxs:
            c_model = varIdxs.index(i)
            p = acorr_function[c_model]
            plt.plot(
                range(len(p)), p, lw=.8, alpha=0.5,
                label="{}".format(par_name))
            xmax = int(.25 * len(p))
    ax.legend(loc=0)
    plt.xlim(-1, xmax)
    # plt.plot((lag_zero, lag_zero), (0, .5), ls='--', c='k')
    # ax.text(lag_zero, .52, "{}".format(lag_zero))


def pl_GW(gs, varIdxs, geweke_z, par_list):
    """
    Geweke plot.
    """
    ax = plt.subplot(gs[4:6, 2:4])
    ax.set_title("Geweke")
    plt.xlabel("First iteration in segment")
    plt.ylabel("z-score")
    plt.axhline(y=2., color='grey', ls=':', lw=1.2, zorder=4)
    plt.axhline(y=-2., color='grey', ls=':', lw=1.2, zorder=4)

    ymin, ymax = [], []
    for i, par_name in enumerate(par_list):
        if i in varIdxs:
            c_model = varIdxs.index(i)
            p = geweke_z[c_model]
            idx, zscore = list(zip(*p))
            plt.plot(
                idx, zscore, ls="-.", linewidth=.7,
                label="{}".format(par_name))
            ymin.append(np.nanmin(zscore))
            ymax.append(np.nanmax(zscore))

    if ymin and ymax:
        ax.set_ylim(max(-2.1, min(ymin)), min(2.1, max(ymax)))
    ax.legend(loc=0)


def pl_mESS(dummy, gs, mESS, minESS, minESS_epsilon):
    """
    mESS plot.
    """
    ax = plt.subplot(gs[4:6, 6:8])
    plt.xlabel(r"$CI\;(=1-\alpha)$")
    plt.ylabel(r"$\epsilon$")
    ax.set_xlim(.01, 1.01)

    plt.plot(
        1. - np.array(minESS_epsilon[0]), minESS_epsilon[1],
        label="minESS ({:.0f})".format(minESS))
    plt.plot(
        1. - np.array(minESS_epsilon[0]), minESS_epsilon[2],
        label="mESS ({:.0f})".format(mESS))
    ax.legend(loc=0)


def plot(N, *args):
    """
    Handle each plot separately.
    """
    plt_map = {
        0: [pl_MAP_lkl, ' MAP likelihood values'],
        1: [pl_MAF, ' MAF vs steps'],
        2: [pl_betas, ' Betas vs steps'],
        3: [pl_Tswaps, ' Tswaps AFs vs steps'],
        4: [pl_tau, ' Tau evolution'],
        # 5: [pl_mESS, args[0]],
        5: [pl_lags, ' Lags plot'],
        6: [pl_GW, ' Geweke plot'],
        7: [pl_tau_histo, ' Tau histogram']
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
