
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d


def traceplots(
    par_name, gs, best_sol, min_max_p, method_args, trace, varIdxs,
        post_trace, pre_trace):
    """
    Parameter sampler chain.
    """
    plot_dict = {
        'metal': [6, 10, 0, 1, 0], 'age': [6, 10, 1, 2, 1],
        'beta': [6, 10, 2, 3, 2], 'ext': [6, 10, 3, 4, 3],
        'dr': [6, 10, 4, 5, 4], 'rv': [6, 10, 5, 6, 5],
        'dist': [6, 10, 6, 7, 6]
    }

    labels = [r'$z$', r'$\log(age)$', r'$\beta$', r'$E_{{(B-V)}}$', r'$DR$',
              r'$R_v$', r'$(m-M)_o$']

    gs_x1, gs_x2, gs_y1, gs_y2, cp = plot_dict[par_name]
    ax = plt.subplot(gs[gs_y1:gs_y2, gs_x1:gs_x2])

    if cp == 0:
        ax.set_title(
            r"Chain with the closest $\tau$ to the median")
    if cp == 6:
        plt.xlabel("steps")
    else:
        ax.tick_params(labelbottom=False)
    plt.ylabel(labels[cp])

    # DEPRECATED May 2020
    # if best_fit_algor in ('ptemcee', 'emcee'):
    acorr_t, med_at_c, mcmc_ess = method_args
    N_pre = pre_trace.shape[-1]
    N_tot = N_pre + post_trace.shape[-1]

    ax.set_xlim(0, N_tot)

    if cp in varIdxs:
        c_model = varIdxs.index(cp)

        if pre_trace is not None:
            # Chains with median Tau
            # Burn-in stage in MCMC
            plt.plot(
                pre_trace[c_model][med_at_c[c_model]], c='grey',
                lw=.5, alpha=0.5)

        # DEPRECATED May 2020
        # if best_fit_algor in ('ptemcee'):
        # Post burn-in in MCMC
        post_trace_plot = post_trace[c_model][med_at_c[c_model]]
        # Filtered mean of all chains.
        N = post_trace.shape[-1]
        xavr = uniform_filter1d(
            np.mean(post_trace[c_model], axis=0), int(.02 * N))
        plt.plot(np.arange(N_pre, N_tot), xavr, c='g')

        ax.set_title(
            (r"$\hat{{\tau}}_{{c}}={:.0f}\;(\hat{{n}}_{{eff}}="
             "{:.0f})$").format(acorr_t[c_model], mcmc_ess[c_model]))

        # elif best_fit_algor == 'boot+GA':
        #     post_trace_plot = post_trace[c_model]

        plt.plot(np.arange(N_pre, N_tot), post_trace_plot, c='k', lw=.8,
                 ls='-', alpha=0.5)

        # Mean
        plt.axhline(
            y=float(best_sol[cp]), linestyle='--', color='blue', zorder=4)
        #  16th and 84th percentiles (1 sigma) around median.
        ph = np.percentile(trace[c_model], 84)
        pl = np.percentile(trace[c_model], 16)
        plt.axhline(y=ph, lw=2, linestyle=':', color='orange', zorder=4)
        plt.axhline(y=pl, lw=2, linestyle=':', color='orange', zorder=4)
        # plt.axhline(
        #     y=float(cp_r[cp]), color='k', ls='--', lw=1.2, zorder=4,
        #     label=r"$\tau={:.0f}\;(\hat{{n}}_{{eff}}={:.0f})$".format(
        #         acorr_t[c_model], mcmc_ess[c_model]))

        ax.set_ylim(min_max_p[cp][0], min_max_p[cp][1])
        # ax.legend(fontsize='small', loc=0, handlelength=0.)


def plot(N, *args):
    """
    Handle each plot separately.
    """
    plt_map = {
        0: [traceplots, args[0] + ' sampler chain']
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
