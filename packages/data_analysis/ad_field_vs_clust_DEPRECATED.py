
from .. import update_progress
import numpy as np
from scipy.stats import anderson_ksamp, gaussian_kde
from scipy.integrate import quad
import warnings


def main(clp, cld_c, ad_runs, flag_make_plot, **kwargs):
    """

    AD test for k-samples: "tests the null hypothesis that k-samples are drawn
    from the same population"

    We want to REJECT the null hypothesis to state that the cluster is
    reasonably different from the field region.

    Significance level: probability of rejecting the null hypothesis when it is
    true (Type I Error). This is: the probability of being wrong saying that
    the samples are drawn from different populations. Equivalent: the
    probability that both samples come from the same population.

    If AD test value > 4 then I can say that the samples come from different
    populations (reject null hypothesis). Ideally I want:

    * large AD values for cluster vs field regions
    * small AD values for field vs field regions
    """

    # Skip test if < 10 members are found within the cluster's radius.
    flag_few_members = False if len(clp['cl_region_c']) > 10 else True

    flag_ad_test = False
    ad_cl, ad_fr, pv_cl, pv_fr = [[[], []] for _ in range(4)]
    ad_cl_fr_p, ad_cl_fr_pk = [], []

    # Check if test is to be applied or skipped. Check if field regions
    # where found.
    if ad_runs <= 0 or 'B3' not in flag_make_plot:
        print("Skipping field vs cluster A-D test")

    elif clp['flag_no_fl_regs_c']:
        print("No field regions. Skipping field vs cluster A-D test")

    elif flag_few_members:
        print("  WARNING: < 10 stars in cluster region"
              "  Skipping field vs cluster A-D test")

    else:
        print("    A-D test ({})".format(ad_runs))
        flag_ad_test = True

        run_total = 2. * int(ad_runs * len(clp['field_regions_c']))
        runs = 0
        # Run first only for photometric data, and then for Plx+PM data (if it
        # exists)
        for i in range(2):
            for run_num in range(ad_runs):

                data_cl = dataExtract(clp['cl_region_c'], i)
                # Field regions
                data_fr = []
                for fr in clp['field_regions_c']:
                    data_fr.append(dataExtract(fr, i))

                # Compare to each defined field region.
                for f_idx, data_fl in enumerate(data_fr):

                    ad_pv = ADtest(data_cl, data_fl)
                    ad_cl[i] += list(ad_pv[0])
                    pv_cl[i] += list(ad_pv[1])

                    # Compare the field region used above with all the
                    # remaining field regions. This results in [N*(N-1)/2]
                    # combinations ("handshakes") of field vs field.
                    for data_fl2 in data_fr[(f_idx + 1):]:
                        ad_pv = ADtest(data_fl, data_fl2)
                        ad_fr[i] += list(ad_pv[0])
                        pv_fr[i] += list(ad_pv[1])

                    runs += 1
                update_progress.updt(run_total, runs)

        # len(pv_cl[0]) = AD runs * N phot dims * N field regs
        # len(pv_fr[0]) = AD runs * N phot dims * [N*(N-1)/2] field regs
        ad_cl_fr_p = kdeplot(pv_cl[0], pv_fr[0], 'phot')
        ad_cl_fr_p = [len(pv_cl[0]), len(pv_fr[0])] + ad_cl_fr_p

        ad_cl_fr_pk = kdeplot(pv_cl[1], pv_fr[1], 'Plx+PM')
        ad_cl_fr_pk = [len(pv_cl[1]), len(pv_fr[1])] + ad_cl_fr_pk

    clp.update({
        'flag_ad_test': flag_ad_test, 'ad_cl': ad_cl, 'ad_fr': ad_fr,
        'ad_cl_fr_p': ad_cl_fr_p, 'ad_cl_fr_pk': ad_cl_fr_pk})
    return clp


def dataExtract(region, idx):
    """
    """
    def photData():
        # Main magnitude. Must have shape (1, N)
        mags = np.array(list(zip(*list(zip(*region))[3])))
        e_mag = np.array(list(zip(*list(zip(*region))[4])))
        mags = normErr(mags, e_mag)

        # One or two colors
        cols = np.array(list(zip(*list(zip(*region))[5])))
        e_col = np.array(list(zip(*list(zip(*region))[6])))
        c_err = []
        for i, c in enumerate(cols):
            c_err.append(normErr(c, e_col[i]))
        cols = np.array(c_err)
        return np.concatenate([mags, cols])

    def kinData():
        # Plx + pm_ra + pm_dec
        kins = np.array(list(zip(*list(zip(*region))[7])))[:3]
        e_kin = np.array(list(zip(*list(zip(*region))[8])))[:3]
        k_err = []
        for i, k in enumerate(kins):
            # Only process if any star contains at least one not 'nan'
            # data.
            if np.any(~np.isnan(k)):
                k_err.append(normErr(k, e_kin[i]))
        return k_err

    if idx == 0:
        data_all = photData()

    elif idx == 1:
        k_err = kinData()
        if k_err:
            data_all = np.array(k_err)
        else:
            # No valid Plx and/or PM data found
            data_all = np.array([[np.nan] * 2, [np.nan] * 2])

    return data_all


def normErr(x, e_x):
    # Randomly move mag and color through a Gaussian function.
    return x + np.random.normal(0, 1, len(x)) * e_x


def ADtest(data_x, data_y):
    """
    Obtain Anderson-Darling test for each data dimension.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ad_vals = []
        # For each dimension
        for i, dd in enumerate(data_x):
            ad_stts = list(anderson_ksamp([dd, data_y[i]]))
            # Store A-D value and p-value.
            ad_vals.append([ad_stts[0], ad_stts[2]])

            # anderson_darling_k([dd, data_y[i]])

    return np.array(ad_vals).T


def kdeplot(p_vals_cl, p_vals_f, data_id):
    """
    """
    def kdeLims(xvals):
        xmin, xmax = max(-1., min(xvals)), min(2., max(xvals))
        xrng = (xmax - xmin) * .3
        return xmin - xrng, xmax + xrng

    def regKDE(p_vals):
        xmin, xmax = kdeLims(p_vals)
        x_kde = np.mgrid[xmin:xmax:1000j]
        # Obtain the 1D KDE for this distribution of p-values.
        kernel = gaussian_kde(p_vals)
        # KDE for plotting.
        kde_plot = np.reshape(kernel(x_kde).T, x_kde.shape)
        return x_kde, kde_plot

    def KDEoverlap(x_kde, p_vals_cl, p_vals_fr):
        """Calculate overlap between the two KDEs."""
        kcl, kfr = gaussian_kde(p_vals_cl), gaussian_kde(p_vals_fr)

        def y_pts(pt):
            y_pt = min(kcl(pt), kfr(pt))
            return y_pt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            overlap = quad(y_pts, min(x_kde), max(x_kde))
        # Store y values for plotting the overlap filled.
        y_over = np.array([float(y_pts(x_pt)) for x_pt in x_kde])

        return overlap[0], y_over

    # Ranges for the p-values
    cl_fr_range = np.ptp(p_vals_cl)
    fr_fr_range = np.ptp(p_vals_f) if np.any(p_vals_f) else 0.

    if cl_fr_range > 0.001:
        x_cl, kde_cl = regKDE(p_vals_cl)
    else:
        # p_vals_rnd = np.random.normal(p_vals_cl[0], 0.0001, len(p_vals_cl))
        # x_cl, kde_cl = regKDE(p_vals_rnd)
        x_cl, kde_cl = p_vals_cl[0], np.array([])

    if fr_fr_range > 0.001:
        x_fr, kde_fr = regKDE(p_vals_f)
    else:
        # p_vals_rnd = np.random.normal(p_vals_f[0], 0.0001, len(p_vals_f))
        # x_fr, kde_fr = regKDE(p_vals_rnd)
        x_fr, kde_fr = [], np.array([])

    if kde_cl.any() and kde_fr.any():
        pmin, pmax = kdeLims(np.concatenate([p_vals_cl, p_vals_f]))
        x_over = np.linspace(pmin, pmax, 1000)
        overlap, y_over = KDEoverlap(x_over, p_vals_cl, p_vals_f)
        # Probability value for the cluster.
        prob_cl = 1. - overlap
    else:
        # If the KDEs for either distribution could not be obtained because
        # the range is too small, assign a probability of 1.
        prob_cl, x_over, y_over = 1., [], np.array([])

    return [prob_cl, kde_cl, kde_fr, x_cl, x_fr, x_over, y_over]


def anderson_darling_k(samples):
    """
    Not used, testing for now.

    Source: https://github.com/aarchiba/kuiper/blob/master/anderson_darling.py#L193

    Apply the Anderson-Darling k-sample test.
    This test evaluates whether it is plausible that all the samples are drawn
    from the same distribution, based on Scholz and Stephens 1987. The
    statistic computed is their A_kn (rather than A_akn, which differs in
    how it handles ties). The significance of the result is computed by
    producing a scaled and standardized result T_kN, which is returned and
    compared against a list of standard significance levels. The next-larger
    p-value is returned.
    """

    _ps = np.array([0.25, 0.10, 0.05, 0.025, 0.01])
    # allow interpolation to get above _tm
    _b0s = np.array([0.675, 1.281, 1.645, 1.960, 2.326])
    _b1s = np.array([-0.245, 0.250, 0.678, 1.149, 1.822])
    _b2s = np.array([-0.105, -0.305, -0.362, -0.391, -0.396])

    samples = [np.array(sorted(s)) for s in samples]
    all = np.concatenate(samples + [[np.inf]])

    values = np.unique(all)
    L = len(values) - 1
    fij = np.zeros((len(samples), L), dtype=np.int)
    H = 0
    for (i, s) in enumerate(samples):
        c, be = np.histogram(s, bins=values) #, new=True)

        assert np.sum(c) == len(s)

        fij[i, :] = c
        H += 1. / len(s)

    ni = np.sum(fij, axis=1)[:, np.newaxis]
    N = np.sum(ni)
    k = len(samples)
    lj = np.sum(fij, axis=0)
    Mij = np.cumsum(fij, axis=1)
    Bj = np.cumsum(lj)

    A2 = np.sum(
        ((1. / ni) * lj / float(N) * (N * Mij - ni * Bj)**2 /
            (Bj * (N - Bj)))[:, :-1])

    h = np.sum(1. / np.arange(1, N))

    i = np.arange(1, N, dtype=np.float)[:, np.newaxis]
    j = np.arange(1, N, dtype=np.float)
    g = np.sum(np.sum((i < j) / ((N - i) * j)))

    a = (4 * g - 6) * (k - 1) + (10 - 6 * g) * H
    b = (2 * g - 4) * k**2 + 8 * h * k + \
        (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6
    c = (6 * h + 2 * g - 2) * k**2 + \
        (4 * h - 4 * g + 6) * k + (2 * h - 6) * H + 4 * h
    d = (2 * h + 6) * k**2 - 4 * h * k

    sigmaN2 = (a * N**3 + b * N**2 + c * N + d) / ((N - 1) * (N - 2) * (N - 3))

    sigmaN = np.sqrt(sigmaN2)

    TkN = (A2 - (k - 1)) / sigmaN

    tkm1 = _b0s + _b1s / np.sqrt(k - 1) + _b2s / (k - 1)

    ix = np.searchsorted(tkm1, TkN)
    if ix > 0:
        p = _ps[ix - 1]
    else:
        p = 1.
    return A2, TkN, (tkm1, _ps.copy()), p
