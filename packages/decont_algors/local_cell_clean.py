
import numpy as np
import random
from astropy.stats import bayesian_blocks, knuth_bin_width
import operator
import warnings
from functools import reduce


def main(
    n_memb, field_regions_c, memb_prob_avrg_sort, flag_decont_skip,
        fld_clean_bin):
    """
    Takes the photometric diagram of the cluster region with assigned MPs,
    divides it into sub-regions (cells) according to the
    density within it, and removes from each sub-region a number of stars
    equal to the average excess due to field star contamination.

    Parameters
    ----------
    field_regions_c : list
        List of arrays, one for each field region defined. Each array
        contains a unique number of stars, and all their attributes with
        shape: (N_stars, Features)
    memb_prob_avrg_sort: list
        Stars within the cluster region with their MPS added to the last
        column, shape: (Nstars, Features + 1)
    flag_decont_skip: bool
        Whether the DA was applied or not.
    fld_clean_bin : str
        Binning method to be used.

    """

    # Prepare photometric data for cluster and field regions.
    mags_cols_cl, mags_cols_all_fl = dataComb(
        memb_prob_avrg_sort, field_regions_c)

    def regSelect(nbins):
        # Obtain bin edges.
        bin_edges = bin_edges_f(fld_clean_bin, mags_cols_cl, nbins=nbins)

        # Convert into single N dimensional array.
        mags_cols_cl_arr = np.array(mags_cols_cl[0] + mags_cols_cl[1])
        # Obtain N-dimensional cluster region histogram.
        cl_hist_p, cl_hist = get_clust_histo(
            memb_prob_avrg_sort, mags_cols_cl_arr, bin_edges)

        # Obtain field regions histogram (only number of stars in each cell).
        f_hist = get_fl_reg_hist(
            field_regions_c, mags_cols_all_fl, bin_edges, cl_hist)

        # Obtain stars separated in list to be used by the best fit function,
        # and the list of the rejected stars not to be used.
        cl_reg_fit, cl_reg_no_fit = get_fit_stars(
            cl_hist_p, f_hist, flag_decont_skip)

        return cl_reg_fit, cl_reg_no_fit, bin_edges

    # This method requires processing the block several times to keep the
    # best run.
    if fld_clean_bin == 'optm':
        diff_memb, data = [], []
        for nbins in range(24, 2, -1):
            cl_reg_fit, cl_reg_no_fit, bin_edges = regSelect(nbins)
            data.append([cl_reg_fit, cl_reg_no_fit, bin_edges])
            diff_memb.append(abs(n_memb - len(cl_reg_fit)))
        i = np.argmin(diff_memb)
        cl_reg_fit, cl_reg_no_fit, bin_edges = data[i]
    else:
        cl_reg_fit, cl_reg_no_fit, bin_edges = regSelect(None)

    # Check the number of stars selected.
    if len(cl_reg_fit) < 10:
        print("  WARNING: less than 10 stars left after reducing\n"
              "  by 'local' method. Using full list.")
        cl_reg_fit, cl_reg_no_fit, bin_edges = memb_prob_avrg_sort,\
            [], None

    return cl_reg_fit, cl_reg_no_fit, bin_edges


def dataComb(memb_prob_avrg_sort, field_regions_c):
    """
    Combine photometric data into a single array.
    """
    # Cluster region data
    mags_cols_cl = [[], []]
    for mag in list(zip(*list(zip(*memb_prob_avrg_sort))[1:][2])):
        mags_cols_cl[0].append(mag)
    for col in list(zip(*list(zip(*memb_prob_avrg_sort))[1:][4])):
        mags_cols_cl[1].append(col)

    # Field regions data
    mags_cols_all_fl = []
    for freg in field_regions_c:
        # Create list with all magnitudes and colors defined.
        mags_cols_fl = []
        for mag in list(zip(*list(zip(*freg))[1:][2])):
            mags_cols_fl.append(mag)
        for col in list(zip(*list(zip(*freg))[1:][4])):
            mags_cols_fl.append(col)

        mags_cols_all_fl.append(mags_cols_fl)

    return mags_cols_cl, mags_cols_all_fl


def bin_edges_f(
    bin_method, mags_cols_cl, lkl_manual_bins=None, nbins=None, min_bins=2,
        max_bins=50):
    """
    Obtain bin edges for each photometric dimension using the cluster region
    diagram. The 'bin_edges' list will contain all magnitudes first, and then
    all colors (in the same order in which they are read).
    """
    bin_edges = []
    if bin_method in (
            'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt'):

        for mag in mags_cols_cl[0]:
            bin_edges.append(np.histogram(mag, bins=bin_method)[1])
        for col in mags_cols_cl[1]:
            bin_edges.append(np.histogram(col, bins=bin_method)[1])

    elif bin_method == 'optm':
        for mag in mags_cols_cl[0]:
            bin_edges.append(np.histogram(mag, bins=nbins * 2)[1])
        for col in mags_cols_cl[1]:
            bin_edges.append(np.histogram(col, bins=nbins)[1])

    elif bin_method == 'fixed':
        # Based on Bonatto & Bica (2007) 377, 3, 1301-1323 but using larger
        # values than those used there (0.25 for colors and 0.5 for magnitudes)
        for mag in mags_cols_cl[0]:
            b_num = int(round(max(2, (max(mag) - min(mag)) / 1.)))
            bin_edges.append(np.histogram(mag, bins=b_num)[1])
        for col in mags_cols_cl[1]:
            b_num = int(round(max(2, (max(col) - min(col)) / .5)))
            bin_edges.append(np.histogram(col, bins=b_num)[1])

    elif bin_method == 'knuth':
        for mag in mags_cols_cl[0]:
            bin_edges.append(knuth_bin_width(
                mag, return_bins=True, quiet=True)[1])
        for col in mags_cols_cl[1]:
            bin_edges.append(knuth_bin_width(
                col, return_bins=True, quiet=True)[1])

    elif bin_method == 'blocks':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for mag in mags_cols_cl[0]:
                bin_edges.append(bayesian_blocks(mag))
            for col in mags_cols_cl[1]:
                bin_edges.append(bayesian_blocks(col))

    elif bin_method == 'blocks-max':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for mag in mags_cols_cl[0]:
                bin_edges.append(slpitArr(bayesian_blocks(mag)))
            for col in mags_cols_cl[1]:
                bin_edges.append(slpitArr(bayesian_blocks(col), 1.))

    elif bin_method == 'manual':
        for mag in mags_cols_cl[0]:
            bin_edges.append(
                np.histogram(mag, bins=int(lkl_manual_bins[0]))[1])
        for i, col in enumerate(mags_cols_cl[1]):
            bin_edges.append(
                np.histogram(col, bins=int(lkl_manual_bins[i + 1]))[1])

    # TODO this method is currently hidden from the params file.
    # To be used when #325 is implemented. Currently used to test
    # multi-dimensional likelihoods.
    #
    # For 4 to 6 dimensions the rule below appears to be a somewhat reasonable
    # rule of thumb for the number of bins for each dimension.
    # There is a trade-off between a large number of smaller bins which
    # better match features of the observed cluster but benefits larger
    # mass values, and fewer larger bins which better match masses but losing
    # finer details of the cluster.
    elif bin_method == 'man':
        d = len(mags_cols_cl[0]) + len(mags_cols_cl[1])
        b_num = [15, 10, 7][d - 4]
        for mag in mags_cols_cl[0]:
            bin_edges.append(np.histogram(mag, bins=int(b_num))[1])
        for col in mags_cols_cl[1]:
            bin_edges.append(np.histogram(col, bins=int(b_num))[1])

    # Impose a minimum of 'min_bins' cells per dimension. The number of bins
    # is the number of edges minus 1.
    for i, be in enumerate(bin_edges):
        N_bins = len(be) - 1
        if N_bins < min_bins:
            # print("  WARNING too few bins in histogram, use 'min_bins'")
            bin_edges[i] = np.linspace(be[0], be[-1], min_bins + 1)

    # Impose a maximum of 'max_bins' cells per dimension.
    for i, be in enumerate(bin_edges):
        N_bins = len(be) - 1
        if N_bins > max_bins:
            # print("  WARNING too many bins in histogram, use 'max_bins'")
            bin_edges[i] = np.linspace(be[0], be[-1], max_bins)

    return bin_edges


def slpitArr(data, step=3.):
    """
    Insert extra elements into array so that the maximum spacing between
    elements is 'step'.
    Source: https://stackoverflow.com/q/52769257/1391441
    """
    d = data.copy()
    d[1:] -= data[:-1]
    m = -(d // -step).astype(int)
    m[0] = 1
    d /= m
    return np.cumsum(d.repeat(m))


def get_clust_histo(memb_prob_avrg_sort, mags_cols_cl, bin_edges):
    """
    Generate the N-dimensional cluster region histogram, with each star
    positioned in its corresponding cell.
    """

    # Cluster region N-dimensional histogram.
    cl_hist = np.histogramdd(np.array(list(
        zip(*mags_cols_cl))), bins=bin_edges)[0]
    # np.shape(cl_hist) gives the tuple containing one element per dimension,
    # indicating how many cells that dimension was divided into.

    # Add a very small amount to each outer-most edge so the 'np.digitize'
    # function will position the stars on the edges correctly.
    for i, b_e in enumerate(bin_edges):
        bin_edges[i][0] = b_e[0] - (abs(b_e[0]) / 100.)
        bin_edges[i][-1] = b_e[-1] + (b_e[-1] / 100.)

    # Position each cluster region star in its corresponding N-dimensional
    # cell/bin.
    cl_st_indx = []
    # Store indexes for each dimension.
    for i, mag_col in enumerate(mags_cols_cl):
        # Set correct indexes for array subtracting 1, since 'np.digitize'
        # counts one more bin to the right by default.
        cl_st_indx.append(np.digitize(mag_col, bin_edges[i]) - 1)

    # Create empty list with the same dimension as the photometric N-histogram.
    cl_hist_p = np.empty(shape=cl_hist.shape + (0,)).tolist()
    # Position stars in their corresponding N-histogram cells. Since the stars
    # are already sorted by their MPs, they will be correctly sorted in the
    # final list here too.
    for i, h_indx in enumerate(list(zip(*cl_st_indx))):
        # Store stars.
        reduce(operator.getitem, list(h_indx), cl_hist_p).append(
            memb_prob_avrg_sort[i])

    return cl_hist_p, cl_hist


def get_fl_reg_hist(field_regions_c, mags_cols_all_fl, bin_edges, cl_hist):
    """
    Obtain the average number of field region stars in each cell defined for
    the N-dimensional cluster region photometric diagram.
    """

    # Empty field region array shaped like the cluster region array.
    f_hist = np.zeros(shape=np.shape(cl_hist))
    # Add stars in all the defined field regions.
    for mags_cols_fl in mags_cols_all_fl:
        # N-dimension histogram for each field region.
        f_hist = f_hist + np.histogramdd(
            np.array(list(zip(*mags_cols_fl))), bins=bin_edges)[0]

    # Average number of stars in each cell/bin and round to integer.
    f_hist = np.around(f_hist / len(field_regions_c), 0)

    return f_hist


def get_fit_stars(cl_hist_p, f_hist, flag_decont_skip):
    """
    Iterate through each N-dimensional cell of the cluster region array and
    remove the excess of field stars in each one, selecting those with the
    lowest assigned MPs if the DA was applied. Otherwise select random stars.
    """

    # DEPRECATED by the minimum of two cells per dimension imposed in
    # bin_edges_f()
    # Only flatten list if more than 1 cell was defined.
    # if len(cl_hist_p) > 1:
    #     cl_hist_p_flat = np.asarray(cl_hist_p).flatten()
    # else:
    #     cl_hist_p_flat = cl_hist_p[0]

    # Flatten arrays to access all of its elements.
    cl_hist_p_flat = np.asarray(cl_hist_p, dtype=object).flatten()
    f_hist_flat = f_hist.flatten()

    cl_reg_fit, cl_reg_no_fit = [], []
    # For each cell defined.
    for i, cl_cell in enumerate(cl_hist_p_flat):

        # Get average number of field regions in this cell.
        N_fl_reg = f_hist_flat[i]

        if N_fl_reg > 0.:
            # Discard the excess of N_reg_fl stars from this cluster region.

            # If the DA was not applied, discard N_fl_reg *random* stars in
            # the cell.
            if flag_decont_skip:

                if int(N_fl_reg) < len(cl_cell):
                    # Generate list with randomized cell indexes.
                    ran_indx = random.sample(
                        range(len(cl_cell)), len(cl_cell))

                    # Store len(cl_cell) - N_fl_reg stars
                    cl_reg_fit +=\
                        [cl_cell[i] for i in ran_indx[:-int(N_fl_reg)]]
                    # Discard N_fl_reg stars.
                    cl_reg_no_fit +=\
                        [cl_cell[i] for i in ran_indx[-int(N_fl_reg):]]
                else:
                    # Discard *all* stars in the cell.
                    cl_reg_no_fit += cl_cell
            else:
                # Discard those N_fl_reg with the smallest MPs, keep the rest.
                cl_reg_fit += cl_cell[:-int(N_fl_reg)]
                cl_reg_no_fit += cl_cell[-int(N_fl_reg):]
        else:
            # No field region stars in this cell, keep all stars.
            cl_reg_fit += cl_cell

    return cl_reg_fit, cl_reg_no_fit
