
import numpy as np
import warnings
from astropy.io import ascii
from astropy.table import Table
from astropy import wcs
import operator
import copy
from functools import reduce
from ..structure import trim_frame
from ..aux_funcs import flatten, list_duplicates


def main(
    npd, nanvals, id_col, x_col, y_col, mag_col, e_mag_col, col_col,
    e_col_col, plx_col, e_plx_col, pmx_col, e_pmx_col, pmy_col, e_pmy_col,
        rv_col, e_rv_col, coords, project, flag_tf, tf_range, **kwargs):
    """
    Read data from the cluster's input file.

    Separate data into "incomplete" and "complete", where the latter means
    all stars in the  input file, and the former only those with all the
    photometric data available.
    """

    data_file = npd['data_file']
    try:
        data = readDataFile(nanvals, id_col, data_file)
        N_all = len(data)

        # Arrange column names in the proper order and shape.
        col_names = [
            id_col, x_col, y_col, mag_col, e_mag_col, col_col, e_col_col,
            [plx_col, pmx_col, pmy_col, rv_col],
            [e_plx_col, e_pmx_col, e_pmy_col, e_rv_col]]
        # Remove not wanted columns.
        col_names_keep = list(filter(bool, list(flatten(col_names))))
        data.keep_columns(col_names_keep)

        # Remove stars with no valid coordinates data in either x or y
        try:
            data = data[(~data[x_col].mask)]
        except AttributeError:
            # Not a masked columns
            pass
        try:
            data = data[(~data[y_col].mask)]
        except AttributeError:
            pass
        N_coords = len(data)

        # Mask bad photometry.
        data = maskBadPhot(data, mag_col, e_mag_col, col_col, e_col_col)

        # Trim frame according to coordinates limits.
        data = trim_frame.main(flag_tf, tf_range, data, x_col, y_col)
        N_trim = len(data)

        # Split into incomplete and complete data.
        data_compl, flag_data_eq = dataSplit(
            mag_col, e_mag_col, col_col, e_col_col, data)

        # Change masked elements with 'nan' values, in place.
        fill_cols(data)
        fill_cols(data_compl)

    except ascii.InconsistentTableError:
        raise ValueError("ERROR: could not read data input file:\n  {}\n"
                         "  Check that all rows are filled (i.e., no blank"
                         " spaces)\n  for all columns\n".format(data_file))

    # Create cluster's dictionary with the *photometrically incomplete* data.
    ids, x, y, mags, cols, kine, em, ec, ek = dataCols(
        data_file, data, col_names)
    x, y, x_offset, y_offset = coordsProject(x, y, coords, project)
    cld_i = {'ids': ids, 'x': x, 'y': y, 'mags': mags, 'em': em,
             'cols': cols, 'ec': ec, 'kine': kine, 'ek': ek}

    # Create cluster's dictionary with the *photometrically complete* data.
    ids, x, y, mags, cols, kine, em, ec, ek = dataCols(
        data_file, data_compl, col_names)
    x, y, _, _ = coordsProject(x, y, coords, project, x_offset, y_offset)
    cld_c = {'ids': ids, 'x': x, 'y': y, 'mags': mags, 'em': em,
             'cols': cols, 'ec': ec, 'kine': kine, 'ek': ek}

    print("Stars read from input file: N={}".format(N_all))
    if N_coords != N_all:
        frac_reject = 1. - (float(N_coords) / N_all)
        print(
            "  WARNING: N={} ({:.1f}%) stars with no valid\n"
            "  coordinates data were removed".format(
                N_all - N_coords, 100. * frac_reject))

    if N_trim != N_coords:
        print("Stars removed by trimming the frame: N={}".format(
            N_coords - N_trim))

    frac_reject = 1. - (float(cld_c['ids'].size) / cld_i['ids'].size)
    if frac_reject > 0.05:
        print(("  WARNING: {} stars ({:.0f}%) contain incomplete " +
               "photometric data").format(
                   cld_i['ids'].size - cld_c['ids'].size, 100. * frac_reject))

    clp = {
        'flag_data_eq': flag_data_eq, 'x_offset': x_offset,
        'y_offset': y_offset, 'col_names_keep': col_names_keep}

    return cld_i, cld_c, clp


def readDataFile(nanvals, id_col, data_file):
    """
    Read input data file.
    """
    # Identify all these strings as invalid entries.
    fill_msk = [('', '0')] + [(_, '0') for _ in nanvals]
    # Read IDs as strings, not applying the 'fill_msk'. Read rest of the
    # data applying the fill mask
    data = ascii.read(
        data_file, fill_values=fill_msk, fill_exclude_names=(id_col,),
        converters={id_col: [ascii.convert_numpy(np.str)]})

    try:
        data[id_col]
    except KeyError:
        raise ValueError(
            "ERROR: the '{}' key could not be found. Check that \n"
            "the 'id' name is properly written, and that all columns \n"
            "have *unique* names\n".format(id_col))

    return data


def fill_cols(tbl, fill=np.nan, kind='f'):
    """
    In-place fill of 'tbl' columns which have dtype 'kind' with 'fill' value.

    Source: https://stackoverflow.com/a/50164954/1391441
    """
    for col in tbl.itercols():
        try:
            if col.dtype.kind == kind:
                col[...] = col.filled(fill)
        except AttributeError:
            # If there where no elements to mask, the type is 'Column' and
            # not 'MaskedColumn', so .filled() is not a valid attribute.
            pass


def maskBadPhot(data, mag_col, e_mag_col, col_col, e_col_col):
    """
    Mask photometric values outside the (mmin, mmax) range.
    """
    # HARDCODED valid magnitude range
    mmin, mmax = -10., 50.

    # Convert to masked table in case it is not.
    data = Table(data, masked=True)

    # # TODO this is a temporary magnitude filter
    # data = data[np.array([float(_) < 18. for _ in data[mag_col[0]]])]

    # TODO this for block is rather slow for large data files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx in [mag_col, e_mag_col, col_col, e_col_col]:
            for c_idx in idx:
                # Reject bad photometric values.
                msk = np.array([mmin < float(_) < mmax for _ in data[c_idx]])
                # Apply masks to each photometric column.
                data[c_idx].mask = [~msk]

    return data


def dataSplit(mag_col, e_mag_col, col_col, e_col_col, data):
    """
    """
    # Define PHOTOMETRIC data columns.
    data_phot = list(flatten([mag_col, e_mag_col, col_col, e_col_col]))
    # Check if there are any masked elements in the photometric data.
    masked_elems = 0
    for col in data_phot:
        # Catch "AttributeError: 'Column' object has no attribute 'mask'"
        # if column is not masked.
        try:
            masked_elems += data[col].mask.sum()
        except AttributeError:
            pass

    # Remove rows with at least one masked *photometric* element.
    flag_data_eq = False
    if masked_elems > 0:
        data_compl = data[reduce(
            operator.and_, [~data[col].mask for col in data_phot])]
    else:
        # If there where no elements to mask, there were no bad photometric
        # values.
        data_compl = copy.deepcopy(data)
        flag_data_eq = True

    return data_compl, flag_data_eq


def dataCols(data_file, data, col_names):
    """
    Separate data into appropriate columns.
    """
    try:
        ids = np.array(data[col_names[0]])
        dups = list(list_duplicates(ids))
        if dups:
            print("ERROR: duplicated IDs found in data file:")
            for dup in dups:
                print("  ID '{}' found in lines: {}".format(
                    dup[0], ", ".join(dup[1])))
            raise ValueError("Duplicated IDs found")

        # Check that all data columns are in the proper 'float64' format.
        # This catches values like '0.343a' which make the entire column
        # be processed with a string format.
        # TODO since columns are masked astropy displays a cryptic message, see
        # https://github.com/astropy/astropy/issues/8071
        for i in (1, 2):
            try:
                data[col_names[i]].dtype = 'float64'
            except ValueError:
                raise ValueError("Bad data value in column '{}'".format(
                    col_names[i]))
        for i in (3, 4, 5, 6):
            for mc in col_names[i]:
                try:
                    data[mc].dtype = 'float64'
                except ValueError:
                    raise ValueError("Bad data value in column '{}'".format(
                        mc))
        for i in (7, 8):
            for k in col_names[i]:
                if k is not False:
                    try:
                        data[k].dtype = 'float64'
                    except ValueError:
                        raise ValueError(
                            "Bad data value in column '{}'".format(k))

        # Read coordinates data.
        x, y = np.array(data[col_names[1]]), np.array(data[col_names[2]])

        # Read magnitudes
        mags, em = [], []
        for mi, emi in zip(*[col_names[3], col_names[4]]):
            mags.append(np.array(data[mi]))
            em.append(np.array(data[emi]))
        # Read colors.
        cols, ec = [], []
        for ci, eci in zip(*[col_names[5], col_names[6]]):
            cols.append(np.array(data[ci]))
            ec.append(np.array(data[eci]))
        mags, cols, em, ec = np.array(mags), np.array(cols), np.array(em),\
            np.array(ec)

        # Read PMs, parallax, RV.
        kine, ek = [], []
        for ki, eki in zip(*[col_names[7], col_names[8]]):
            if ki is not False:
                kine.append(np.array(data[ki]))
                ek.append(np.array(data[eki]))
            else:
                kine.append(np.full(ids.size, np.nan))
                ek.append(np.full(ids.size, np.nan))
        kine, ek = np.array(kine), np.array(ek)

    except IndexError:
        raise IndexError("ERROR: data input file:\n  {}\n  contains "
                         "fewer columns than those given "
                         "in 'params_input.dat'".format(data_file))

    # Check if the range of any coordinate column is zero.
    data_names = ['x_coords', 'y_coords']
    for i, dat_lst in enumerate([x, y]):
        if np.nanmin(dat_lst) == np.nanmax(dat_lst):
            raise ValueError("ERROR: the range for the '{}' column\n"
                             "is zero. Check the input data format".format(
                                 data_names[i]))
    # Check if the range of any photometric column is zero.
    data_names = ['magnitude', 'color']
    for i, dat_lst in enumerate([mags, cols]):
        for mc in dat_lst:
            if np.nanmin(mc) == np.nanmax(mc):
                raise ValueError(
                    "ERROR: the range for {} column {} is\nzero."
                    " Check the input data format".format(data_names[i], i))

    return ids, x, y, mags, cols, kine, em, ec, ek


def coordsProject(x, y, coords, project, ra_cent=None, dec_cent=None):
    """
    Sinusoidal projection.
    """
    if coords == 'deg' and project:

        if ra_cent is None:
            # Use this method to transform the coordinates before applying
            # the sinusoidal projection. This handles the cases where the frame
            # is splitted between quadrants
            ww = wcs.WCS(naxis=2)
            # Sanson-Flamsteed (“global sinusoid”)
            ww.wcs.ctype = ["RA---SFL", "DEC--SFL"]
            x_px, y_px = ww.wcs_world2pix(x, y, 1)
            # Remove the cosine
            x_px /= np.cos(np.deg2rad(y_px))
            # DFind the origin for the observed frame
            ra_cent = (max(x_px) + min(x_px)) / 2.
            dec_cent = (max(y_px) + min(y_px)) / 2.
            # Apply the sinusoidal projection
            x = (x_px - ra_cent) * np.cos(np.deg2rad(y_px))
            y = (y_px - dec_cent)
        else:
            x = (x - ra_cent) * np.cos(np.deg2rad(y))
            y = (y - dec_cent)
        x_offset = ra_cent if ra_cent > 0. else (ra_cent + 360.)
        y_offset = dec_cent
    else:
        x_offset, y_offset = 0., 0.

    return x, y, x_offset, y_offset


if __name__ == '__main__':
    main()
