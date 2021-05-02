
from os.path import join, isfile
import traceback
from packages.inp import input_params


def check(mypath, file_end):
    """
    Check the existence of the 'params_input.dat' file, and check that
    the parameters are properly written.
    """

    # Check if params_input_XX.dat file exists.
    pars_f_name = 'params_input' + file_end + '.dat'
    pars_f_path = join(mypath, pars_f_name)
    if file_end == '':
        if not isfile(pars_f_path):
            raise ValueError("'{}' file does not exist.".format(pars_f_name))
    else:
        if not isfile(pars_f_path):
            print("  WARNING: {} file does not exist.\n  Falling back to"
                  " 'params_input.dat' file.\n".format(pars_f_name))

            # Fall back to default file.
            pars_f_name = 'params_input.dat'
            pars_f_path = join(mypath, pars_f_name)
            if not isfile(pars_f_path):
                raise ValueError("'{}' file does not exist.".format(
                    pars_f_name))
    # Check if params_input_XX.dat file is properly formatted.
    try:
        # Read input parameters from params_input.dat file.
        pd = input_params.main(pars_f_path)
    except Exception:
        # Halt code.
        print(traceback.format_exc())
        raise ValueError("'{}' is badly formatted.".format(pars_f_name))

    # Add to parameters dictionary.
    pd['file_end'] = file_end

    if pd['best_fit_algor'] not in pd['optimz_algors']:
        raise ValueError("the selected best fit method '{}' does not match"
                         " a valid input.".format(pd['best_fit_algor']))

    return pd
