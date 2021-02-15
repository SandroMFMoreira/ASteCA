
import time
from os.path import join, realpath, dirname
from os import getcwd
import argparse
import traceback
from ._version import __version__
from .checker import check_all


def num_exec():
    """
    Parse optional command-line argument. The integer (if) passed, will be used
    as the two last characters in the file name of a
    'params_input_XX.dat' file.
    Integer must be smaller than 99, else default 'params_input.dat' file is
    used.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", help="Integer. Set the params_input_N.dat"
                        " file to be used by this run.", type=int)
    args = parser.parse_args()
    file_end = ''
    if args.N:
        if args.N < 99:
            file_end = "_{:0>2}".format(args.N)
            print("Will load parameters from 'params_input{}.dat'"
                  " file.\n".format(file_end))
        else:
            print("Integer must be smaller than 99. Fall back to\ndefault"
                  " 'params_input.dat' file.\n")

    return file_end


def main():
    """
    Reads input data files and calls the container function.
    """
    # Start timing loop.
    start = time.time()

    print('\n-------------------------------------------')
    print('             [ASteCA {}]'.format(__version__))
    print('-------------------------------------------\n')

    # Root path where the code is running. Remove 'packages' from path.
    mypath = realpath(join(getcwd(), dirname(__file__)))[:-8]

    # Read command-line argument.
    file_end = num_exec()

    # Checker function to verify that things are in place before running.
    # As part of the checking process, and to save time, the isochrone
    # files are read and stored here.
    cl_files, pd = check_all(mypath, file_end)

    # Prepare tracks and other required data
    if pd['best_fit_algor'] != 'n':
        from packages.synth_clust import tracksPrep
        pd = tracksPrep.main(pd)

    # Import here to ensure the check has passed and all the necessary
    # packages are installed.
    from packages import func_caller

    # Iterate through all cluster files.
    for cl_file in cl_files:
        try:
            # Call module that calls all sub-modules sequentially.
            func_caller.main(cl_file, pd)
        except Exception:
            print('\n!!! --> {}/{} '.format(cl_file[-2], cl_file[-1])
                  + 'could not be successfully processed <-- !!!\n')
            print(traceback.format_exc())

    # End of run.
    elapsed = time.time() - start
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print('Full run completed in {:.0f}h {:.0f}m {:.0f}s'.format(h, m, s))
