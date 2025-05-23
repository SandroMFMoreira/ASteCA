import os
import numpy as np
import pandas as pd

# IMPORTANT: this dictionary contains fixed data required to read isochrones from
# each of the models. If this ever changes, this function will no longer work
#
# col_names: default column names for the initial mass, metallicity, and age
# comment_char: character used to indicate comments in the isochrone files
# sep_cols: regular expression to separate columns in the isochrone files
# idx_col_line: index of the line that contains the column names
# myr_to_log10: flag to indicate whether to convert Myr to log10
phot_systs_data = {
    "PARSEC": {
        "col_names": {"mass_col": "Mini", "met_col": "Zini", "age_col": "logAge"},
        "comment_char": "#",
        "sep_cols": r"\s+",
        "idx_col_line": -1,
        "parsec_stage_9_col": "label",  # column name for the post-AGB stage
        "parsec_stage_9_id": "9",  # ID for the post-AGB stage
    },
    "MIST": {
        "col_names": {
            "mass_col": "initial_mass",
            "met_col": "Zinit",
            "age_col": "log10_isochrone_age_yr",
        },
        "comment_char": "#",
        "sep_cols": r"\s+",
        "idx_col_line": -1,
    },
    "BASTI": {
        "col_names": {
            "mass_col": "M/Mo(ini)",
            "met_col": "Z =",
            "age_col": "Age (Myr) =",
        },
        "comment_char": "#",
        "sep_cols": r"\s+",
        "idx_col_line": -2,
        "myr_to_log10": True,
    },
    "BARAFFE":{
        "col_names": {
            "mass_col": "M/Ms",
            "age_col": "Age",
            "met_col": "Z",
        },
            "sep_cols": r"\s+",
            "comment_char": "#",
            "idx_col_line": -1,
    },
}


def load(
    model: str,
    isochs_path: str,
    magnitude: str,
    color: tuple,
    color2: tuple | None,
    column_names: dict | None,
    N_interp: int,
    parsec_rm_stage_9: bool,
) -> tuple[np.ndarray, list, dict, int]:
    """Load the theoretical isochrones and return processed data.

    :param model: The model must be one of the supported isochrone services.
    :type model: str
    :param isochs_path: Path to the isochrone files.
    :type isochs_path: str
    :param magnitude: Magnitude filter to load.
    :type magnitude: str
    :param color: Tuple with the two filters to generate the first color.
    :type color: tuple
    :param color2: Tuple with the two filters to generate the second color,
        defaults to None
    :type color2: tuple | None
    :param column_names: Dictionary with the column names for the isochrones,
        defaults to None
    :type column_names: dict | None
    :param N_interp: Number of points to interpolate.
    :type N_interp: int
    :param parsec_rm_stage_9: Flag to indicate whether to remove post-AGB stage for
        PARSEC models, defaults to True
    :type parsec_rm_stage_9: bool

    :raises ValueError: If there is a shape mismatch in the loaded isochrones

    :return: Array of isochrones, individual filters for each color defined,
        dictionary with metallicities and ages, and number of files read.
    :rtype: tuple[np.ndarray, list, dict, int]
    """
    cols_keep, mass_col, met_col, age_col = get_columns(
        column_names, model, magnitude, color, color2
    )

    f_paths = extract_paths(isochs_path)

    # isochrones.shape = (N_photsyst, N_z, N_a, N_interp, N_cols)
    # met_age_arr.shape = (N_photsyst, N_z*N_a, 2)
    met_age_vals, isoch_dataframes = read(
        model,
        parsec_rm_stage_9,
        f_paths,
        met_col,
        age_col,
        cols_keep,
    )

    isochrones = interp_df(N_interp, met_age_vals, isoch_dataframes)

    isochrones, met_age_arr = merge_ps_massini_check(mass_col, isochrones)
    try:
        np.shape(isochrones)
    except ValueError:
        raise ValueError(
            "Shape mismatch in loaded isochrones. This usually means that an\n"
            + "incorrect number of ages and/or metallicities are stored in the\n"
            + "isochrone files."
        )

    # Create dictionary of mets and ages
    all_m, all_a = np.array(met_age_arr).T
    # Discard duplicate values
    all_m = np.array(list(dict.fromkeys(all_m))).astype(float)
    all_a = np.array(list(dict.fromkeys(all_a))).astype(float)
    met_age_dict = {"met": all_m, "loga": all_a}

    # theor_tracks.shape = (N_z, N_a, N_cols, N_interp)
    theor_tracks, color_filters = shape_isochrones(
        magnitude, color, color2, mass_col, isochrones
    )

    return theor_tracks, color_filters, met_age_dict, len(f_paths)


def get_columns(
    column_names: dict | None,
    model: str,
    magnitude: str,
    color: tuple,
    color2: tuple | None,
) -> tuple[list, str, str, str]:
    """Get the column names for the isochrones.

    :param column_names: Dictionary with the column names for the isochrones,
        defaults to None
    :type column_names: dict | None
    :param model: Isochrone model name.
    :type model: str
    :param magnitude: Magnitude filter to load.
    :type magnitude: str
    :param color: Tuple with the two filters to generate the first color.
    :type color: tuple
    :param color2: Tuple with the two filters to generate the second color,
        defaults to None
    :type color2: tuple | None

    :return: List of columns to keep, mass column name, metallicity column name,
        and age column name.
    :rtype: tuple[list, str, str, str]
    """
    if column_names is None:
        mass_col = phot_systs_data[model]["col_names"]["mass_col"]
        met_col = phot_systs_data[model]["col_names"]["met_col"]
        age_col = phot_systs_data[model]["col_names"]["age_col"]
    else:
        mass_col = column_names["mass_col"]
        met_col = column_names["met_col"]
        age_col = column_names["age_col"]

    # Select columns to keep
    all_filters = [magnitude] + list(color)
    if color2 is not None:
        all_filters += list(color2)
    cols_keep = list(dict.fromkeys(all_filters)) + [mass_col]

    return cols_keep, mass_col, met_col, age_col


def extract_paths(isochs_path: str) -> list:
    """Extract isochrone files from `isochs_path`.

    :param isochs_path: Path to the isochrone files.
    :type isochs_path: str

    :raises FileNotFoundError: If no files are found in the isochrones path.

    :return: List of isochrone file paths.
    :rtype: list
    """
    # Check if path is to file or folder
    if os.path.isfile(isochs_path):
        f_paths = [isochs_path]
    else:
        f_paths = []
        # Iterate over files in directory
        for path, folders, files in os.walk(isochs_path):
            # Skip hidden folders
            if path.split("/")[-1].startswith("."):
                continue
            for filename in files:
                # Skip hidden files
                if not filename.startswith("."):
                    f_paths.append(os.path.join(path, filename))

        if len(f_paths) == 0:
            raise FileNotFoundError(
                f"No files found in isochrones path '{isochs_path}'"
            )

    return f_paths


def read(
    model: str,
    parsec_rm_stage_9: bool,
    f_paths: list,
    met_col: str,
    age_col: str,
    cols_keep: list,
) -> tuple[list[str], list[pd.DataFrame]]:
    """Read isochrone files and store them as pandas DataFrame along with its associated
    metallicity and age values.

    :param model: Isochrone model name.
    :type model: str
    :param parsec_rm_stage_9: Remove post-AGB stage for PARSEC models.
    :type parsec_rm_stage_9: bool
    :param f_paths: List of isochrone file paths.
    :type f_paths: list
    :param met_col: Metallicity column name.
    :type met_col: str
    :param age_col: Age column name.
    :type age_col: str
    :param cols_keep: List of columns to keep.
    :type cols_keep: list

    :return: First list contains the met and age values (as strings), second list
     contains the associated isochrones as pandas DataFrames
    :rtype: tuple[list[str], list[pd.DataFrame]]
    """

    met_age_vals, isoch_dataframes = [], []
    for file_path in f_paths:
        # Extract columns names and full header
        col_names, full_header = get_header(model, file_path)

        # Columns to keep for this photometric system
        cols_keep_ps = list(set(col_names) & set(cols_keep))

        # Load file
        df_file_path = pd.read_csv(
            file_path,
            comment=phot_systs_data[model]["comment_char"],
            header=None,
            names=col_names,
            sep=phot_systs_data[model]["sep_cols"],
        )

        if model == "PARSEC":
            # Group by metallicity
            df_blocks = df_file_path.groupby(met_col, sort=False)
            # Process metallicity blocks
            for _, met_df in df_blocks:
                # Group by age
                age_blocks = met_df.groupby(age_col, sort=False)
                for _, df in age_blocks:
                    # Remove post-AGB stage
                    if parsec_rm_stage_9 is True:
                        phot_systs_data[model]["parsec_stage_9_col"]
                        msk = (
                            df[phot_systs_data[model]["parsec_stage_9_col"]]
                            != phot_systs_data[model]["parsec_stage_9_id"]
                        )
                        df = df[msk]
                    # Extract met, age values (use first element, all are equal in
                    # column)
                    met = str(np.array(df[met_col])[0])
                    age = str(np.array(df[age_col])[0])
                    # Store data
                    met_age_vals.append([met, age])
                    isoch_dataframes.append(df[cols_keep_ps].astype(float))

        elif model == "MIST":
            met = get_MIST_z_val(met_col, full_header)
            # Group by age
            df_blocks = df_file_path.groupby(age_col, sort=False)
            # Process age blocks
            for _, df in df_blocks:
                age = str(df[age_col].values[0])
                # Store data
                met_age_vals.append([met, age])
                isoch_dataframes.append(df[cols_keep_ps].astype(float))

        elif model == "BASTI":
            met, age = get_BASTI_z_a_val(full_header, met_col, age_col)
            # Store data
            met_age_vals.append([met, age])
            isoch_dataframes.append(df_file_path[cols_keep_ps].astype(float))

        elif model == "BARAFFE":
            # Group by age
            age_blocks = df_file_path.groupby(age_col, sort=False)

            # Process age groups
            for _, df in age_blocks:
                # Extract met, age values (use first element, all are equal in column)
                met = str(df[met_col].iloc[0])  # Assuming metallicity is constant
                age = str(df[age_col].iloc[0])

                # Store data
                met_age_vals.append([met, age])
                isoch_dataframes.append(df[cols_keep_ps].astype(float))

    return met_age_vals, isoch_dataframes


def get_header(model: str, file_path: str) -> tuple[list, list]:
    """Iterate through each line in the file to get the header.
    Extract the column names from the header.

    :param model: Name of the isochrones model used
    :type model: str
    :param file_path: Path to the isochrone file.
    :type file_path: str

    :return: List of column names and full header.
    :rtype: tuple[list, list]
    """
    # Extract full header
    with open(file_path, mode="r") as f_iso:
        full_header = []
        for line in f_iso:
            if not line.startswith(phot_systs_data[model]["comment_char"]):
                break
            full_header.append(line)

    # Extract column names
    column_line = full_header[phot_systs_data[model]["idx_col_line"]]
    column_names = column_line.replace(
        phot_systs_data[model]["comment_char"], ""
    ).split()

    return column_names, full_header


def get_MIST_z_val(met_col: str, full_header: list) -> str:
    """Extract metallicity value for MIST files.

    :param met_col: Metallicity column name.
    :type met_col: str
    :param full_header: Full header of the isochrone file.
    :type full_header: list

    :raises ValueError: If the metallicity cannot be read from the header.

    :return: Metallicity value.
    :rtype: str
    """
    met = None
    for i, line in enumerate(full_header):
        line = line.replace(phot_systs_data["MIST"]["comment_char"], "")
        if met_col in line:
            # Identify lines that contain the metallicity column
            z_header_cols = line.split()
            # Assume the metallicity value is in the next line
            next_line = full_header[i + 1].replace(
                phot_systs_data["MIST"]["comment_char"], ""
            )
            z_header_vals = next_line.split()
            # Extract metallicity value as string
            met_idx = z_header_cols.index(met_col)
            met = z_header_vals[met_idx]
            break

    if met is None:
        raise ValueError("Could not read header from MIST isochrone")

    return met


def get_BASTI_z_a_val(full_header: list, met_col: str, age_col: str) -> tuple[str, str]:
    """Extract metallicity and age values for BASTI files.

    :param full_header: Full header of the isochrone file.
    :type full_header: list
    :param met_col: Metallicity column name.
    :type met_col: str
    :param age_col: Age column name.
    :type age_col: str

    :raises ValueError: If the metallicity or age cannot be read from the header.

    :return: Metallicity and age values.
    :rtype: tuple[str, str]
    """
    met, age = None, None
    for line in full_header:
        line = line.replace(phot_systs_data["BASTI"]["comment_char"], "").strip()
        if met_col in line:
            # Assume both values exist in the same line
            met = line.split(met_col)[1].split()[0]
            age = line.split(age_col)[1].split()[0]

            # Basti ages are expressed in Myr, convert to log10
            if phot_systs_data["BASTI"]["myr_to_log10"]:
                age = np.log10(float(age) * 1e6)
                # Back to (rounded) string
                age = str(round(age, 5))
            break

    if met is None or age is None:
        raise ValueError("Could not read header from Basti isochrone")

    return met, age


def interp_df(
    N_interp: int,
    met_age_vals: list,
    isoch_dataframes: list[pd.DataFrame],
) -> dict:
    """Interpolate the isochrone data.

    :param N_interp: Number of points to interpolate.
    :type N_interp: int
    :param met_age_vals: List of metallicity and ages
    :type met_age_vals: list
    :param isoch_dataframes: List of isochrones as pd.DataFrame
    :type isoch_dataframes: list[pd.DataFrame]

    :return: Dictionary of isochrones.
    :rtype: dict
    """
    # Convert the string elements to floats for sorting
    met_age_f = [(float(x), float(y)) for x, y in met_age_vals]

    # Get the indexes that sort the list by metallicities first and ages second
    sorted_indexes = sorted(
        range(len(met_age_f)), key=lambda i: (met_age_f[i][0], met_age_f[i][1])
    )
    # Sort metallicity and age values
    met_age_sorted = [met_age_vals[i] for i in sorted_indexes]
    # Sort data frames
    dfs_sorted = [isoch_dataframes[i] for i in sorted_indexes]

    # Interpolate (if required)
    isochrones = {}
    for i, (met, age) in enumerate(met_age_sorted):
        # Generate entry in dictionary
        try:
            isochrones[met]
        except KeyError:
            isochrones[met] = {}
        try:
            isochrones[met][age]
        except KeyError:
            isochrones[met][age] = []

        df = dfs_sorted[i]
        # Only interpolate if there are extra points to add
        if len(df) >= N_interp:
            isoch_interp = df
        else:
            # Interpolate
            xx = np.linspace(0.0, 1.0, N_interp)

            # # Works but the binary sequence is really affected...
            # N_interp = 1000
            # N1, N2, N3, N4 = int(.1 * N_interp), int(.2 * N_interp), int(.3 * N_interp), int(.4 * N_interp)
            # xx = np.array(list(
            #     np.linspace(.0, .25, N1))
            #     + list(np.linspace(.25, .5, N2))
            #     + list(np.linspace(.5, .75, N3))
            #     + list(np.linspace(.75, 1, N4))
            # )

            xp = np.linspace(0.0, 1.0, len(df))
            df_new = {}
            for col in df.keys():
                df_new[col] = np.interp(xx, xp, df[col])
            isoch_interp = pd.DataFrame(df_new)

        isochrones[met][age].append(isoch_interp)

    return isochrones


def merge_ps_massini_check(mass_col: str, isochrones: dict) -> tuple[list, list]:
    """Combine photometric systems, and check initial masses.

    1. Combine photometric systems if more than one was used.
    2. Check that initial masses are equal across all systems

    The isochrone parameter 'mass_col' is assumed to be equal across
    photometric systems, for a given metallicity and age. We check here that
    this is the case.

    :param mass_col: Name of the mass column.
    :type mass_col: str
    :param isochrones: Dictionary of isochrones.
    :type isochrones: dict

    :raises ValueError: If initial mass values differ across photometric systems.

    :return: Ordered isochrones and metallicity-age array.
    :rtype: tuple[list, list]
    """
    met_age_arr, isochrones_ordered = [], []
    for met_k in isochrones.keys():
        met_vals = []
        for age_k, age_list in isochrones[met_k].items():
            met_age_arr.append([met_k, age_k])

            df_age_0 = age_list[0]
            # Check masses if more than one photometric system was used
            if len(age_list) > 1:
                for df_age_X in age_list[1:]:
                    mass_diff = abs(
                        df_age_0[mass_col].values - df_age_X[mass_col].values
                    ).sum()
                    if mass_diff > 0.001:
                        raise ValueError(
                            "Initial mass values differ across photometric systems"
                        )
                    # Drop 'mass_col' column from this photometric system
                    df_age_X = df_age_X.drop([mass_col], axis=1)
                    # Merge data frames and replace original one
                    df_age_0 = pd.concat([df_age_0, df_age_X], axis=1)

            met_vals.append(df_age_0)
        isochrones_ordered.append(met_vals)

    return isochrones_ordered, met_age_arr


def shape_isochrones(
    magnitude: str,
    color: tuple,
    color2: tuple | None,
    initial_mass: str,
    isochrones: list,
) -> tuple[np.ndarray, list]:
    """Reshape the isochrones array.

    isochrones.shape = Nz, Na, Ni, Nd
    Nz: number of metallicities
    Na: number of log(age)s
    Ni: number of interpolated values
    Nd: number of data columns --> mini + mag + 2 * colors (one mag per color)

    Return list structured as:

    theor_tracks = [m1, m2, .., mN]
    mX = [age1, age2, ..., ageM]
    ageX = [f,.., c1, c2,.., Mini, fb,.., c1b, c2b,.., Minib]

    where:
    N     : number of metallicities in grid
    M     : number of ages in grid
    f     : magnitude
    cX    : colors
    Mini  : initial mass

    # These columns are added later by the synthetic class:
    fb    : binary magnitude --
    cXb   : binary colors     |
    Minib : binary masses -----

    theor_tracks.shape = (Nz, Na, Nd, Ni)
    Nz: number of metallicities
    Na: number of log(age)s
    Nd: number of data columns
    Ni: number of interpolated values

    :param magnitude: Magnitude filter to load.
    :type magnitude: str
    :param color: Tuple with the two filters to generate the first color.
    :type color: tuple
    :param color2: Tuple with the two filters to generate the second color,
        defaults to None
    :type color2: tuple | None
    :param initial_mass: Name of the initial mass column.
    :type initial_mass: str
    :param isochrones: List of isochrones.
    :type isochrones: list

    :return: Reshaped isochrones array and list of color filters.
    :rtype: tuple[np.ndarray, list]
    """

    # Notice that Ni and Nd are inverted here compared to the shape of the final
    # 'theor_tracks' array
    Nz, Na, Ni, Nd = np.array(isochrones).shape

    all_colors = [color]
    if color2 is not None:
        all_colors.append(color2)
    N_colors = len(all_colors)

    # Array that will store all the interpolated tracks. The '2' makes room for the
    # magnitude and first color which are required. Adding 'N_colors' to that value
    # makes room for an optional second color, and the initial mass column.
    theor_tracks = np.zeros([Nz, Na, (2 + N_colors), Ni])

    # Store the magnitudes to generate the colors separately. Used only by the
    # binarity process
    color_filters = []

    for i, met in enumerate(isochrones):
        met_lst = []
        for j, df_age in enumerate(met):
            # Store magnitude
            theor_tracks[i][j][0] = df_age[magnitude]
            # Store colors
            cols_dict = {}
            for k, color_filts in enumerate(all_colors):
                f1 = df_age[color_filts[0]]
                f2 = df_age[color_filts[1]]
                # Store color. The '1' makes room for the magnitude that goes first.
                theor_tracks[i][j][1 + k] = f1 - f2

                # Individual filters for colors, used for binarity
                cols_dict[color_filts[0]] = f1
                cols_dict[color_filts[1]] = f2

            met_lst.append(cols_dict)

            # Add initial mass column to the end of the array
            theor_tracks[i][j][2 + N_colors - 1] = df_age[initial_mass]
        color_filters.append(met_lst)

    return theor_tracks, color_filters
