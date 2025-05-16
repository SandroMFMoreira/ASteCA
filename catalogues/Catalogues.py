import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import os

par_names = ['cluster', 'l', 'b', 'ra', 'dec', 'pmra', 'pmdec', 'e_pmra', 'e_pmdec', 'plx', 'e_plx', 'rv',
             'e_rv', 'logage', 'e_logage', 'FeH', 'iso_dist', 'e_iso_dist', 'av']


def get_catalogue(cat_names):
    valid_catalogues = ['Dias', 'Dias2002', 'CantatGaudin', 'Hunt', 'Cavallo', 'Almeida', 'Alfonso']
    cat_list = []

    # Get the directory of the current file (catalogues.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for cat_name in cat_names:
        if cat_name not in valid_catalogues:
            raise ValueError(f"Invalid catalogue name: {cat_name}. Must be one of {valid_catalogues}.")

        # Construct the path to the CSV file
        csv_file_path = os.path.join(current_dir, f'{cat_name}.csv')

        # Read the CSV file
        try:
            catalogue = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {csv_file_path}")

        if cat_name == 'Dias':
            catalogue.rename(columns={"_Glon": par_names[1],
                                      "_Glat": par_names[2],
                                      "Cluster": par_names[0],
                                      "RA_ICRS": par_names[3],
                                      "DE_ICRS": par_names[4],
                                      "pmRA": par_names[5],
                                      "pmDE": par_names[6],
                                      "e_pmDE": par_names[8],
                                      "e_pmRA": par_names[7],
                                      "Plx": par_names[9],
                                      "e_Plx": par_names[10],
                                      "RV": par_names[11],
                                      "e_RV": par_names[12],
                                      "[Fe/H]": par_names[15],
                                      "Dist": par_names[16],
                                      "e_Dist": par_names[17],
                                      "Av": par_names[18]
                                      }, inplace=True)

        elif cat_name == 'CantatGaudin':
            catalogue.rename(columns={"GLON": par_names[1],
                                      "GLAT": par_names[2],
                                      "Cluster": par_names[0],
                                      "RA_ICRS": par_names[3],
                                      "DE_ICRS": par_names[4],
                                      "pmRA": par_names[5],
                                      "pmDE": par_names[6],
                                      "e_pmDE": par_names[8],
                                      "e_pmRA": par_names[7],
                                      "AgeNN": par_names[13],
                                      "AVNN": par_names[18],
                                      "DistPc": par_names[16],
                                      }, inplace=True)

        elif cat_name == 'Hunt':
            catalogue.rename(columns={"name": par_names[0],
                                      "pmdec_error": par_names[8],
                                      "pmra_error": par_names[7],
                                      "parallax": par_names[9],
                                      "parallax_error": par_names[10],
                                      "radial_velocity": par_names[11],
                                      "radial_velocity_error": par_names[12],
                                      }, inplace=True)
            
            catalogue[par_names[16]] = catalogue['distance_50']
            catalogue[par_names[13]] = catalogue['log_age_50']
            catalogue[par_names[18]] = catalogue['a_v_50']

            catalogue = catalogue[(catalogue.kind == 'o') & (catalogue.cst > 5) & (catalogue.class_50 > 0.5) &
                                  (catalogue.mass_jacobi > 40) & (catalogue.p_jacobi > 0.5)]

        elif cat_name == 'Cavallo':
            catalogue.rename(columns={"Cluster": par_names[0],
                                      "RA": par_names[3],
                                      "DEC": par_names[4],
                                      }, inplace=True)

            catalogue[par_names[13]] = catalogue['logAge_50']
            catalogue[par_names[16]] = catalogue['dMod_50']
            catalogue[par_names[15]] = catalogue['FeH_50']
            catalogue[par_names[18]] = catalogue['Av_50']

            catalogue = catalogue[(catalogue.kind == 'o') & (catalogue.CMDclass > 0.5) &
                                  ((catalogue.quality == 0) | (catalogue.quality == 1))]

            icrs = SkyCoord(ra=catalogue[par_names[3]].values * u.degree, dec=catalogue[par_names[4]].values * u.degree,
                            distance=1000/catalogue[par_names[9]].values * u.pc, frame='icrs')

            galactic = icrs.transform_to('galactic')

            catalogue[par_names[1]] = galactic.l.value
            catalogue[par_names[2]] = galactic.b.value

        elif cat_name == 'Almeida':
            catalogue.rename(columns={"Cluster": par_names[0],
                                      "RA_ICRS": par_names[3],
                                      "DE_ICRS": par_names[4],
                                      "age": par_names[13],
                                      "e_age": par_names[14],
                                      "dist": par_names[16],
                                      "e_dist": par_names[17]
                                      }, inplace=True)

            pref_distance = catalogue.iso_dist*1000

            icrs = SkyCoord(ra=catalogue[par_names[3]].values * u.degree, dec=catalogue[par_names[4]].values * u.degree,
                            distance=pref_distance.values * u.pc, frame='icrs')

            galactic = icrs.transform_to('galactic')

            catalogue[par_names[1]] = galactic.l.value
            catalogue[par_names[2]] = galactic.b.value

        elif cat_name == 'Alfonso':
            catalogue.rename(columns={"Cluster": par_names[0],
                                      "RA_ICRS": par_names[3],
                                      "DE_ICRS": par_names[4],
                                      "pmRA": par_names[5],
                                      "pmDE": par_names[6],
                                      "Plx": par_names[9],
                                      "logAge": par_names[13]
                                      }, inplace=True)

        elif cat_name == 'Dias2002':
            catalogue.rename(columns={"Cluster": par_names[0],
                                      "RAJ2000_deg": par_names[3],
                                      "DEJ2000_deg": par_names[4],
                                      "Age": par_names[13],
                                      "pmRA": par_names[5],
                                      "pmDE": par_names[6],
                                      "RV": par_names[9],
                                      "Dist": par_names[16]
                                      }, inplace=True)

            pref_distance = catalogue.iso_dist

        if cat_name not in ('Almeida', 'Dias2002'):
            catalogue = catalogue[catalogue.plx > 0]
            catalogue['plx_dist'] = 1000 / catalogue.plx
            pref_distance = catalogue.plx_dist

        icrs = SkyCoord(ra=catalogue[par_names[3]].values * u.degree, dec=catalogue[par_names[4]].values * u.degree,
                        distance=pref_distance.values * u.pc, frame='icrs')

        galactic_cartesian = icrs.transform_to('galactic').cartesian
        galactocentric = icrs.transform_to('galactocentric')

        catalogue['z'] = galactic_cartesian.z.value
        catalogue['x'] = galactic_cartesian.x.value
        catalogue['y'] = galactic_cartesian.y.value

        catalogue['z_gc'] = galactocentric.z.value
        catalogue['x_gc'] = galactocentric.x.value
        catalogue['y_gc'] = galactocentric.y.value

        catalogue['r_sun'] = np.sqrt(catalogue.x**2 + catalogue.y**2)

        catalogue['age'] = (10 ** catalogue.logage) * 10 ** (-6)

        cat_list.append(catalogue)



    # Step 1: Create a mapping dictionary from catalogue_1 to catalogue_0
    #def create_name_mapping(catalogue_0, catalogue_1):
    #    # Reverse mapping from names in catalogue_1 to standard names in catalogue_0
    #    name_mapping = {}
    #
    #    # Create a reverse mapping based on catalogue_1
    #    for _, row in catalogue_1.iterrows():
    #        name_all = row['name_all'].split(',')
    #        standard_name = next((name for name in name_all if name in catalogue_0['cluster'].values), None)
    #        if standard_name:
    #            for name in name_all:
    #                if name not in name_mapping:
    #                    name_mapping[name] = standard_name
    #        else:
    #            # If no standard name is found, map each name to itself
    #            for name in name_all:
    #                if name not in name_mapping:
    #                    name_mapping[name] = name
    #
    #    return name_mapping
    #
    ## Create the mapping dictionary
    #name_mapping = create_name_mapping(cat_list[0], cat_list[2])
    #
    ## Step 2: Standardize names in the new DataFrame using the mapping dictionary
    #def standardize_names(name, mapping):
    #    return mapping.get(name, name)  # Default to original name if not in the mapping
    #
    #cat_list[2]['cluster'] = cat_list[2]['cluster'].apply(lambda x: standardize_names(x, name_mapping))

    return cat_list
