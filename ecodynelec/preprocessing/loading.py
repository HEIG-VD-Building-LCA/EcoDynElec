"""
Module to load production and cross-border flows from ENTSO-E
"""

import os
from time import time

import numpy as np
import pandas as pd

#################### Local functions
from ecodynelec.checking import check_frequency, check_regularity_frequency, check_residual_availability
from ecodynelec.preprocessing.autocomplete import get_steps_per_hour
from ecodynelec.preprocessing.auxiliary import load_gap_content
from ecodynelec.preprocessing.extracting import extract
from ecodynelec.preprocessing.residual import include_global_residual
from ecodynelec.progress_info import ProgressInfo


# +
# Module to load production and cross-border flows from Entso-E

# +

#####################################
# ####################################
# IMPORT DATA
# ####################################
# ####################################

# -

def import_data(ctry, start=None, end=None, freq="H", involved_countries=None, path_gap=None, sg_data=None,
                enr_prod_ch=None, net_exchange=False,
                path_gen=None, gen_preprocessed=None, path_imp=None, imp_preprocessed=None, savedir=None,
                residual_global=False, correct_imp=True,
                clean_data=True, n_hours=2, days_around=7, limit=.4, is_verbose=True,
                progress_bar: ProgressInfo = None):
    """
    Main function managing the import and pre-treatment of Entso-e production and cross-border flow data.

    Parameters
    ----------
        ctry: list
            list of countries to include in the computing (list)
        start:
            starting date, as str or datetime
        end:
            ending date, as str or datetime
        freq: str, default to 'H'
            frequency of time steps to consider
        involved_countries: list, default to None
            list of all countries involved, with the countries to include in the computing
            and their neighbours (to implement the exchanges with 'Other' countries)
        path_gap: str or None, default to None
            path to the file containing the information about the nature of the residual
            refer to parameter.path.gap for more information
        sg_data: pandas.DataFrame, default to None
            information from Swiss Grid
        enr_prod_ch: pandas.DataFrame, default to None
            Wind and solar production in Switzerland, as modeled with EcoDynElec-Enr-Model
            See Parameter.ch_enr_model_path for more information
        net_exchange: bool, default to False
            to simplify cross-border flows to net after resampling
        path_gen: str, default to None
            directory where raw Entso-e generation files are stored
        path_imp: str, default to None
            directory where raw Entso-e files for cross-border flow data are stored
        gen_preprocessed: str, default to None
            directory where preprocessed Entso-e generation files are stored
        imp_preprocessed: str, default to None
            directory containing the files for preprocessed cross-border flow data
        savedir: str, default to None
            directory to save information
        residual_global: bool, default to False
            to consider the production residual as produced electricity that can be exchanged with neighbour countries
        correct_imp: bool, default to False
            to replace cross-border flow of Entso-e for Swizerland with data from Swiss Grid
        clean_data: bool, default to True
            to enable automatic data cleaning / filling
        n_hours: int, default to 2
            max number of successive missing hours to be considered as occasional event
        days_around: int, default to 7
            number of days after and before a gap to consider to create a 'typical mean day'
        limit: float, default to 0.4
            max relative length of a gap to fill the data. Longer gaps are filled with zeros.
        is_verbose: bool, default to False
            to display information
        progress_bar: ProgressInfo, default to None
            to display a progress bar
    
    Returns
    -------
    pandas.DataFrame
        pandas DataFrame with all productions and all exchanges from all included countries.
    """

    t0 = time()

    if progress_bar:
        progress_bar = ProgressInfo("Import generation data...", 6, width='40%')
    ### GENERATION DATA
    Gen = import_generation(path_gen=path_gen, path_prep=gen_preprocessed, ctry=ctry, start=start, end=end,
                            savedir=savedir, n_hours=n_hours, days_around=days_around, limit=limit,
                            clean_generation=clean_data, is_verbose=is_verbose,
                            progress_bar=progress_bar)  # import generation data

    if progress_bar:
        progress_bar.progress('Adjust generation data...')
    Gen = adjust_generation(Gen, freq=freq, residual_global=residual_global, sg_data=sg_data, start=start, end=end,
                            path_gap=path_gap, enr_prod_ch=enr_prod_ch, is_verbose=is_verbose)  # adjust the generation data

    if progress_bar:
        progress_bar.progress('Import exchanges data...')
    ### EXCHANGE DATA
    Cross = import_exchanges(ctry=ctry, start=start, end=end, savedir=savedir,
                             n_hours=n_hours, days_around=days_around, limit=limit, clean_imports=clean_data,
                             path_imp=path_imp, path_prep=imp_preprocessed, freq=freq, is_verbose=is_verbose,
                             progress_bar=progress_bar)  # Imprt data
    if progress_bar:
        progress_bar.progress('Adjust exchanges data...')
    Cross = adjust_exchanges(Cross=Cross, neighbourhood=involved_countries, net_exchange=net_exchange, freq=freq,
                             sg_data=sg_data if correct_imp else None, is_verbose=is_verbose)

    if progress_bar:
        progress_bar.progress('Join data...')
    ### GATHER GENERATION AND EXCHANGE
    electric_mix = _join_generation_exchanges(Gen=Gen, Cross=Cross, is_verbose=is_verbose)

    if progress_bar:
        progress_bar.hide()
    if is_verbose: print("Import of data: {:.1f} sec".format(time() - t0))
    return electric_mix


# +

#####################################
# ####################################
# Import Generation
# ####################################
# ####################################

# -

def import_generation(ctry, start, end, path_gen=None, path_prep=None, savedir=None,
                      n_hours: int = 2, days_around: int = 7, limit: float = .4, clean_generation: bool = True,
                      is_verbose=False, progress_bar: ProgressInfo = None):
    """
    Function to import generation data from Entso-e information source.
    
    Parameters
    ----------
        ctry: list
            countries to incldue in the study (list)
        start:
            starting date, as str or datetime
        end:
            ending date, as str or datetime
        path_gen: str, default to None
            directory where raw Entso-e generation files are stored (str) [secondary path]
        path_prep: str, default to None
            directory where preprocessed Entso-e generation files are stored (str) [prioritary path]
            Where preprocessed files are saved if both `path_prep` and `path_gen` are passed and different.
        savedir: str, default to None
            directory path to save results (str, default: None)
        n_hours: int, default to 2
            max number of successive missing hours to be considered as occasional event
        days_around: int, default to 7
            number of days after and before a gap to consider to create a 'typical mean day'
        limit: float, default to 0.4
            max relative length of a gap to fill the data. Longer gaps are filled with zeros.
        clean_generation: bool, default to True
            to enable automatic data cleaning / filling
        is_verbose: bool, default to False
            to display information

    Returns
    -------
    dict
        processed generation data per country
    """
    path, savegen = _infer_paths(path_prep, path_gen, case='Generation')

    #######################
    ###### Generation data
    #######################

    if is_verbose: print("Load generation data...")
    # Selecton of right files according to the choice of countres
    if path == path_prep:
        files = {}
        for c in ctry:  # Gather prepared files per country
            try:
                files[c] = [f for f in os.listdir(path) if ((f.startswith(c)) & (f.endswith('MW.csv')))][0]
            except Exception as e:
                raise KeyError(f"No pre-processed generation data for {c}: {e}")

        Gen = {}  # Dict for the generation of each country

    elif path == path_gen:  # Just fill the Gen directly for row files
        Gen = extract(ctry=ctry, start=start, end=end, dir_gen=path, savedir_gen=savegen, save_resolution=savedir,
                      n_hours=n_hours, days_around=days_around, limit=limit, correct_gen=clean_generation,
                      is_verbose=is_verbose, progress_bar=progress_bar)  # if from raw files

    for c in ctry:  # Preprocess all files / data per country
        # Extract the generation data file
        if path == path_prep:  # Load preprocessed files
            Gen[c] = pd.read_csv(os.path.join(path, files[c]), index_col=0)  # Extraction of preprocessed files

        # Check and modify labels if needed
        Gen[c].columns = Gen[c].columns.str.rstrip() + " "  # (first remove if any, then) set additional ' ' at the end

        # Set indexes to time data
        Gen[c].index = pd.to_datetime(Gen[c].index, yearfirst=True)  # Convert index into datetime

        # Only select the required piece of information
        Gen[c] = Gen[c].loc[start:end]

        source = list(Gen[c].columns)  # production plants types
        if "Other " in source:  # Expected this label for "Other fossil" from ENTSO-E data
            source[source.index("Other ")] = "Other fossil "  # rename one specific column

        Gen[c].columns = [s.replace(" ", "_") + c for s in source]  # rename columns

    return Gen


# +

#####################################
# ####################################
# Adjust Generation
# ####################################
# ####################################

# -

def adjust_generation(Gen, freq='h', residual_global=False, start=None, end=None,
                      sg_data=None, path_gap=None, enr_prod_ch=None, is_verbose=False):
    """Function that leads organizes the data adjustment.
    It sorts finds and sorts missing values, fills it, resample the data and
    add a residual as global production
    
    Parameters
    ----------
        Gen: dict
            dict of dataFrames containing the generation for each country
        freq: str, default to 'H'
            time step durtion
        residual_global: bool, default to False
            whether to include the residual or not
        start: str or None, default to None
            starting date of the study
        end: str or None, default to None
            ending date of the study
        sg_data: pandas.DataFrame, default to None
            information from Swiss Grid
        path_gap: str or None, default to None
            path to the file containing the information about the nature of the residual
            refer to parameter.path.gap for more information
        enr_prod_ch: pandas.DataFrame, default to None
            Renewable energy production in Switzerland, as modeled with EcoDynElec-Enr-Model
            See parameter.ch_enr_model_path for more information
        is_verbose: bool, default to False
            whether to display information or not.
        
    Returns
    -------
    dict
        dict of pandas DataFrames containing modified Gen dict.
    """
    ### Resample data to the right frequence
    if is_verbose: print(f"\t4/{4 + int(residual_global)} - Resample exchanges to {freq} steps...")
    Gen = resample_data(Gen, freq=freq)

    ### Load gap data -> if Residual
    if residual_global:
        if is_verbose: print('Loading gap data')
        if enr_prod_ch is not None:
            delta = enr_prod_ch - Gen['CH'].loc[:, enr_prod_ch.columns]
            delta[delta < 0] = 0
        else:
            delta = None
        prod_gap = load_gap_content(path_gap=path_gap, start=start, end=end, freq=freq, enr_prod_residual_ch=delta)
    else:
        prod_gap = None

    ### Include the enr production as modeled with EcoDynElec-Enr-Model
    if enr_prod_ch is not None:
        # Check the availability of enr production data
        check_residual_availability(prod=Gen['CH'], residual=enr_prod_ch, freq=freq)
        if start.year < 2023:
            enr_prod_ch = enr_prod_ch.drop(columns=['Hydro_Pumped_Storage_CH', 'Hydro_Pumpage_CH']) # Using ENTSO-E data for 2023 and under
        # And include it
        Gen['CH'].loc[:, enr_prod_ch.columns] = enr_prod_ch

    ### Then include residual production
    if residual_global:
        Gen = include_global_residual(Gen=Gen, freq=freq, sg_data=sg_data, prod_gap=prod_gap,
                                      is_verbose=is_verbose)
    return Gen


# +

#####################################
# ####################################
# Import Exchanges
# ####################################
# ####################################

# -

def import_exchanges(ctry, start, end, path_imp=None, path_prep=None, savedir=None, freq='H',
                     n_hours: int = 2, days_around: int = 7, limit: float = .4, clean_imports: bool = True,
                     is_verbose=False, progress_bar: ProgressInfo = None):
    """
    Function to import the cross-border flows.
    Finds the useful files to load, load the data, group relevant information and adjust time step.
    
    Parameters
    ----------
        ctry: list
            countries to incldue in the study (list)
        start:
            starting date, as str or datetime
        end:
            ending date, as str or datetime
        path_imp: str, default to None
            directory where raw Entso-e exchange files are stored (str) [secondary path]
        path_prep: str, default to None
            directory where preprocessed Entso-e exchange files are stored (str) [prioritary path]
            Where preprocessed files are saved if both `path_prep` and `path_imp` are passed and different.
        savedir: str, default to None
            directory path to save results (str, default: None)
        freq: str, default to 'H'
            the frequency to consier
        n_hours: int, default to 2
            max number of successive missing hours to be considered as occasional event
        days_around: int, default to 7
            number of days after and before a gap to consider to create a 'typical mean day'
        limit: float, default to 0.4
            max relative length of a gap to fill the data. Longer gaps are filled with zeros.
        clean_generation: bool, default to True
            to enable automatic data cleaning / filling
        is_verbose: bool, default to False
            to display information

    Returns
    -------
    dict
        dict of pandas.DataFrame containing cross-border flows.
    """
    path, saveimp = _infer_paths(path_prep, path_imp, case='Exchanges')

    if is_verbose: print("Get and reduce importation data...")

    ### Files to consider
    if path == path_prep:
        files = {}
        for c in ctry:
            try:
                files[c] = [f for f in os.listdir(path) if ((f.startswith(c)) & (f.endswith('MW.csv')))][0]
            except Exception as e:
                raise KeyError(f'No pre-processed exchange data for "{c}": {e}')

        Cross = {}  # Dict for the generation of each country

    elif path == path_imp:  # Just fill the Gen directly for row files
        Cross = extract(ctry=ctry, start=start, end=end, dir_imp=path, savedir_imp=saveimp, save_resolution=savedir,
                        n_hours=n_hours, days_around=days_around, limit=limit, correct_imp=clean_imports,
                        is_verbose=is_verbose, progress_bar=progress_bar)  # if from raw files

    for i, c in enumerate(ctry):  # File extraction
        if path == path_prep:
            if is_verbose: print("\t{}/{} - {}...".format(i + 1, len(files), c))
            Cross[c] = pd.read_csv(os.path.join(path, files[c]), index_col=0)  # Extraction

        # Transform index in time data, then keeps only period of interest
        Cross[c].index = pd.to_datetime(Cross[c].index, yearfirst=True)  # Considered period only
        Cross[c] = Cross[c].loc[start:end]  # select right period

    return Cross


# +

#####################################
# ####################################
# Adjust exchanges
# ####################################
# ####################################

# -

def adjust_exchanges(Cross, neighbourhood, net_exchange=False, freq='H', sg_data=None, is_verbose=False):
    """
    Bring adjustments to the exchange data: add SwissGrid data, fill data,
    adjust frequency and set exchanges to net.
    
    Parameters
    ----------
        Cross: dict
            the Cross-border flow data, as dict of pandas DataFrame
        neighbourhood: list
            list of involved countries, as main countries or neighbours
        net_exchange: bool, default to False
            to consider net cross-border flows
        freq: str, default to 'H'
            time step
        sg_data: pandas.DataFrame, default to None
            information from Swiss Grid
        is_verbose: bool, default to False
            to display information
    
    Returns
    -------
    dict
        dict of pandas DataFrame with adjusted cross-border flow data.
    """
    ### ADJUST THE FREQUENCY AND CONVERT TO MWh
    if is_verbose: print(f"Resample exchanges to {freq} steps...")
    Cross = resample_data(Cross, freq=freq)

    ### ADJUST WITH SWISSGRID DATA (AT SWISS BORDER ONLY)
    if sg_data is not None:  # Adjust with SG data
        Cross = set_swissGrid(Cross, sg_data)

    ### CREATE THE 'OTHER' AND REMOVE UNUSED
    for c in Cross:
        other = [k for k in neighbourhood if k not in Cross.keys()]  # Label as 'other' all non-main selected countries
        Cross[c]['Other'] = Cross[c].loc[:, other].sum(axis=1).copy()  # Add the aggregated 'Other'

        involved = [k for k in neighbourhood if k in Cross.keys()] + ['Other']  # All neighbours involved in computation
        Cross[c] = Cross[c].loc[:, involved]  # Select only relevant information
        Cross[c] = Cross[c].rename(columns=lambda s: f"Mix_{s}_{c}")  # Rename columns

    ### DEAL WITH NET-RAW EXCHANGES
    if net_exchange:
        Cross = create_net_exchange(Cross)

    return Cross


#
###############################################################################
# ###########################
# # Set SwissGrid at Swiss border
# ###########################
# ###########################
#

def set_swissGrid(Cross, sg_data):
    """
    Function to replace the cross-border flow data of ENTSO-E by the cross-border flow data of SwissGrid. Data passed must be in 15min.
    
    Parameters
    ----------
        Cross: dict
            the Cross-border flow data, as dict of pandas DataFrame
        sg_data: pandas.DataFrame
            information from Swiss Grid
    
    Returns
    -------
    dict
        dict of pandas DataFrame with cross-border flow data for all the countries of the studied area.
    """
    #### Replace the data in the DataFrames
    places = ["AT", "DE", "FR", "IT"]  # Neighbours of Swizerland (as the function is only for Swizerland)

    for c in places:
        if c in Cross['CH'].columns:
            Cross["CH"].loc[:, c] = sg_data.loc[:, f"Mix_{c}_CH"]  # Swiss imports
        if c in Cross.keys():
            Cross[c].loc[:, 'CH'] = sg_data.loc[:, f"Mix_CH_{c}"]  # Swiss exports

    return Cross


#
###############################################################################
# ###########################
# # Create net exchanges
# ###########################
# ###########################
#

def create_net_exchange(Cross):
    """
    Adapt the cross-border flow to consider exchanges at each border and time step as net.
    Net exchange means that electricity can only go from A to B or from B to A, but not in 
    both directions at the same time.
    """
    # d = data.copy()
    ctry = list(Cross.keys())

    # Correction of the cross-border (turn into net exchanges) over each time step
    for i in range(len(ctry)):
        for j in range(len(ctry) - 1, i, -1):
            decide = (Cross[ctry[j]].loc[:, f"Mix_{ctry[i]}_{ctry[j]}"]
                      >= Cross[ctry[i]].loc[:, f"Mix_{ctry[j]}_{ctry[i]}"])  # direction
            diff = (Cross[ctry[j]].loc[:, f"Mix_{ctry[i]}_{ctry[j]}"]
                    - Cross[ctry[i]].loc[:, f"Mix_{ctry[j]}_{ctry[i]}"])  # exchange difference

            Cross[ctry[j]].loc[:, f"Mix_{ctry[i]}_{ctry[j]}"] = decide * diff  # if flow i to j --> +value
            Cross[ctry[i]].loc[:, f"Mix_{ctry[j]}_{ctry[i]}"] = (decide - 1) * diff  # if j to i <-- -value

    return Cross


# +

#####################################
# ####################################
# Join Generation Exchanges
# ####################################
# ####################################

# -

def _join_generation_exchanges(Gen, Cross, is_verbose=False):
    """Function to join generation and cross-border flow information."""

    if is_verbose: print("Gather generation and importation...")
    ### Union of all tables of importation and generation together
    Union = {}
    for f in Gen.keys():  # for all countries
        Union[f] = pd.concat([Gen[f], Cross[f]], axis=1)  # gathering of the data

    return pd.concat([Union[f] for f in Union.keys()], axis=1)  # Join all the tables together


# +

#####################################
# ####################################
# Infer paths
# ####################################
# ####################################

# -

def _infer_paths(path_prep, path_raw, case='Generation'):
    """Function to correctly set the paths to files and savin directory"""
    path, savegen = None, None

    if ((path_prep is None) & (path_raw is None)):
        raise KeyError(f"No path is given for {case} data.")
    elif ((path_prep is None) & (path_raw is not None)):  # Need raw file
        path = path_raw  # Then use raw file
    elif ((path_prep is not None) & (path_raw is None)):  # Got a file prepared
        path = path_prep  # Then use prepared file
    else:  # Both are not None
        path, savegen = path_raw, path_prep  # Then use raw and save in path_prep.

    return path, savegen


# +

#####################################
# ####################################
# Resample Data
# ####################################
# ####################################

# -

def resample_data(Data, freq):
    """
    Function that turns data from MW to MWh and adapts its frequency.
    The data is assumed to be in MW, in a table with 15min indexes.
    
    Parameters
    ----------
        Data: dict
            dict of DataFrames containing the generation data.
        freq: str
            the frequency (length of time step)
    
    Returns
    -------
    dict
        dict of pandas DataFrame wiht resampled and converted energy
    """
    ### VERIFY THE FREQUENCY
    check_frequency(freq)
    if check_regularity_frequency(freq):  # If frequency is regular for pandas
        ### Normal resampling and MW -> MWh conversion
        for f in Data:  # For all keys
            conv_factor = get_steps_per_hour(freq, dtype=float)  # Factor to convert MW to MWh
            # Resample Power and turn into energy
            Data[f] = (Data[f]
                       .resample(freq)
                       .mean()  # Mean works also to downscale
                       .interpolate()
                       .fillna(0)) / conv_factor

    else:  # Frequency is month or year
        ### Use Hours to convert MW -> MWh, then resample to correct frequency
        for f in Data:  # For all keys
            conv_factor = get_steps_per_hour('H')  # Factor to convert MW to MWh
            # Resample Power and turn into energy
            Data[f] = (((Data[f]
                         .resample('H')
                         .mean()  # Average as power still
                         .interpolate()
                         .fillna(0)
                         )
                        / conv_factor  # Turn MW -> MWh
                        )
                       .resample(freq)
                       .sum()  # Sum as energy now
                       )

    return Data
