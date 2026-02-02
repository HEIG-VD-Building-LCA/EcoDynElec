"""
Module containing the functions loading the downloaded data
from the ENTOS-E databases
"""

import concurrent
import os
from functools import partial
from time import time

import numpy as np
import pandas as pd

from ecodynelec.preprocessing.autocomplete import autocomplete
from ecodynelec.progress_info import ProgressInfo


# +

#################
#################
# ## Extract
##############

# -


def extract(ctry: list, start=None, end=None, dir_gen=None, dir_imp=None, correct_gen: bool = True,
            correct_imp: bool = True,
            savedir_gen: str = None, savedir_imp: str = None, save_resolution: str = None,
            n_hours: int = 2, days_around: int = 7, limit: float = .4, is_verbose=False,
            progress_bar: ProgressInfo = None):
    """Extracts all the data at once. Master function of the module.

    Parameters
    -----------
        ctry: list
            list of countries to involve in the computation
        start:
            starting date, as str or datetime
        end:
            ending date, as str or datetime
        dir_gen: str, default is None
            path to local directory with ENTSO-E generation data files
        dir_imp: str, default is None
            path to local directory with ENTSO-E exchange data files
        correct_gen: bool, default to True
            to auto-complete the generation data
        correct_imp: bool, default to True
            to auto-complete the exchange data
        savedir_gen: str, default to None
            directory to save the processed generation data.
        savedir_imp: str, default to None
            directory to save the processed exchange data.
        save_resolution: str, default to None
            directory to save information about frequency of each extracted time series
        n_hours: int, default to 2
            max number of hours of missing data in a row to consider a gap as short gap
            and fill it with linear interpolation
        days_around: int, default to 7
            number of days before and after a long gap to build an average day to infer values.
        limit: float, default to 0.4
            max size of gap relative to the whole series to auto-complete. Gaps larger are
            filled with zeros
        is_verbose: bool, default to False
            to display information

    Returns
    -------
    dict
        Generation data for each country, if `dir_gen` is not `None`
    dict
        Importation data for each country, if `dir_imp` is not `None`

    """
    t0 = time()
    if os.path.isdir(r"{}".format(dir_gen)):
        if is_verbose: print("\tGeneration data.")
        Gen = create_per_country(path_dir=dir_gen, case='generation', start=start, end=end, ctry=ctry,
                                 savedir=savedir_gen,
                                 savedir_resolution=save_resolution, is_verbose=is_verbose,
                                 n_hours=n_hours, days_around=days_around, limit=limit, correct_data=correct_gen,
                                 progress_bar=progress_bar)

    if os.path.isdir(r"{}".format(dir_imp)):
        if is_verbose: print("\tCross-border flow data.")
        Imp = create_per_country(path_dir=dir_imp, case='import', start=start, end=end, ctry=ctry, savedir=savedir_imp,
                                 savedir_resolution=save_resolution, is_verbose=is_verbose,
                                 n_hours=n_hours, days_around=days_around, limit=limit, correct_data=correct_imp,
                                 progress_bar=progress_bar)

    if is_verbose: print("\tExtraction time: {:.2f} sec.".format(time() - t0))

    if [dir_gen, dir_imp].count(None) == 2:
        raise KeyError("No files passed to extract data.")
    elif ((dir_gen is not None) & (dir_imp is None)):
        return Gen
    elif ((dir_gen is None) & (dir_imp is not None)):
        return Imp
    else:
        return Gen, Imp


# +

#################
#################
# ## Create per country
##############

# -


def create_per_country(path_dir: dict, case: str, start=None, end=None, ctry: list = None, savedir: str = None,
                       savedir_resolution: str = None,
                       n_hours: int = 2, days_around: int = 7, limit: float = .4, correct_data: bool = True,
                       is_verbose=False, progress_bar: ProgressInfo = None):
    """Extracts all the data for every country.

    Parameters
    -----------
        path_dir: str, default is None
            path to local directory with ENTSO-E data files
        case: str
            'generation' or 'import' to select the type of data to expect.
        start:
            starting date, as str or datetime
        end:
            ending date, as str or datetime
        ctry: list, default to None
            list of countries to involve in the computation
        savedir: str, default to None
            directory to save the processed data.
        save_resolution: str, default to None
            directory to save information about frequency of each extracted time series
        n_hours: int, default to 2
            max number of hours of missing data in a row to consider a gap as short gap
            and fill it with linear interpolation
        days_around: int, default to 7
            number of days before and after a long gap to build an average day to infer values.
        limit: float, default to 0.4
            max size of gap relative to the whole series to auto-complete. Gaps larger are
            filled with zeros
        correct_data: bool, default to True
            to auto-complete the data
        is_verbose: bool, default to False
            to display information

    Returns
    -------
    dict
        Transformed data for each country.

    """

    # Obtain parameter set for the specific case
    destination, origin, data, area = get_parameters(case)

    # Import content of raw files
    df = load_files(path_dir, start, end, destination, origin, data, area, is_verbose=is_verbose,
                    progress_bar=progress_bar)

    # Get auxilary information
    prod_units = get_origin_unit(df, origin)  # list of prod units or origin countries
    time_line = get_time_line(unique_dates=df.DateTime.unique())  # time line of the data

    # Format and save files for every country
    Data = {}  # Data storage object
    t0 = time()
    for i, c in enumerate(ctry):  # for all countries
        if is_verbose: print(f"Extracting {case} for {c} ({i + 1}/{len(ctry)})...", end="\r")
        if progress_bar: progress_bar.set_sub_label(f"Extracting {case} for {c} ({i + 1}/{len(ctry)})...")
        # Get data from the country and sort by date
        country_data = df[df.loc[:, destination] == c].drop_duplicates().sort_values(by="DateTime")

        # Select only the Generation data, then resample in 15min and interpolate (regardless of ResolutionCode)
        Data[c] = country_data.pivot(index='DateTime', columns=origin, values=data)
        Data[c]['Hydro_Pumpage'] = 0.0
        del country_data  # free memory space

    ### AUTOCOMPLETE THE DATA
    if progress_bar:
        progress_bar.progress()
        progress_bar.set_sub_label(f"Autocompleting {case} data...")
    Data, resolution = autocomplete(Data, n_hours=n_hours, days_around=days_around, limit=limit,
                                    ignore=(not correct_data), is_verbose=is_verbose)
    if savedir_resolution is not None:
        resolution.to_csv(f'{savedir_resolution}resolution_{case}.csv', index=True)

    if progress_bar: progress_bar.set_sub_label('Formatting data...')
    ### ADD ALL COLUMNS AND FILL REST WITH ZERO
    for i, c in enumerate(ctry):
        # Add all columns
        country_detailed = pd.DataFrame(None, columns=prod_units, index=time_line,
                                        dtype='float32').resample('15min').asfreq()  # init. with NaNs
        country_detailed.loc[:, Data[c].columns] = Data[c]  # fill with data

        # Save files
        if savedir is not None:
            country_detailed.to_csv(f"{savedir}{c}_{case}_MW.csv")
        Data[c] = country_detailed.copy()  # Store information in variables (with non-missing NaNs)
        del country_detailed  # free memory space
    if is_verbose: print(f"Extraction raw {case}: {time() - t0:.2f} sec.             ")
    if progress_bar: progress_bar.reset_sub_label()
    return Data


# +

#################
#################
# ## Load files
##############

# -


def load_files(path_dir, start=None, end=None, destination=None, origin=None, data=None, area=None, case=None,
               is_verbose=False, progress_bar: ProgressInfo = None):
    """Load the ENTSO-E data and concatenate the information
    """
    if None in [destination, origin, data, area]:
        if case is None:
            raise KeyError("Missing information to load files: what 'case' is it ?")
        else:
            destination, origin, data, area = get_parameters(case)

    # Types to reduce size of data table
    column_types = {'Year': 'int8', 'Month': 'int8', 'Day': 'int8', 'FlowValue': 'float32',
                    'ActualGenerationOutput': 'float32', 'ActualConsumption': 'float32'}
    if type(data) == dict:
        useful = [destination, origin] + list(data.values())  # columns to keep
        date_col = [data['start'], data['end']]
        area_level = 'CTA'
        status_col = data['status']
    else:
        useful = ['DateTime', destination, 'ResolutionCode', origin, data]  # columns to keep
        date_col = ['DateTime']
        area_level = 'CTY'
        status_col = None

    files = _get_file_list(start=start, end=end, path_dir=path_dir)  # gather file pathways
    if len(files) == 0:
        raise FileNotFoundError(f"No file found in {path_dir} between {start} and {end}. Make sure to download the required entsoe data.")
    t0 = time()
    if progress_bar: progress_bar.set_sub_label(f"Extract {len(files)} files...")
    # Single processing
    if len(files) <= 2:  # for 2 files or less: use a for-loop
        container = []
        for i, f in enumerate(files):  # For all files
            if is_verbose: print(f"Extract file {i + 1}/{len(files)}...", end="\r")
            container.append(load_single_files(f, column_types, area, useful, date_col=date_col, area_level=area_level, status_col=status_col))

    # Multi-processing
    else:
        if is_verbose: print(f"Extract {len(files)} files...", end='\r')
        subfunc = partial(load_single_files, column_types=column_types, area=area, useful=useful, date_col=date_col, area_level=area_level, status_col=status_col)
        container = []
        with concurrent.futures.ProcessPoolExecutor() as pool:
            for d in pool.map(subfunc, files):
                container.append(d)

    # Concatenates all files
    if is_verbose: print("Concatenate all files...", end="\r")
    if progress_bar: progress_bar.set_sub_label('concatenate all files...')
    df = pd.concat(container)
    del container  # free memory space

    if is_verbose: print(f"Data loading: {round(time() - t0, 2)} sec")
    if is_verbose: print(f"Memory usage table: {round(df.memory_usage().sum() / (1024 ** 2), 2)} MB")
    if progress_bar: progress_bar.reset_sub_label()
    return df


# +

#################
#################
# ## Set Time
##############

# -


def _get_file_list(start, end, path_dir):
    """Set the list of files to load.
    If start is None: take all before end.
    If end is None: take all after start.
    """
    list_files = sorted(os.listdir(path_dir))

    if ((start is None) & (end is None)):  # Both -> Take all
        start = list_files[0][:7].replace("_", "-") + "-01"  # Date of first file
        end = list_files[-1][:7].replace("_", "-") + "-01"  # Date of last file
    elif end is None:  # Start but no end: until now
        end = list_files[-1][:7].replace("_", "-") + "-01"  # Date of last file
    elif start is None:  # End but no start -> All until end
        start = list_files[0][:7].replace("_", "-") + "-01"  # Date of first file

    all_months = pd.period_range(start=start, end=end, freq='M')
    possibles = [f"{a.year}_{a.month:02d}_" for a in all_months]
    list_files = [path_dir + f for f in list_files if f[:8] in possibles]

    return list_files


# +

#################
#################
# ## Load single file
##############

# -


def load_single_files(file_path, column_types, area, useful, date_col=['DateTime'], area_level='CTY', status_col=None):
    """Load the ENTSO-E data for a single file
    """
    # Extract the information
    d = pd.read_csv(file_path, sep="\t", encoding='utf-8', parse_dates=date_col, dtype=column_types)

    # Only select country level & Useful columns
    d = d.loc[d.loc[:, area] == area_level, useful]
    if status_col:
        d = d.loc[(d[status_col] != 'Cancelled') & (d[status_col] != 'Withdrawn')]
        d['FromFile'] = file_path
    return d


# +

####################
####################
# ## Get origin unit
##############

# -


def get_origin_unit(df, origin):
    """Gets ordered list of sources (origin countries or production units)"""
    return np.sort(df.loc[:, origin].unique())


# +

#################
#################
# ## Get time line
##############

# -


def get_time_line(unique_dates):
    """Gets the time line and corrects it if needed"""
    # Get the time line
    time_line = pd.DatetimeIndex(np.sort(unique_dates))

    # Add last hour in 15min, if not already here
    if time_line[-1].minute == 0:
        time_line = pd.DatetimeIndex(time_line.to_list() + [time_line[-1] + pd.Timedelta("45T")])

    return time_line


# +

#################
#################
# ## Get parameters
##############

# -

def get_parameters(case):
    """Function used to define parameters for later code"""
    if case == 'import':
        destination = 'InMapCode'
        origin = 'OutMapCode'
        data = 'FlowValue'
        area = 'OutAreaTypeCode'

    elif case == 'generation':
        destination = 'MapCode'
        origin = 'ProductionType'
        data = 'ActualGenerationOutput'
        area = 'AreaTypeCode'

    elif case == 'capacities':
        destination = 'MapCode'
        origin = 'ProductionType'
        data = 'AggregatedInstalledCapacity'
        area = 'AreaTypeCode'

    elif case == 'unavailabilities_gen' or case == 'unavailabilities_prod':
        destination = 'MapCode'
        origin = 'ProductionType'
        data = {
            'nominal': 'InstalledCapacity',
            'available': 'AvailableCapacity',
            'start': 'StartOutage',
            'end': 'EndOutage',
            'version': 'Version',
            'status': 'Status',
            'id': 'PowerResourceEIC',
            'mrid': 'MRID',
            'type': 'Type'
        }
        area = 'AreaTypeCode'

    else:
        raise KeyError(f"case {case} not understood.")

    return destination, origin, data, area
