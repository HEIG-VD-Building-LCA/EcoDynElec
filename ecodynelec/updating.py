"""
Module, whose only objective is to update and copy the data necessary for the software to work correctly.
The source can be anywhere and is defaulted to the support_files (assuming the file is used from within
the repository and not after an install. If installed, the files must be specified each time). Updating
the information from SwissGrid uses source codes from https://swissgrid.ch. It is much more intensive and
requires a specific parametrization of the updater function.
"""
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from time import time

import numpy as np
import pandas as pd
from pandas import DataFrame

from ecodynelec.preprocessing.auxiliary import get_default_file, read_ofen_pdf_file
from ecodynelec.preprocessing.enr_residual_utils import get_enr_data_from_pronovo_ec
from ecodynelec.preprocessing.loading import adjust_generation, import_generation
from ecodynelec.preprocessing.sfoe_extracting import updating_ofen_data


def update_all(path_dir=None, path_swissGrid=None, is_verbose=False):
    """
    Updates all possible software files at once.
    
    Parameter
    ----------
        path_dir: str, optional
            path to the directory containing updated files. Typically, this is the `support_files/`
            directory of the cloned git repository of EcoDynElec. The directory must contain ALL
            the files of interest, otherwise the execution is aboarded.
            If None, an attempt to use a default path is made, with no promises.
        path_swissGrid: str, optional
            path to a directory containting files downloaded from https://swissgrid.ch. If not given,
            the swiss-grid information is not updated. This will have no impact if no path_dir is given.
            To solely update the swiss-grid data, please use the `update_sg` function.
        is_verbose: bool, optional
            to display information. Default to False.
    """
    ### Verify the path_dir
    if path_dir is None:
        ### Try to reach a default directory
        path_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'support_files')

    if not os.path.isdir(path_dir):  # Verify if path is valid
        raise FileNotFoundError(
            f"Need to specify a directory containing updated files to save them into software files.")

    ### Verify the names if using path_dir
    expected = ["Neighbourhood_EU.csv", "Unit_Impact_Vector.csv", "SFOE_data.csv", "Residual_model.xlsx"]
    files = os.listdir(path_dir)
    if not all(exp in files for exp in expected):
        missing = [f for f in expected if f not in files]
        raise FileNotFoundError(
            f"The following files are missing. Please use individual update functions for individual updates. {missing}")

    ### Process all common updates
    update_neighbours(os.path.join(path_dir, "Neighbourhood_EU.csv"))
    if is_verbose: print(f"Updated Neighbourhood file")
    update_UIVector(os.path.join(path_dir, "Unit_Impact_Vector.csv"))
    if is_verbose: print(f"Updated UI vector file")
    update_UVEKVector(path_dir, is_verbose)
    if is_verbose: print(f"Updated UVEK UI vector file")
    update_SFOE(os.path.join(path_dir, "ofen_data"))
    update_Losses(os.path.join(path_dir, "SFOE_data.csv"))
    if is_verbose: print(f"Updated Losses file")
    update_residual_share(os.path.join(path_dir, "Residual_model.xlsx"), save=True)
    if is_verbose: print(f"Updated Residual share file")

    ### Go on with SwissGrid
    if path_swissGrid is not None:
        if os.path.isdir(path_swissGrid):
            content = os.listdir(path_swissGrid)
            if len(content) > 0:
                if all(os.path.splitext(f)[1].startswith(".xls") for f in content):
                    update_sg(path_dir=os.path.abspath(path_swissGrid), save=True, is_verbose=is_verbose)
                else:
                    raise KeyError(f"Not all files are source .xls or .xlsx in directory {path_swissGrid}")
            else:
                raise FileNotFoundError(f"{path_swissGrid} is an empty directory...")
        else:
            raise FileNotFoundError(f"{path_swissGrid} is no directory.")
    elif is_verbose:
        print("SwissGrid files were not updated.")

    ### Go on with ENR data
    update_enr_data_from_pronovo(path_dir, verbose=is_verbose)
    if is_verbose: print(f"Updated ENR data files")
    return


###################################
####### GENERAL UPDATES ###########
###################################

def update_copy(path, name):
    ### Verify
    if not os.path.isfile(path): raise FileNotFoundError(f"Could not find {path}")

    ### Where to save
    savepath = get_default_file(name)

    ### Copy
    shutil.copy(path, savepath)


def update_neighbours(path):
    update_copy(path, "Neighbourhood_EU.csv")


def update_UIVector(path):
    update_copy(path, "Unit_Impact_Vector.csv")


def update_UVEKVector(path, is_verbose):
    mapping_path = os.path.join(path, "Mapping_UVEK.csv")
    UI_vector_path = os.path.join(path, "Unit_Impact_Vector.csv")
    UVEK_path = os.path.join(path, "UVEK_LCA.xlsx")
    vector = impact_mapping_matching(UVEK_path, UI_vector_path, mapping_path, is_verbose=is_verbose)
    vector.to_csv(os.path.join(path, "UVEK_Unit_Impact_Vector.csv"), index=True)
    update_copy(os.path.join(path, "UVEK_Unit_Impact_Vector.csv"), "UVEK_Unit_Impact_Vector.csv")


def update_SFOE(path):
    DF = []
    keys = ['annee', 'mois']
    for file in os.listdir(path):
        if "2023" in file:
            continue  # skip 2023

        if not file.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(path, file)
        DF.append(updating_ofen_data(pdf_path))
    df = DF[0].merge(DF[1][keys].drop_duplicates(), on=keys, how="left", indicator=True) \
        .query("_merge == 'left_only'") \
        .drop(columns="_merge")
    df_all = pd.concat([df, DF[1]], ignore_index=True)
    df_all.to_csv(os.path.join(os.path.dirname(path), 'SFOE_data.csv'), index=False, encoding="utf-8")


def update_Losses(path):
    update_copy(path, "SFOE_data.csv")


def update_residual_share(path, save=True):
    """Extracts the data relative to residual share estimate and save it in software files"""
    ### Verification
    # Error will be raised by pandas if needed

    ### Extraction
    interest = {'Centrales au fil de l’eau': "Hydro_Run-of-river_and_poundage_Res",
                'Centrales à accumulation': "Hydro_Water_Reservoir_Res",
                'Centrales therm. classiques et renouvelables': "Other_Res"}
    df = pd.read_excel(path, header=60, index_col=0).loc[interest.keys()].rename(index=interest)
    df = df.T
    df.index = pd.to_datetime(df.index, dayfirst=True, errors='coerce')

    ### Saving
    if save:
        savepath = get_default_file("Share_residual.csv")
        df.to_csv(savepath, index=True)

    return df


def extract_entsoe_daily_generation_for_residuals(config, path_dir=None, n_hours=2, days_around=7, limit=.4, save=True,
                                                  is_verbose=False):
    """
    Extracts the daily entsoe generation data that will be used for the residual share estimation. The extracted data
    corresponds to the days in the 'Redisual_model.xlsx' file. The data is saved in a file named
    'daily_entsoe_data_for_residual.csv' in path_dir.

   **Note that the required entsoe data isn't downloaded automatically. See downloading.py for more information.**

    Parameters
    ----------
    config: ecodynelec.Parameter or str
        a set of configuration parameters to govern the computation,
        either as Parameter object or str pointing at an xlsx file.

        Only the 'target', 'start', 'end', 'path.generation' and 'path.exchanges' parameters are used.
    path_dir: str, optional
        path to the directory containing updated files. Typically, this is the `support_files/`
        directory of the cloned git repository of EcoDynElec. The directory must contain ALL
        the files of interest, otherwise the execution is aborted.
        If None, an attempt to use a default path is made, with no promises.
    n_hours: int, default to 2
        max number of successive missing hours to be considered as occasional event
    days_around: int, default to 7
        number of days after and before a gap to consider to create a 'typical mean day'
    limit: float, default to 0.4
        max relative length of a gap to fill the data. Longer gaps are filled with zeros.
    save: bool, optional
        to decide whether to overwrite the software files with the new extracted data
        default is True.
    is_verbose: bool, optional
        to display information. Default to False.

    Returns
    --------
        A pandas.DataFrame containing the daily entsoe data that will be used for the residual share estimation.
    """

    ### Verify the path_dir
    if path_dir is None:
        ### Try to reach a default directory
        path_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'support_files')

    if not os.path.isdir(path_dir):  # Verify if path is valid
        raise FileNotFoundError(
            f"Need to specify a directory containing updated files to save them into software files.")

    start = config.start
    end = config.end
    freq = 'D'
    target = 'CH'
    ### Get the days when we need to search for entsoe data
    if is_verbose: print(f"Extracting days from Residual_model.xlsx")
    residual_model = pd.read_excel(path_dir + '/Residual_model.xlsx', header=16, index_col=0)
    dates = residual_model.columns
    print(dates)
    dates = pd.DatetimeIndex(dates)
    date_range = pd.date_range(start=start, end=end, freq=freq)
    dates = [date.strftime('%Y-%m-%d') for date in dates if date in date_range]
    ### Compute generation data for all days in the range
    if is_verbose: print(f"Computing {target} generation data")
    generation_per_day = import_generation(path_gen=config.path.generation, path_prep=config.path.exchanges,
                                           ctry=[target],
                                           start=start, end=end,
                                           savedir=config.path.savedir, n_hours=n_hours, days_around=days_around,
                                           limit=limit,
                                           clean_generation=False, is_verbose=is_verbose)  # import generation data
    generation_per_day = adjust_generation(generation_per_day, freq=freq, residual_global=False, sg_data=None,
                                           path_gap=None, enr_prod_ch=None,
                                           is_verbose=is_verbose)  # adjust the generation data
    ### And save it
    if save:
        if is_verbose: print(f"Saving {target} generation data")
        csv_content = ''
        csv_content += ';'
        for date in dates:
            csv_content += date + ';'
        csv_content += '\n'
        for production in generation_per_day[target].columns:
            csv_content += production.replace(f'_{target}', '').replace('_', ' ') + ';'
            for date in dates:
                csv_content += str(generation_per_day[target][production].loc[date] / 1000).replace('.', ',') + ';'
            csv_content += '\n'

        csv_file = open(path_dir + '/daily_entsoe_data_for_residual.csv', 'w')
        csv_file.write(csv_content)
        csv_file.close()
    return generation_per_day


def extract_ofen_typical_days_for_residual(year, post_process_fun, path_dir=None, save=True):
    """
    Extracts the typical days from the OFEN data that will be used for the residual share estimation. The extracted data
    corresponds to the days in the 'Redisual_model.xlsx' file. The data is saved in a file named
    'ofen_data/daily_ofen_data_for_residual_year.csv' in path_dir.

   **The input data is the annual report of the OFEN, in a pdf that should be named 'year.pdf' and in the directory path_dir'/ofen_data'.**

    Parameters
    ----------
    year: str
        the year of the OFEN report to use
    post_process_fun: function
        A function to post-process the pdf table data.
        It takes a list of columns and should return the modified list of columns.
        Refer to auxiliary.read_ofen_pdf_file for details.
    path_dir: str, optional
        path to the directory containing input and output files (that should be in a 'ofen_data' sub directory).
        Typically, this is the `support_files/` directory of the cloned git repository of EcoDynElec.
        If None, an attempt to use a default path is made, with no promises.
    save: bool, optional
        to decide whether to overwrite the software files with the new extracted data
        default is True.

    Returns
    --------
        A dict containing the ofen data of the typical days that will be used for the residual share estimation.
    """

    ### Verify the path_dir
    if path_dir is None:
        ### Try to reach a default directory
        path_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'support_files')

    if not os.path.isdir(path_dir):  # Verify if path is valid
        raise FileNotFoundError(
            f"Need to specify a directory containing updated files to save them into software files.")

    ### Read the pdf file and retrieve the typical days
    path_dir = os.path.join(path_dir, 'ofen_data')
    file = os.path.join(path_dir, f'{year}.pdf')
    if not os.path.isfile(file):
        raise FileNotFoundError(
            f"File {file} doesn't exist. Please download it from OFEN website then rename and put it in {path_dir}.")
    ofen_data = read_ofen_pdf_file(file, post_process_fun=post_process_fun)
    ### Then save them
    if save:
        file = f'{path_dir}/daily_ofen_data_for_residual_{year}.csv'
        print(f"Saving {year} ofen generation data to {file}")
        csv_content = ''
        csv_content += ';'
        for date in ofen_data.keys():
            csv_content += date + ';'
        csv_content += '\n'
        production_means = ['Centrales au fil de l’eau', 'Centrales à accumulation', 'Centrales nucléaires',
                            'Centrales therm. classiques et renouvelables', 'Excédent d’importation',
                            'Fourniture totale', 'Excédent d’exportation', 'Consommation du pays avec pompage',
                            'Pompage d’accumulation', 'Consommation du pays sans pompage']
        for i in range(0, len(production_means)):
            csv_content += production_means[i] + ';'
            for date in ofen_data.keys():
                csv_content += str(ofen_data[date][i]).replace('.', ',').replace('-', '0').replace('–', '0') + ';'
            csv_content += '\n'
        csv_file = open(file, 'w')
        csv_file.write(csv_content)
        csv_file.close()
    return ofen_data


###################################
##### SPECIFIC TO SWISS-GRID ######
###################################

def update_sg(path_dir=None, path_files=None, save=True, is_verbose=False):
    """
    Function to update the SwissGrid values from source files.
    It requires the source files from swissgrid.ch to be downloaded manually.
    The files are downloaded in parallel to save time, as .xlsx are
    particularly long to load.
    
    The data is returned and automatically overwrites previous version in the
    software files if save=True.
    
    Parameters
    -----------
        path_dir: str, optional
            path do directory containing EXCLUSIVELY the files from swissgrid.ch
            Either path_dir or path_files must be specified.
        path_files: list-like, optional
            list of paths to the files downloaded from swissgrid.ch on local computer
            Either path_dir or path_files must be specified.
        save: bool, optional
            to decide whether to overwrite the software files with the new extracted data
            default is True.
        is_verbose: to display information
        
    Returns
    --------
        pandas.DataFrame
    """
    ### List the elements / files
    if (path_dir is None) & (path_files is None):
        raise FileNotFoundError("Needs to specify a directory or a list of files")

    elif path_dir is not None:
        files = [os.path.abspath(os.path.join(path_dir, f))
                 for f in sorted(os.listdir(path_dir))]
    else:
        files = path_files

    ### Verification
    faulty = [f for f in files if not os.path.isfile(f)]
    if faulty: raise FileNotFoundError(f"Following files were not found: {faulty}")

    ### Extract data
    if is_verbose: print("Extracting SwissGrid files...")
    t0 = time()
    whole_sg = []
    with ProcessPoolExecutor() as pool:
        for table in pool.map(_prepare_sg_year, files):
            whole_sg.append(table)
    whole_sg = pd.concat(whole_sg, axis=0).sort_index()
    if is_verbose:
        print(f"\tLoaded {len(files)} tables: {time() - t0:.2f} sec")
        print(f"\tMemory usage: {whole_sg.memory_usage().sum() / 1024 ** 2:.1f} MB")

    ### Save the data
    if save:
        target = get_default_file("SwissGrid_total.csv")
        if is_verbose: print(f"Re-writing {target}...")
        ## Build the path to file
        whole_sg.to_csv(target, index=True)

    ### Return
    if is_verbose: print(f"Updating SG total: {time() - t0:.2f} sec")
    return whole_sg


def _rename_columns_sg(columns):
    new_cols = {}
    for c in columns:
        if 'energy consumed by end users' in c:
            new_cols[c] = "Consommation_CH"
        elif 'energy production' in c:
            new_cols[c] = 'Production_CH'
        elif 'energy consumption' in c:
            new_cols[c] = 'Consommation_Brut_CH'
        elif "->" in c:
            new_cols[c] = c.strip()[-6:]
    return new_cols


def _set_index_sg(idx):
    start = pd.to_datetime(idx[0])
    new_idx = pd.date_range(start=start, freq='15T', periods=len(idx))
    return new_idx - pd.Timedelta('15min')


def _prepare_sg_year(path, year=None):
    if os.path.isfile(path):
        sg_file = path
    else:
        if year is None:
            raise ValueError("If path does not point at a file, a year value is needed")
        elif any(isinstance(year, k) for k in (int, float, np.number)):
            year = str(int(year))
        elif not isinstance(year, str):
            raise TypeError(f"year must be a string or a number. Not {year}")

        sg_file = os.path.join(path, [f for f in os.listdir(path) if year in f][0])

    ### Import data
    col_selection = 'A:D,K:R'
    data = pd.read_excel(sg_file, sheet_name='Zeitreihen0h15', header=0, index_col=0,
                         parse_dates=False, usecols=col_selection).drop(index='Zeitstempel', errors='ignore')

    ### Clean data
    data.index = _set_index_sg(data.index)
    return data.rename(columns=_rename_columns_sg(data.columns)).astype("int32")


##############################################
##### SPECIFIC TO ENERGY CHARTS-PRONOVO ######
##############################################


def update_enr_data_from_pronovo(path_dir=None, output_file=None, verbose=False):
    """
    | Function to update the renewable electricity data from Pronovo and Energy Charts.
    | This updates the `enr_prod_2016-2022_completed.csv` file in the `support_files` directory.
    | The source files should be manually downloaded following the procedure described in the documentation of the
        :ref:`ecodynelec.preprocessing.enr_residual_utils <pronovo-and-energycharts-data-downloading>` module.

    | The source files should be placed in the `path_dir` directory :
    - a 'pronovo_data' sub-directory should contain the files from Pronovo ('prod_year' directories containing the .csv
      files downloaded on the Pronovo's website, AND a 'EC_Solar_year.csv' EnergyCharts solar production file to scale the
      hourly Pronovo data to the real daily production given by EnergyCharts
      (see :ref:`ecodynelec.preprocessing.enr_residual_utils <pronovo-and-energycharts-data-downloading>` module documentation)
    - a 'energy_charts_data' sub-directory should contain the files from Energy Charts
    - a 'enr_prod_2016-2019.csv' file should be in the `path_dir` directory
    | This file is generated using the Ecd-EnrModel project. It contains predicted solar and wind electricity production
        from 2016 to 2019.

    Parameters
    ----------
    path_dir : str, optional
        path to the directory containing the source files. Typically, this is the `support_files/`
        directory of the cloned git repository of EcoDynElec. The directory must contain ALL
        the files of interest, otherwise the execution is aborted.
        If None, an attempt to use a default path is made, with no promises.
    output_file : str, optional
        path to the file to save the updated data. If None, the file is saved in a 'enr_prod_2016-2022_completed.csv'
        file in the `path_dir` directory.
    verbose : bool, optional
        to display information. Default to False.

    Returns
    -------
    None
    """
    ### Verify the path_dir
    if path_dir is None:
        ### Try to reach a default directory
        path_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'support_files')
    if not os.path.isdir(path_dir):  # Verify if path is valid
        raise FileNotFoundError(
            f"Need to specify a directory containing updated files to save them into software files.")

    predicted_data = os.path.join(path_dir, 'enr_prod_2016-2019.csv')
    if not os.path.isfile(predicted_data):
        raise FileNotFoundError(
            f"The file {predicted_data} is missing in {path_dir}. Please generate it using Ecd-EnrModel project and save it in {path_dir}.")
    predicted_data = pd.read_csv(predicted_data, index_col=0, parse_dates=[0])
    mapped_data = get_enr_data_from_pronovo_ec(path_dir, verbose)
    # Merge the predicted data (ending in 2019) with the mapped data (starting in 2020) and ensure they don't overlap
    ndf = pd.concat([predicted_data.loc[:(mapped_data.index[0] - timedelta(hours=1))], mapped_data], axis=0)
    ndf.fillna(0, inplace=True)
    if output_file is None:
        output_file = os.path.join(path_dir, 'enr_prod_2016-2024_completed.csv')
    if verbose:
        print(f"Saving {output_file}...")
    ndf.to_csv(output_file)



def extract_uvek_impact(ctry, vector_path, is_verbose=False) -> DataFrame:
    """
    Extracts the impact data from the provided file and processes the data to filter and rename specified
    impact categories.

    The function reads the input file, identifies and processes specific categories of impact-related data,
    performs renaming, and adjusts the resultant dataset accordingly. It returns a cleaned matrix containing
    the processed data.

    Parameters
    ----------
    ctry : list, tuple, or str
        The country or list of countries for which the data extraction is performed.

    vector_path : str
        Path to the input CSV file containing the impact data.

    is_verbose : bool, optional
        If True, the function prints progress or debug messages. Default is False.

    Returns
    -------
    pandas.DataFrame
        The cleaned matrix containing filtered and renamed impact categories and their corresponding data.

    Raises
    ------
    TypeError
        If `ctry` is not of type `list`, `tuple`, or `str`.

    ValueError
        If the header "Catégorie d'impact" is not found in the input file.
    """

    ### Check the country list
    if is_verbose: print("Extraction of UVEK impact")
    # Test the type of country
    if type(ctry) != list:
        raise TypeError("Parameter ctry should be a list, tuple or str")

    df_temp = pd.read_excel(vector_path, header=None)
    header_idx = df_temp.index[df_temp.iloc[:, 0] == "Catégorie d'impact"].tolist()

    if not header_idx:
        raise ValueError("Header 'Catégorie d'impact' not found in file.")

    matrix = pd.read_excel(vector_path, header=header_idx[0])
    row_to_keep = ["Climate change - Fossil", "Land use", "Particulate matter", "Human toxicity, cancer"]
    matrix = matrix.rename(columns={"Catégorie d'impact": 'Category'})
    matrix = matrix[matrix['Category'].isin(row_to_keep)]

    if len(matrix.columns) > 1:
        matrix = matrix.drop(matrix.columns[1], axis=1)
    matrix = matrix.set_index('Category')

    row_mapping = {
        "Climate change - Fossil": "Carbon intensity",
        "Land use": "Land use",
        "Particulate matter": "Fine particulate matter formation",
        "Human toxicity, cancer": "Human carcinogenic toxicity"
    }
    matrix = matrix.rename(index=row_mapping).T
    re_order = ["Carbon intensity", "Human carcinogenic toxicity", "Fine particulate matter formation","Land use" ]
    matrix = matrix[re_order]
    matrix_clean = matrix[~matrix.index.duplicated(keep='first')]

    return matrix_clean

def impact_mapping_matching(uvek_path, ui_vector_path, mapping_path, is_verbose=False):
    """
    Matches technologies with their environmental impact values by mapping data from
    various sources (UVEK, UI impact vector, and mapping details). Generates a new
    dataframe aligning UI technologies to their corresponding impact values.

    Parameters
    ----------
    uvek_path : str
        File path to the UVEK impact data file.
    ui_vector_path : str
        File path to the UI impact vector data file.
    mapping_path : str
        File path to the mapping file linking EcoInvent and UVEK.
    is_verbose : bool, optional
        Flag to enable verbose output during the process, default is False.

    Returns
    -------
    pandas.DataFrame
        A new impact vector with combined environmental data for technologies from
        the UI impact vector and the mapping to UVEK impact data.
    """

    # Loading UVEK impact data
    ctry = ['CH','AT','DE','FR','CZ','IT']
    uvek_impact = extract_uvek_impact(ctry, vector_path=uvek_path, is_verbose=is_verbose)

    # Loading Mapping data for EcoInvent and UVEK
    mapping = pd.read_csv(mapping_path, sep=';', encoding='latin-1')
    mapping['Pourcentage'] = mapping['Pourcentage'].astype(str).str.replace(',', '.', regex=False)
    mapping['Pourcentage'] = pd.to_numeric(mapping['Pourcentage'], errors='coerce')

    # Loading UI impact vector
    ui_impact = pd.read_csv(ui_vector_path)

    new_impact_vector = pd.DataFrame(columns=uvek_impact.columns)
    for techno in ui_impact.iloc[:, 0]:
        techno_map = mapping[mapping['EcoDynElec'] == techno]
        if not techno_map.empty:
            new_impact_vector.loc[techno] = [0,0,0,0]
            for index, row in techno_map.iterrows():
                if pd.isna(row['UVEK']): pass
                else:
                    new_impact_vector.loc[techno] += row['Pourcentage']*uvek_impact.loc[row['UVEK']]
        else:
            if is_verbose:
                print(f"No mapping found for : {techno}")
    return new_impact_vector

