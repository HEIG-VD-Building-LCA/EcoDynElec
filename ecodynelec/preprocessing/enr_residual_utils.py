"""
Helper functions to load and preprocess the renewable electricity data from Pronovo and EnergyCharts.
"""

import os
import re
from datetime import datetime

import pandas as pd

pronovo_types_map = {
    '*': {
        'Wind': 'Wind (kWh)',
        'Solar': 'Photovoltaik (kWh)',
        'Biogas': 'Biogas (kWh)',
        'Biomass_1_crops': 'Energiepflanze (kWh)',
        'Biomass_2_waste': 'Forst- und Landwirtschaftliche Abfälle (kWh)',
        'Waste_1': 'Kehrichtverbrennung (erneuerbar) (kWh)',
        'Sewage_gas': 'Klärgas (kWh)',
    },
    '5': {
        'Wind': 'Wind',
        'Solar': 'Photovoltaic',
        'Biomass_all': 'Biomasse'
    },
    '2': {
        'Wind': '-A.Windenergie [kWh]',
        'Solar': '-A.Photovoltaik [kWh]',
        'Biogas': '-A.Biogas [kWh]',
        'Biomass_1_crops': '-A.Energiepflanze [kWh]',
        'Biomass_2_waste': '-A.Forst- und Landwirtschaftliche Abfälle [kWh]',
        'Waste_1': '-A.Kehrichtverbrennung [kWh]',
        'Waste_2.50': '-A.Kehrichtverbrennung (erneuerbar).50 [kWh]',
        'Waste_3.100': '-A.Kehrichtverbrennung (erneuerbar).100 [kWh]',
        'Waste_4_no_enr': '-A.Kehrichtverbrennung (nicht erneuerbar) [kWh]',
        'Sewage_gas': '-A.Klärgas [kWh]',
        # 'Gas_1': '-A.Erdgas Dampfturbine [kWh]',
        # 'Gas_2': '-A.Gas- und Dampfkombikraftwerk [kWh]',
        # 'Gas_3': '-A.Gasturbine [kWh]',
        # 'Unknown': '-A.Leichtwasserreaktor [kWh]', #light water  -> matches nuclear production
        # 'Combustion_engine': '-A.Verbrennungsmotor [kWh]'
        'Hydro_Pumpage': '+A.Pumpspeicherkraftwerk [kWh]',
        'Hydro_Pumped_Storage': '-A.Pumpspeicherkraftwerk [kWh]'
    },
    '7': {
        'Wind': '-A.Windturbine [kWh]',
        'Solar': '-A.Photovoltaik [kWh]',
        'Biogas': '-A.Biogas [kWh]',
        'Biomass_1_crops': '-A.Energiepflanze [kWh]',
        'Biomass_2_waste': '-A.Forst- und Landwirtschaftliche Abfälle [kWh]',
        'Waste_1': '-A.Kehrichtverbrennung [kWh]',
        'Waste_2.50': '-A.Kehrichtverbrennung (erneuerbar).50 [kWh]',
        'Waste_3.100': '-A.Kehrichtverbrennung (erneuerbar).100 [kWh]',
        'Waste_4_no_enr': '-A.Kehrichtverbrennung (nicht erneuerbar) [kWh]',
        'Sewage_gas': '-A.Klärgas [kWh]',
        # 'Gas_1': '-A.Erdgas Dampfturbine [kWh]',
        # 'Gas_2': '-A.Gas- und Dampfkombikraftwerk [kWh]',
        # 'Gas_3': '-A.Gasturbine [kWh]',
        # 'Unknown': '-A.Leichtwasserreaktor [kWh]', #light water  -> matches nuclear production
        # 'Combustion_engine': '-A.Verbrennungsmotor [kWh]'
        'Hydro_Pumpage': '+A.Pumpspeicherkraft [kWh]',
        'Hydro_Pumped_Storage': '-A.Pumpspeicherkraft [kWh]'
    }
}
"""
Mapping linking pronovo column names to the actual types of plants used in this project.
There is a different mapping for each pronovo file format (see load_pronovo_file).
"""

ec_types_to_types = {
    'Photovoltaïque': 'Solar',
    'Éolienne': 'Wind',
    'Biogaz': 'Biogas',
    'Biomasse': 'Biomass_all',
    'Cultures énergétiques': 'Biomass_1_crops',
    'Déchets forestiers et agricoles': 'Biomass_2_waste',
    'Incinération': 'Waste_1',
    "Biogaz de station d'épuration": 'Sewage_gas',
    "Gaz d'égout": 'Sewage_gas'
}
"""
Mapping linking energy charts column names to the actual types of plants used in this project.
"""

data_mappings = {
    'Solar': [
        {
            'start': '2020-01-01',
            'end': 'last',
            'source': 'Pronovo',
            'series': 'Solar'
        },
        {
            'start': '2020-01-01',
            'end': 'last',
            'from_start': '2020-01-01',
            'from_end': 'last',
            'source': 'EC',
            'series': 'Solar'
        }
    ],
    'Wind': [
        {
            'start': '2020-01-01',
            'end': 'last',
            'source': 'Pronovo',
            'series': 'Wind'
        }
    ],
    'Waste': [
        {
            'start': '2020-05-01',
            'end': '2022-09-30',
            'source': 'Pronovo',
            'series': 'Waste_1'
        },
        {
            'start': '2022-10-01',
            'end': '2022-11-30',
            'from_start': '2021-10-01',
            'from_end': '2021-11-30',
            'source': 'EC',
            'series': 'Waste_1'
        },
        {
            'start': '2023-01-01',
            'end': 'last',
            'source': 'Pronovo',
            'series': 'Waste_1'
        }
    ],
    'Biogas': [
        {
            'start': '2020-05-01',
            'end': 'last',
            'source': 'Pronovo',
            'series': 'Biogas'
        }
    ],
    'Sewage_gas': [
        {
            'start': '2020-05-01',
            'end': 'last',
            'source': 'Pronovo',
            'series': 'Sewage_gas'
        }
    ],
    'Biomass_1_crops': [
        {
            'start': '2020-05-01',
            'end': '2022-12-31',
            'source': 'Pronovo',
            'series': 'Biomass_1_crops'
        },
        {
            'start': '2023-01-01',
            'end': 'last',
            'source': 'Pronovo',
            'series': 'Biomass_1_crops'
        }
    ],
    'Biomass_2_waste': [
        {
            'start': '2020-05-01',
            'end': '2021-12-31',
            'source': 'Pronovo',
            'series': 'Biomass_2_waste'
        },
        {
            'start': '2022-01-01',
            'end': '2022-12-31',
            'from_start': '2021-01-01',
            'from_end': '2021-12-31',
            'source': 'Pronovo',
            'series': 'Biomass_2_waste'
        },
        {
            'start': '2023-01-01',
            'end': 'last',
            'source': 'Pronovo',
            'series': 'Biomass_2_waste'
        }
    ],
    'Hydro_Pumped_Storage': [
        {
            'start': '2023-01-01',
            'end': 'last',
            'source': 'Pronovo',
            'series': 'Hydro_Pumped_Storage'
        }
    ],
    'Hydro_Pumpage': [
        {
            'start': '2023-01-01',
            'end': 'last',
            'source': 'Pronovo',
            'series': 'Hydro_Pumpage'
        }
    ]
}
"""
**The reorganization rules are only valid for the 2020-2024 period.**
    
A dict giving the mapping between the columns of the final data (result) and the source data from Pronovo and EnergyCharts. 
Used by the reorganize_enr_data method. 

Should follow the structure :

.. code-block:: python

    'Category1': [
        {
            'start': 'start date', # start date in the result (and in source data if from_start isn't set)
            'end': 'end date', # end date in the result (and in source data if from_end isn't set) (use 'last' to get the real end of the source data)
            'from_start': '2021-01-01', # optional, start date in the source data (allows to copy data from one date to another)
            'from_end': '2021-12-31', # optional, end data in the source data
            'source': 'Pronovo' or 'EC', # source of the data
            'series': 'Series_Name_In_Source_Data' # name of the column to take in the source data
        },
        ...
    ],
    ...
"""


def get_enr_data_from_pronovo_ec(path_dir, verbose=False):
    """
    Reads the pronovo and energy charts data from the given directory, and returns a dataframe containing the
    reorganized data. The reorganized data is the best estimation of the real renewable electricity productions (solar, wind,
    waste...), from what is available.

    **The reorganization rules are only valid for the 2020-2022 period.**

    Parameters
    ----------
    path_dir : str
        The path of the directory containing the pronovo and energy charts data.

        The directory should contain two subdirectories:

        - pronovo_data: containing the pronovo data (a 'prod_year' directory for each input)

        - ec_data: containing the energy charts data (annual files)

        See the documentation below for more details.
    verbose : bool, optional
        Whether to print debug information. The default is False.

    Returns
    -------
    mapped_data : pd.DataFrame
        A dataframe containing the reorganized data, indexed by date.
    """
    pronovo_data = read_enr_data_from_pronovo(path_dir, verbose=verbose)
    ec_data = read_enr_data_from_energy_charts(path_dir, verbose=verbose)
    mapped_data = reorganize_enr_data(pronovo_data, ec_data)
    return mapped_data


def read_enr_data_from_pronovo(path_dir, verbose=False):
    """
    Reads all the pronovo data from the given directory, and returns a dataframe containing the data.

    Parameters
    ----------
    path_dir : str
        The path of the directory containing the pronovo data.
        The directory should contain a subdirectory for each year, named 'prod_year' (e.g. 'prod_2020').
        Each subdirectory should contain the pronovo data files of this year (.csv files).
        See the documentation below for more details.
    verbose : bool, optional
        Whether to print debug information. The default is False.

    Returns
    -------
    pronovo_data : pd.DataFrame
        A dataframe containing the pronovo data, indexed by date.
    """
    pronovo_dir = os.path.join(path_dir, 'pronovo_data')
    if not os.path.isdir(pronovo_dir):
        raise FileNotFoundError(
            f"Directory {pronovo_dir} doesn't exist. Please create it and add actual pronovo data directories (follow the procedure explain in the documentation).")
    # Read pronovo data
    years = []
    for file in os.listdir(pronovo_dir):
        if file.startswith('prod_') and os.path.isdir(os.path.join(pronovo_dir, file)):
            years.append(file)
    if verbose:
        print(f'Reading pronovo directories: {years}')
    types = list(pronovo_types_map['2'].keys())
    types.append('Biomass_all')
    pronovo_data = load_all_pronovo_files(root_dir=pronovo_dir + '/', types=types,verbose=verbose)
    return pronovo_data


def read_enr_data_from_energy_charts(path_dir, verbose=False):
    """
    Reads all the energy charts data from the given directory, and returns a dataframe containing the data.

    Parameters
    ----------
    path_dir : str
        The path of the directory containing the energy charts data.
        The directory should contain a subdirectory named 'ec_data'.
        The 'ec_data' directory should contain the yearly energy charts data files (.csv files).
        See the documentation below for more details.
    verbose : bool, optional
        Whether to print debug information. The default is False.

    Returns
    -------
    df_ec_data : pd.DataFrame
        A dataframe containing the energy charts data, indexed by date.
    """
    ec_dir = os.path.join(path_dir, 'ec_data')
    if not os.path.isdir(ec_dir):
        raise FileNotFoundError(
            f"Directory {ec_dir} doesn't exist. Please create it and add actual energy charts data files (follow the procedure explain in the documentation).")
    # Read EnergyCharts data
    ec_data = []
    for f in os.listdir(ec_dir):
        if f.endswith('.csv'):
            if verbose: print('Reading ' + f)
            data = pd.read_csv(ec_dir + '/' + f, index_col=0)
            data = data.drop(index=data.index[0], columns=[col for col in data.columns if col.startswith('Unnamed')])
            data = data.rename(columns=ec_types_to_types)
            ec_data.append(data)
    df_ec_data = pd.concat(ec_data, axis=0).fillna(0).astype(float)
    # create DatetimeIndex from D.M.Y format to match pronovo data
    df_ec_data.index = pd.to_datetime(df_ec_data.index, format='%d.%m.%Y')
    return df_ec_data


def reorganize_enr_data(pronovo_data: pd.DataFrame, ec_data: pd.DataFrame) -> pd.DataFrame:
    """
    | Reorganizes the pronovo and energy charts data to match the final data format.
    | The reorganized data is the best estimation of the real renewable electricity productions, from what is available.
    | **The reorganization rules are only valid for the 2020-2024 period.**

    Parameters
    ----------
    pronovo_data : pd.DataFrame
        The pronovo data, indexed by date.
    ec_data : pd.DataFrame
        The energy charts data, indexed by date.

    Returns
    -------
    mapped_data : pd.DataFrame
        A dataframe containing the reorganized data, indexed by date.
    """

    def _dedupe_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        | Deduplicates the index of the dataframe.
        :param df: DataFrame
        :return: A dataframe with deduplicated index
        """
        df = df.sort_index()
        if not df.index.is_unique:
            # aggregates duplicated timestamps (e.g., DST fall-back hour) by sum
            df = df.groupby(level=0).sum()
        return df

    pronovo_data = _dedupe_index(pronovo_data)
    ec_data = _dedupe_index(ec_data)

    mapped_data = pd.DataFrame(index=pronovo_data.index, columns=data_mappings.keys())
    real_end = ec_data.index[-1]
    if pronovo_data.index[-1] < real_end:
        real_end = pronovo_data.index[-1]
    real_end = str(real_end.date())
    for col in data_mappings.keys():
        maps = data_mappings[col]
        for mapping in maps:
            from_ec = mapping['source'] == 'EC'
            src_df = ec_data.copy() if from_ec else pronovo_data
            mapping_end_str = real_end if mapping.get('end') == 'last' else mapping['end']
            start = datetime.strptime(mapping['start'], '%Y-%m-%d')
            end = datetime.strptime(mapping_end_str, '%Y-%m-%d') + pd.Timedelta(hours=23)
            if ('from_start' in mapping) or ('from_end' in mapping):
                from_start_str = mapping.get('from_start', mapping['start'])
                from_end_str = mapping.get('from_end', mapping_end_str)
                if from_end_str in ('last', 'end'):
                    from_end_str = mapping_end_str
                from_start = datetime.strptime(from_start_str, '%Y-%m-%d')
                from_end = datetime.strptime(from_end_str, '%Y-%m-%d') + pd.Timedelta(hours=23)
                if from_ec:
                    # Scale past pronovo hours to actual energy charts daily production
                    prod_ec = src_df.loc[start:end, mapping['series']]
                    prod_pronovo = pronovo_data.loc[from_start:from_end, mapping['series']].copy()
                    prod_pronovo.index = mapped_data.loc[start:end, col].index
                    daily_y = prod_pronovo.resample('D').sum()
                    daily_y.index = prod_ec.index
                    prod_ec = prod_ec * 1e6  # Convert to kWh
                    factors = prod_ec / daily_y
                    # Adjust the index to include the hours of the last day
                    adjusted_dates = pd.date_range(start=factors.index[0],
                                                   end=factors.index[-1] + pd.Timedelta(hours=23),
                                                   freq='H')
                    resampled_dates = factors.reindex(adjusted_dates, method='ffill')
                    prod_pronovo = prod_pronovo.multiply(resampled_dates)
                else:
                    # Directly copy from one date to another date, from Pronovo data
                    prod_pronovo = src_df.loc[from_start:from_end, mapping['series']].copy()
                    prod_pronovo.index = mapped_data.loc[start:end, col].index
                mapped_data.loc[start:end, col] = prod_pronovo
            else:
                if mapping['source'] == 'EC':
                    # Convert daily EnergyCharts data to hourly data
                    src_df = src_df * 1000000  # Convert to kWh
                    src_df = src_df.resample(
                        'H').ffill() / 24  # Convert to hourly data (with a uniform repartition over the day in first approximation)
                    print('Warning: uniform daily distribution of EC data was used for column', col)
                # simple copy
                mapped_data.loc[start:end, col] = src_df.loc[start:end, mapping['series']]
    return mapped_data


def load_all_pronovo_files(root_dir: str, types: [str], verbose: bool = False) -> pd.DataFrame:
    """
    Loads all pronovo files in the given directories, applying daily scaling with energy charts ecd_enr_model (the hourly variation
    comes from the pronovo ecd_enr_model, and the daily total from energy charts ecd_enr_model, if available).
    The scaling is done with csv files starting by "EC". All other csv files are considered as pronovo files.

    :param root_dir: The root directory containing the pronovo 'prod_year' directories
    :param types:  The types of plants to extract (in ['Wind', 'Solar'])
    :param verbose:  Whether to print debug information
    :return:  A dataframe containing the pronovo ecd_enr_model for all 'types', indexed by date
    """
    Ys = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(subdir, file)
                try:
                    f_y = load_pronovo_file(full_path, types, verbose=verbose)
                    Ys.append(f_y)
                except Exception as e:
                    if verbose:
                        print(f"[WARN] {full_path}: {e}")
    if not Ys:
        raise RuntimeError(f"Aucun CSV exploitable trouvé sous {root_dir}")

    pronovo_data = pd.concat(Ys).sort_index()
    return pronovo_data


def _parse_mixed_pronovo_index(idx_like: pd.Series) -> pd.DatetimeIndex:
    """
    Parse the index which can contains:
        - datetimes 'dd.mm.yyyy HH:MM'
        - Excel numbers (44256.35 or 44'256.35)

    :param idx_like: A pandas Series or Index of strings/numbers representing timestamps in mixed formats.
    :return: A  pandas DatetimeIndex with all valid timestamps parsed and rounded to the nearest 15 minutes.
    """

    _DATE_STRICT_FMT = '%d.%m.%Y %H:%M'
    _DATE_RE = re.compile(r"^\s*\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}\s*$")
    _EXCEL_RE = re.compile(r"^\s*\d{2}'\d{3}(?:\.\d+)?\s*$|^\s*\d{5}(?:\.\d+)?\s*$")

    s = idx_like.astype(str).str.strip()

    mask_txt = s.str.match(_DATE_RE)
    dt = pd.to_datetime(s.where(mask_txt),
                        format=_DATE_STRICT_FMT,
                        errors='coerce',
                        dayfirst=True)

    mask_xl = s.str.match(_EXCEL_RE)
    if mask_xl.any():
        xl = (s.where(mask_xl)
              .str.replace("'", "", regex=False)  # thousands Switzerland
              .str.replace(",", ".", regex=False))
        xl_num = pd.to_numeric(xl, errors='coerce')
        dt_xl = pd.to_datetime(xl_num, unit='d', origin='1899-12-30', errors='coerce')
        dt_xl = dt_xl.dt.round('15min')
        dt = dt.fillna(dt_xl)

    if dt.isna().any():
        dt_fallback = pd.to_datetime(s, errors='coerce', dayfirst=True)
        dt = dt.fillna(dt_fallback)

    return pd.DatetimeIndex(dt)


def load_pronovo_file(file: str, types: [str], verbose: bool = False) -> pd.DataFrame:
    """
    Load pronovo ecd_enr_model from a csv file.
    Supports years from 2020 to 2022 (historically the format of the files changes every semester).

    :param file: the path of the file to load
    :param types:  The types of plants to extract (in ['Wind', 'Solar'])
    :param verbose:  Whether to print debug information
    :return:  A dataframe containing the pronovo ecd_enr_model for all 'types', indexed by date
    """

    if file.endswith('2020.csv'):
        format = 5 if int(file[-11:-9]) < 5 else 6
    elif file.endswith('2021.csv'):
        format = 3 if int(file[-11:-9]) < 8 else 4
    elif file.endswith('2022.csv'):
        format = 1
    elif file.endswith('202303_CH_Total_Quartal_def.csv'):
        format = 7
    else:
        format = 2
    if verbose: print(f'Load fmt {format} {file}', end='\n')
    if format == 5:
        pronovo_data = pd.read_csv(f'{file}', index_col=0, skiprows=2,
                                   encoding='windows-1252', sep=';')
        pronovo_types = pronovo_types_map['5']
    elif format == 6:
        pronovo_data = pd.read_csv(f'{file}', index_col=0, skiprows=10,
                                   encoding='windows-1252', sep=';')
        pronovo_types = pronovo_types_map['*']
    elif format == 3:
        pronovo_data = pd.read_csv(f'{file}', index_col=0, skiprows=16,
                                   encoding='windows-1252', sep=';')
        pronovo_types = pronovo_types_map['*']
    elif format == 4:
        pronovo_data = pd.read_csv(f'{file}', index_col=0, skiprows=18,
                                   encoding='windows-1252', sep=';')
        pronovo_types = pronovo_types_map['*']
    elif format == 1:
        pronovo_data = pd.read_csv(f'{file}', index_col=0, skiprows=17,
                                   encoding='windows-1252', sep=';')
        pronovo_types = pronovo_types_map['*']
    elif format == 2:
        pronovo_data = pd.read_csv(f'{file}', index_col=1, skiprows=1,
                                   encoding='windows-1252', sep=';')
        pronovo_types = pronovo_types_map[str(format)]
    elif format == 7:
        pronovo_data = pd.read_csv(f'{file}', index_col=1, skiprows=1,
                                   encoding='windows-1252', sep=';')
        pronovo_types = pronovo_types_map[str(format)]
    else:
        raise Exception('Unknown format')
    dt_idx = _parse_mixed_pronovo_index(pronovo_data.index.to_series())
    valid = dt_idx.notna()
    if not valid.any():
        raise ValueError(f"No valid date detected in {file}")
    pronovo_data = pronovo_data.loc[valid].copy()
    pronovo_data.index = dt_idx[valid]
    pronovo_types_a = [pronovo_types[tpe] for tpe in types if
                       tpe in pronovo_types and pronovo_types[tpe] in pronovo_data.columns]
    pronovo_data = pronovo_data[pronovo_types_a]
    pronovo_types_inv = {v: k for k, v in pronovo_types.items()}
    for i in range(len(types)):
        pronovo_data.rename(columns=pronovo_types_inv, inplace=True)
    pronovo_data = pronovo_data.applymap(
        lambda x: float(x) if type(x) != str else float(x.replace('\'', '').replace('’', '')))
    pronovo_data = pronovo_data.resample('H').sum()
    pronovo_data = pronovo_data.iloc[:-1]  # last value is first hour if the next month
    return pronovo_data
