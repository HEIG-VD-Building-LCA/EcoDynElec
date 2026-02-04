"""
This module defines functions to compute the dynamic environmental impact of Hydro Pumped
Storage energy, as well as the dynamic shares of storage inflows, outflows, and losses.

The `dynamic_impact` function calculates production and mix-related impact at an hourly
granularity, adjusting for the dynamic usage of storage systems across multiple countries.
The function also updates the impact for global and individual country levels.

The `dynamic_storage_shares` function determines the dynamics of storage utilization
(e.g., natural and pumped inflows, production, losses), based on storage data and energy
flow information.

"""
import pandas as pd
import os

def dynamic_impact(prod_imp : dict, mix_imp : dict, flows : pd.DataFrame, prod_mix : dict, mix_dict : dict, impact_matrix : pd.DataFrame, network_impact : dict,  step_imp_memory : dict, parameter, is_verbose=False) -> (dict,dict):
    """
    Computes the dynamic environmental impacts of storage for given production and mix data
    within a time series. Adjusts the impact values dynamically based on storage factors
    such as natural inflow and pumpage inflow percentages, and applies this computation
    across specified countries and a global scope.

    Parameters
    ----------
    prod_imp : dict
        A dictionary containing production impact data with hierarchical structures
        per country and by energy technology types.
    mix_imp : dict
        A dictionary containing environmental mix impact data with hierarchical structures
        per country and by energy technology types.
    flows : pd.DataFrame
        A DataFrame containing raw production and consumption data for each technologies in country.
    prod_mix : dict
        A dictionary that provides data regarding production mixes for energy technologies
        per country.
    mix_dict : dict
        A dictionary containing mix-related data used for adjusting mix impacts per country.
    impact_matrix : pd.DataFrame
        A DataFrame representing the initial impact values per technology type for
        dynamic computation adjustments.
    step_imp_memory : dict
        Dictionnary containing the last STEP impact of the last year
    parameter : class
        Class encapsulating various settings, including parameters for
        interpolated storage, start and end times, and storage path configuration.
    is_verbose : bool, optional
        A flag indicating whether detailed information should be logged during the
        computation process. Default is False.

    Returns
    -------
    prod_imp : dict
        Updated dictionary for production impacts after applying the dynamic storage
        impact computation.
    mix_imp : dict
        Updated dictionary for mix environmental impacts after applying the dynamic
        storage impact computation.
    step_imp : dict
        Dictionnary containing the last STEP impact of the last year
    """
    if parameter.interpolated_stock:
        # Load storage data from Energy Charts
        if parameter.storage_path is None:
            raise ValueError("Storage path is not set. Please set the storage directory path in the config file.")
        storage = load_concat_files(parameter.storage_path, is_verbose=is_verbose)
        storage = interpolate_storage_hourly(storage)
        storage = storage.loc[parameter.start:parameter.end]
        storage_mode = 'interpolated'
    else:
        # No storage data interpolated -> no storage data
        storage = None
        storage_mode = 'flow-based'

    # Determine storage shares : Natural inflow % / Pumpage inflow % in the stock
    storage_shares = dynamic_storage_shares(storage, flows, storage_mode, is_verbose=is_verbose)

    # Initial conditions
    if not step_imp_memory:
        step_imp = impact_matrix.loc['Hydro_Pumped_Storage_CH'].to_dict()
        step_imp = {k: [v] for k, v in step_imp.items()}
    else:
        step_imp = step_imp_memory[parameter.start.year-1] # to get 2023 data if we are in 2024 for example
        step_imp = {k: [v] for k, v in step_imp.items()}

    # Determine the dynamic impact of STEP and apply it to all the countries
    for k in prod_imp['CH'].keys():
        if k == 'Global':
            continue
        else:
            # Remove the Hydro_Pumped_Storage_CH from columns to avoid counting it twice.
            mix = mix_imp['CH'][k].drop(columns=['Hydro_Pumped_Storage_CH'], inplace=False)

            # Hourly Dynamic impact computation
            for i in range(1, len(mix.index)):
                t = mix.index[i]
                if storage_mode == "interpolated":
                    # Pumping factor
                    fp = storage_shares['turbine consumption'].iloc[i]/storage_shares['SP'].iloc[i]
                    # Dynamic impact computing
                    step_imp[k].append(fp*mix.loc[t].sum() + (1-fp)*step_imp[k][i-1])
                elif storage_mode == "flow-based":
                    # Dynamic impact computing without storage data
                    step_imp[k].append(mix.loc[t].sum())
            # Apply dynamic impact to all countries
            for country in prod_imp.keys():
                # Add STEP impact to 'Carbon intensity', 'Human carcinogeric toxicity' ...
                if country == 'CH':
                    prod_imp[country][k]['Hydro_Pumped_Storage_CH'] = prod_mix[country]['Hydro_Pumped_Storage_CH']*step_imp[k] + prod_mix[country]['Hydro_Pumped_Storage_CH']*network_impact['CH']['Infra PHS'][k] # prod_imp is a dictionary concerning only the production technology of the country in question
                mix_imp[country][k]['Hydro_Pumped_Storage_CH'] = mix_dict[country]['Hydro_Pumped_Storage_CH']*step_imp[k] + mix_dict[country]['Hydro_Pumped_Storage_CH']*network_impact['CH']['Infra PHS'][k]

                # Recalculate 'Global' impact with STEP impact added
                prod_imp[country]['Global'][k] = prod_imp[country][k].sum(axis=1)
                mix_imp[country]['Global'][k] = mix_imp[country][k].sum(axis=1)

            step_imp[k] = step_imp[k][-1] # keep in memory the last impact of each category

    return prod_imp, mix_imp, step_imp


def dynamic_storage_shares(storage : pd.DataFrame, flows : pd.DataFrame, storage_mode : str, is_verbose : bool ) -> pd.DataFrame:
    """
    Calculate dynamic storage shares based on provided storage data, flows, and production mix.

    This function computes the dynamic shares of storage considering turbine production, turbine
    consumption, natural pumping, and overflow mechanisms depending on the specified storage
    mode. It also tracks the losses and categorizes storage into natural and pumped categories
    over each time step. The computation begins with initial conditions and iteratively updates
    the shares for each subsequent time step.

    Parameters
    ----------
    storage : pd.DataFrame
        DataFrame containing information about storage levels and maxima. If None, the index
        will be derived from the production mix.

    flows : pd.DataFrame
        A DataFrame containing raw production and consumption data for each technologies in country.

    storage_mode : str
        Mode of storage calculation. It supports two values:
        - 'interpolated': Storage levels are calculated with respect to maximum storage values.
        - 'flow-based': Assumes no overflow, with constant storage levels.

    is_verbose : bool
        Flag to enable or disable detailed verbosity during the computation process.

    Returns
    -------
    pd.DataFrame
        DataFrame containing time-series of computed storage shares. Columns include:
        - 'sj': Current storage level
        - 'sjj': Previous storage level
        - 'turbine production': Turbine outflow (production from storage)
        - 'turbine consumption': Turbine inflow (consumption into storage)
        - 'overflow': Overflow due to exceeding maxima
        - 'natural pumping': Residual balance of natural pumping
        - 'losses': Total losses including turbine production and overflow
        - 'SN': Share of natural inflows
        - 'SP': Share of pumping inflows
    """
    # Define the index to use for storage shares
    if storage is None:
        index_to_use = flows.index
    else:
        index_to_use = storage.index

    storage_shares = pd.DataFrame(index=index_to_use)

    # --- Storage level : adapted method for storage_mode ---
    if storage_mode == 'interpolated':
        canton_cols = [c for c in storage.columns if c.lower() != 'storage max']
        sj_raw = storage[canton_cols].sum(axis=1)
        storage_max_gwh = storage['storage max']
        overflow_mech = (sj_raw - storage_max_gwh).clip(lower=0)
        storage_shares['sj'] = sj_raw.clip(upper=storage_max_gwh)  # determine storage level at j
        storage_shares['sjj'] = storage_shares['sj'].shift(1)  # determine storage level at j-1

    elif storage_mode == 'flow-based':
        overflow_mech = pd.Series(0.0, index=index_to_use) # hypothesis : no overflow, infinite storage
        constant_stock_series = pd.Series(1.0, index=index_to_use) # hypothesis : constant stock
        storage_shares['sj'] = constant_stock_series
        storage_shares['sjj'] = constant_stock_series

    # --- Turbine flows ---
    flow = flows.copy()
    inflow = abs(flow['Hydro_Pumpage_CH']) # STEP consumption, abs to have positive values
    outflow = flow['Hydro_Pumped_Storage_CH'] # STEP production

    storage_shares['turbine production'] = outflow
    storage_shares['turbine consumption'] = inflow

    # --- Natural pumping (residual balance term) ---
    natural_pumping = storage_shares['sj'] - storage_shares['sjj'] + storage_shares['turbine production'] - storage_shares['turbine consumption']
    storage_shares['overflow'] = overflow_mech + natural_pumping.clip(upper=0).abs()
    storage_shares['natural pumping'] = natural_pumping.clip(lower=0)

    # --- Retirements ---
    storage_shares['losses'] = storage_shares['turbine production'] + storage_shares['overflow']

    # --- Initial conditions ---
    share_pump = 0.24
    SN = [(1 - share_pump) * storage_shares['sj'].iloc[0]]  # Share of natural inflows at j = 0
    SP = [share_pump * storage_shares['sj'].iloc[0]]  # Share of pumping inflows at j = 0

    # --- Share of storage without losses ---
    for k in range(1, len(storage_shares)):
        S_tilde = SN[k - 1] + SP[k - 1] + storage_shares['natural pumping'].iloc[k] + storage_shares['turbine consumption'].iloc[k]
        S_N_tilde = SN[k - 1] + storage_shares['natural pumping'].iloc[k]
        S_P_tilde = SP[k - 1] + storage_shares['turbine consumption'].iloc[k]
        R = storage_shares['losses'].iloc[k]

        # --- Natural share ---
        if S_tilde <= 0:
            fN = 0
        else:
            fN = S_N_tilde / S_tilde
        SN.append(S_N_tilde - fN * R)
        SP.append(S_P_tilde - (1 - fN) * R)

    storage_shares['SN'] = SN
    storage_shares['SP'] = SP

    return storage_shares


def interpolate_storage_hourly(df):
    """
    Interpolates storage data to obtain hourly values.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex and numeric columns

    Returns
    -------
    DataFrame with hourly frequency and interpolated values
    """

    if len(df) == 0:
        return df

    # Resample to daily frequency
    df_hourly = df.resample('h').asfreq()

    # Interpolate missing values
    df_hourly = df_hourly.interpolate(method='time')

    return df_hourly

def load_concat_files(folder_path, is_verbose=False):
    """
    Concatenates all CSV files found in the specified folder.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing the yearly CSV files.
    is_verbose : bool, optional
        If True, prints each file being read and raises errors when files are missing.
        Default is False.

    Returns
    -------
    df : DataFrame with DatetimeIndex
    """

    # Configuration for each dataset type
    dataset_config = {
        'storage': {
            'prefix': 'Storage',
            'skiprows': [1],
            'date_col': 'Date (TC+1)',
            'rename_cols': ['Valais', 'Grisons', 'Tessin', 'Reste de la Suisse', 'storage max'],
            'convert': 1e6
        }
    }

    results = {}

    for dataset_name, config in dataset_config.items():
        # Get list of files matching the prefix
        # Determine the path to search in
        search_path = os.path.join(folder_path, config['folder']) if 'folder' in config else folder_path
        files = sorted([f for f in os.listdir(search_path)
                    if f.startswith(config['prefix'])])

        # AJOUTEZ CE BLOC POUR LE DÉBUG
        if not files:
            raise FileNotFoundError(f"No files found starting with '{config['prefix']}' in directory: {search_path}")
        # FIN DU BLOC DE DÉBUG

        if is_verbose:
            print(f"Loading {dataset_name}: {len(files)} files found")

        # Load and concatenate files
        dfs = []
        for f in files:
            if is_verbose:
                print(f"  Reading: {f}")
            skiprows = config.get('skiprows', None)
            df = pd.read_csv(os.path.join(search_path, f), skiprows=skiprows)
            dfs.append(df)

        # Concatenate all dataframes
        if len(dfs) > 1:
            df_concat = pd.concat(dfs, ignore_index=True)
        else:
            df_concat = dfs[0]

        # Process datetime index
        date_col = config['date_col']
        df_concat[date_col] = pd.to_datetime(df_concat[date_col], utc=True).dt.tz_localize(None)
        df_concat = df_concat.set_index(date_col).sort_index()

        # Rename columns if specified
        if 'rename_cols' in config:
            df_concat.columns = config['rename_cols']

        # Apply conversion if specified
        if 'convert' in config:
            df_concat = df_concat * config['convert']

        if is_verbose:
            print(f"  Loaded {len(df_concat)} rows")

    return df_concat