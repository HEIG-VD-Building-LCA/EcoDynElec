"""This module contains the functions to run the `pipeline` module

This module contains:
    - load_config: loads the configuration parameters.
    - check_download: checks if the entso-e data needs to be downloaded.
    - load_raw_prod_exchanges: loads the raw production and exchanges data.
    - load_impact_matrix: loads the impact matrix.
    - check_mapping: checks the mapping between production and impacts.
    - get_mix: computes the electric mix of the target countries.
    - get_impacts: computes the impacts of these mixs.
    - get_productions : computes production values in kWh for each source.
    - translate_to_timezone: translates the data into the timezone.
    - save_results: saves the results.
    - localize_from_utc: shifts the time-zone from results.
"""
import os.path

import numpy as np
import pandas as pd

from ecodynelec import saving
from ecodynelec.impacts import compute_impacts
from ecodynelec.parameter import Parameter
from ecodynelec.preprocessing import auxiliary as aux
from ecodynelec.preprocessing.auxiliary import load_ch_enr_model
from ecodynelec.preprocessing.downloading import download
from ecodynelec.preprocessing.load_impacts import extract_mapping, extract_UI
from ecodynelec.preprocessing.loading import import_data
from ecodynelec.progress_info import ProgressInfo
from ecodynelec.tracking import track_mix


def load_config(config) -> Parameter:
    """Loads the configuration parameters.

    Parameters
    ----------
        config: ecodynelec.Parameter or str
            a set of configuration parameters to govern the computation,
            either as Parameter object or str pointing at an xlsx file.

    Returns
    -------
    ecodynelec.Parameter
        The loaded Parameter object.
    """
    if isinstance(config, Parameter):  # If a parameter object
        p = config
    elif isinstance(config, str):
        if any([config.endswith(k) for k in ('.xlsx', '.xls', '.ods')]):
            p = Parameter(excel=config)
        else:
            raise NotImplementedError(f"File extension for {config} is not supported.")
    else:
        raise ValueError('Missing a configuration to pass parameters.')
    # Convert target to list is not already
    if isinstance(p.target, str):
        p.target = [p.target]
    return p


def check_download(parameters: Parameter, is_verbose: bool = False) -> None:
    """Checks if the config requires to download the entso-e data, and if so, downloads it.

    Parameters
    ----------
        parameters: ecodynelec.Parameter
            Parameters returned by :py:func:`load_config`.
        is_verbose: bool, default to False
            To display information
    """
    if parameters.server.useServer:
        if None in [parameters.path.generation, parameters.path.exchanges]:  # If one path was not given
            raise KeyError("Can not download files: missing path raw_generation and/or raw_exchange to save files.")
        if is_verbose: print("Download Entso-E data from server...")
        download(config=parameters, is_verbose=is_verbose)  # Save files in a local dirrectory


def load_raw_prod_exchanges(parameters: Parameter | str, is_verbose: bool = False,
                            progress_bar: ProgressInfo = None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Loads the raw production and exchange data from entso-e.

    Parameters
    ----------
        parameters: ecodynelec.Parameter or str
            a set of configuration parameters to govern the computation,
            either as Parameter object or str pointing at an xlsx file.
        is_verbose: bool, default to False
            To display progress information
        progress_bar: ecodynelec.ProgressInfo, default to None
            The progress bar to use, or None

    Returns
    -------
        praw_prodExch: DataFrame with all productions and all exchanges from all included countries.
    """
    p = load_config(parameters)
    check_download(p, is_verbose=is_verbose)

    if progress_bar:
        progress_bar.set_sub_label('Load auxiliary datasets...')
    # Load SwissGrid -> if Residual or SG exchanges
    if np.logical_or(p.residual_global, p.sg_imports):
        if is_verbose: print('Load SwissGrid data...')
        # Load SwissGrid data, adjusting the date parameters for the time zone difference
        # EcoDynElec is using UTC while SwissGrid is using Europe/Zurich (UTC+1 in winter, UTC+2 in summer)
        # The SwissGrid data is already converted to Etc/GMT+1 in by the updating script
        # We use the most little time step possible to avoid issues with the time zones
        range = pd.date_range(start=p.start, end=p.end, freq="15min")
        range = range.tz_localize('Etc/GMT+1').tz_convert('UTC').tz_localize(
            None)  # Translate to Europe/Zurich timezone then remove tz info (for comparison with data source files)
        sg_data = aux.load_swissGrid(path_sg=p.path.swissGrid, start=str(range[0]), end=str(range[-1]), freq='15min')
        # Shift the SwissGrid data index back by one hour to convert from Etc/GMT+1 to UTC
        sg_data.index = pd.date_range(start=p.start, end=p.end, freq="15min")
        # Resample the SwissGrid data to the desired frequency, after the time zone conversion
        sg_data = sg_data.resample(p.freq).sum()
    else:
        sg_data = None
    if progress_bar:
        progress_bar.progress()
        progress_bar.set_sub_label('Load neighbors...')

    # Load Country of interest -> Always
    neighbours = aux.load_useful_countries(path_neighbour=p.path.neighbours, ctry=p.ctry)

    # Load enr production from EcoDynElec-Enr-Model, if enabled
    if progress_bar:
        progress_bar.progress()
        progress_bar.set_sub_label('Load Swiss enr production model...')
    if p.ch_enr_model_path is not None and 'CH' in p.ctry:
        if is_verbose: print('Loading Swiss enr production model')
        enr_prod_ch = load_ch_enr_model(p.ch_enr_model_path, p.start, p.end, p.freq)
    else:
        enr_prod_ch = None

    if progress_bar:
        progress_bar.progress()
        progress_bar.set_sub_label('Load ENTSO-E data...')

    # Load generation and exchange data from entso-e
    raw_prod_exch = import_data(ctry=p.ctry, start=p.start, end=p.end, freq=p.freq, involved_countries=neighbours,
                                path_gap=p.path.gap, sg_data=sg_data, enr_prod_ch=enr_prod_ch,
                                path_gen=p.path.generation, path_imp=p.path.exchanges,
                                savedir=p.path.savedir, net_exchange=p.net_exchanges,
                                residual_global=p.residual_global, correct_imp=p.sg_imports,
                                clean_data=p.data_cleaning, is_verbose=is_verbose, progress_bar=progress_bar)
    return raw_prod_exch


def load_impact_matrix(parameters: Parameter, is_verbose: bool = False) -> pd.DataFrame:
    """Loads the impact matrix from the specified parameters.

    Parameters
    ----------
        parameters: ecodynelec.Parameter
            Parameters returned by load_config.
        is_verbose: bool, default to False
            To display progress information

    Returns
    -------
        impact_matrix: pandas.DataFrame
            The unit impact vector of electricity production
    """

    # Load impact matrix (UI vector by default)
    if parameters.path.mapping is not None:  # Priority to the mapping spreadhseet, as soon as it is specified
        impact_matrix = extract_mapping(ctry=parameters.ctry, mapping_path=parameters.path.mapping,
                                        cst_import=parameters.cst_imports,
                                        residual=parameters.residual_global,
                                        target=parameters.target, is_verbose=is_verbose)
    else:  # If no mapping specified, go for the UI vector: it can grab the default vector automatically
        impact_matrix = extract_UI(path_ui=parameters.path.ui_vector, ctry=parameters.ctry, target=parameters.target,
                                   cst_imports=parameters.cst_imports,
                                   residual=parameters.residual_global)
    return impact_matrix


def get_mix(parameters: Parameter, raw_prod_exch: pd.DataFrame, return_matrix: bool = False,
            return_prod_mix: bool = False,
            is_verbose: bool = False,
            progress_bar: ProgressInfo = None):
    """Loads the mix from the specified parameters.
    If return_matrix is True, returns the whole consumption mix matrix instead of the mix dictionary.
    Else, returns the consumption mix dictionary (the impact matrix sorted by target country).
    If return_prod_mix is True, the production mix in the studied countries is also returned (as a tuple with the consumption mix).

    Parameters
    ----------
        parameters: ecodynelec.Parameter
            Parameters returned by load_config.
        raw_prod_exch: pandas.DataFrame
            The raw production and exchange data, as returned by load_raw_prod_exchanges.
        return_matrix: bool, default to False
            If True, returns the whole mix matrix instead of the mix dictionary (the impact matrix sorted by target country).
            (overrides return_prod_mix).
        return_prod_mix: bool, default to False
            If True, returns the production mix in the studied countries, as a tuple with the consumption mix.
            Has no effect if return_matrix is True.
        is_verbose: bool, default to False
            To display progress information
        progress_bar: ecodynelec.ProgressInfo, default to None
            To display progress information

    Returns
    -------
        return_matrix is False and return_prod_mix is False:
            A dictionary of pandas.DataFrame, with the consumption mix for each target (parameter.target)
            (containing the contribution of each production mean in 1 kWh consumed in the target country).
            This includes the local residual if parameters.residual_local is True.
        return_matrix is False and return_prod_mix is True:
            A tuple (prod_mix_dict, cons_mix_dict) of two dictionaries of pandas.DataFrame, with the consumption mix
            (containing the contribution of each production mean in 1 kWh consumed in the target country) and
            the production mix (containing the contribution of each production mean in 1 kWh produced in the studied countries)
            for each target (parameter.target).
        return_matrix is True:
            The raw table with the electricity mix in the studied countries
            (parameter.ctry + 'Other'), containing each production mean of each country.
            This is a list of pandas.DataFrame, for each time step.

    """
    # Load network losses -> if Network Loss asked
    if parameters.network_losses:
        network_losses = aux.load_grid_losses(network_loss_path=parameters.path.networkLosses, start=parameters.start,
                                              end=parameters.end)
    else:
        network_losses = None

    # Load production and consumption mixes
    if return_prod_mix:
        mix_df, prod_mix = track_mix(raw_data=raw_prod_exch, freq=parameters.freq, network_losses=network_losses,
                                     residual_global=parameters.residual_global, return_prod_mix=return_prod_mix,
                                     is_verbose=is_verbose, progress_bar=progress_bar)
    else:
        prod_mix = None
        mix_df = track_mix(raw_data=raw_prod_exch, freq=parameters.freq, network_losses=network_losses,
                           residual_global=parameters.residual_global, return_prod_mix=return_prod_mix,
                           is_verbose=is_verbose, progress_bar=progress_bar)
    if return_matrix:
        # old behavior for 'get_inverted_matrix' pipeline
        time_steps = []
        for step in mix_df.index.levels[0]:
            time_steps.append(mix_df.loc[step, :])
        return time_steps
    prod_mix_dict = {}
    cons_mix_dict = {}
    for target in parameters.target:
        cons_mix_dict[f'{target}'] = mix_df[f'Mix_{target}'].unstack()
        if return_prod_mix:
            df = prod_mix[[cl for cl in prod_mix.columns if cl.endswith(f'_{target}')]]
            # Here we only consider the local production means (not the imports)
            df = df.drop(columns=[k for k in df.columns if
                                  k.startswith("Mix_Other_")])  # delete "Mix_Other_xx" (other countries)
            # Normalize the production mix
            df = df / df.sum(axis=1).values.reshape(-1, 1)
            prod_mix_dict[f'{target}'] = df

    return (prod_mix_dict, cons_mix_dict) if return_prod_mix else cons_mix_dict


def get_impacts(mix_dict: dict, impact_matrix: pd.DataFrame, missing_mapping: str = 'error',
                is_verbose: bool = False) -> dict:
    """Computes the impact of the given electrical mix.

    Parameters
    ----------
        mix_dict: dict of pandas.DataFrame
            The mix for each target, as returned by get_mix.
        impact_matrix: pandas.DataFrame
            The impact matrix, as returned by load_impact_matrix.
        missing_mapping: str, default to 'error'
            strategy for handling producing units with not mapping.
            'error' (default) raises an error, 'worst' takes the highest impact value in
            the available set, 'unit' takes the highest impact value available from a
            similar unit type, defaults to 'error'
        is_verbose: bool, default to False
            To display progress information

    Returns
    -------
    imp_dict: dict
        A dictionary containing the impacts for each target.
        For each country, this is a dict of tables containing the dynamic impacts of 1kWh of electricity.
    """
    imp_dict = {}
    for target in mix_dict.keys():
        imp_dict[target] = compute_impacts(mix_data=mix_dict[target], impact_data=impact_matrix,
                                           strategy=missing_mapping, is_verbose=is_verbose)
    return imp_dict


def get_flows_kwh(parameters: Parameter, raw_prod_exch: pd.DataFrame) -> dict:
    """
    Compute the total production, imports and exports for each country, in kWh, at each time step.
    Useful to compute the mixes in kWh (instead of %).

    Parameters
    ----------
    parameters: ecodynelec.Parameter
        Parameters returned by load_config.
    raw_prod_exch: pandas.DataFrame
        The raw production and exchange data, as returned by load_raw_prod_exchanges.

    Returns
    -------
    flows_dict: dict of DataFrame
        A dictionary of countries and their production/imports/exports, in kWh.
    """

    flows_dict = {}
    for target in parameters.target:
        flows_dict[target] = pd.DataFrame(index=raw_prod_exch.index, columns=['production', 'imports', 'exports'])
        # We compute the incoming power (production + import) at each time step
        # then we multiply it by the relative mix matrix to get the mix in kWh
        flows_dict[target]['production'] = raw_prod_exch[
            [col for col in raw_prod_exch.columns if not col.startswith(f'Mix_') and col.endswith(target)]].sum(
            axis=1)
        flows_dict[target]['imports'] = raw_prod_exch[
            [col for col in raw_prod_exch.columns if col.startswith(f'Mix_') and col.endswith(target)]].sum(axis=1)
        flows_dict[target]['exports'] = raw_prod_exch[
            [col for col in raw_prod_exch if col.startswith(f'Mix_{target}')]].sum(axis=1)
    return flows_dict


def translate_to_timezone(parameters: Parameter, flows_dict: dict = None, prod_mix_dict: dict = None,
                          mix_dict: dict = None, prod_imp_dict: dict = None, imp_dict: dict = None,
                          is_verbose: bool = False):
    """Translates the data to the right timezone.
    The data is modified in place.

    Parameters
    ----------
        parameters: ecodynelec.Parameter
            Parameters returned by load_config.
        flows_dict: dict or None
            A dictionary of pandas.DataFrame, with the raw productions/imports/exports (in kWh) for each target.
            As returned by get_flows_kwh.
        prod_mix_dict: dict or None
            A dictionary of pandas.DataFrame, with the production mix for each target.
            As returned by get_mix.
        mix_dict: dict or None
            A dictionary of pandas.DataFrame, with the consumption mix for each target.
            As returned by get_mix.
        prod_imp_dict: dict or None
            A dictionary containing the electricity production impacts for each target.
            As returned by get_impacts.
        imp_dict: dict or None
            A dictionary containing the electricity consumption impacts for each target.
            As returned by get_impacts.
        is_verbose: bool, default to False
            To display progress information

    Returns
    -------
    (flows_dict, prod_mix_dict, mix_dict, prod_imp_dict, imp_dict): tuple of dict
        The same dictionaries, with the data translated to the right timezone.
    """
    if parameters.timezone is not None:
        if is_verbose: print(f"Adapt timezone: UTC >> {parameters.timezone}")
        for target in parameters.target:
            if flows_dict is not None:
                flows_dict[target] = localize_from_utc(data=flows_dict[target], timezone=parameters.timezone)
            if prod_mix_dict is not None:
                prod_mix_dict[target] = localize_from_utc(data=prod_mix_dict[target], timezone=parameters.timezone)
            if mix_dict is not None:
                mix_dict[target] = localize_from_utc(data=mix_dict[target], timezone=parameters.timezone)
            if prod_imp_dict is not None:
                for k in prod_imp_dict[target].keys():
                    prod_imp_dict[target][k] = localize_from_utc(data=prod_imp_dict[target][k],
                                                                 timezone=parameters.timezone)
            if imp_dict is not None:
                for k in imp_dict[target].keys():
                    imp_dict[target][k] = localize_from_utc(data=imp_dict[target][k], timezone=parameters.timezone)
    return flows_dict, prod_mix_dict, mix_dict, prod_imp_dict, imp_dict


def save_results(parameters: Parameter, flows_dict: dict = None, impact_matrix: pd.DataFrame = None,
                 prod_mix_dict: dict = None, mix_dict: dict = None, prod_imp_dict: dict = None, imp_dict: dict = None,
                 is_verbose: bool = False) -> None:
    """Saves the results in the specified directory (parameters.path.savedir).

    Parameters
    ----------
    parameters: ecodynelec.Parameter
        Parameters returned by load_config.
    flows_dict: dict or None
        A dictionary of pandas.DataFrame, with the raw productions/imports/exports (in kWh) for each target.
        As returned by get_flows_kwh.
    impact_matrix: pandas.DataFrame or None
        The impact matrix, as returned by load_impact_matrix.
    prod_mix_dict: dict or None
        A dictionary of pandas.DataFrame, with the production mix for each target, as returned by get_mix.
    mix_dict: dict or None
        A dictionary of pandas.DataFrame, with the consumption mix for each target, as returned by get_mix.
    prod_imp_dict: dict or None
        A dictionary containing the electricity production impacts for each target, as returned by get_impacts.
    imp_dict: dict or None
        A dictionary containing the electricity consumption impacts for each target, as returned by get_impacts.
    is_verbose: bool, default to False
        To display progress information
    """
    if parameters.path.savedir is not None:
        if is_verbose: print("Save data...")
        if parameters.path.mapping is not None and impact_matrix is not None:  # Impact vector saved only if use of Mapping xlsx
            saving.save_impact_vector(impact_matrix, savedir=parameters.path.savedir, cst_import=parameters.cst_imports,
                                      residual=parameters.residual_global)
        for country in parameters.target:
            path = os.path.abspath(f'{parameters.path.savedir}{country}/')
            if not os.path.isdir(path):
                os.makedirs(path)
            if flows_dict is not None:
                saving.save_dataset(data=flows_dict[country], savedir=f'{parameters.path.savedir}{country}/',
                                    name="RawFlows",
                                    freq=parameters.freq)
            if prod_mix_dict is not None:
                saving.save_dataset(data=prod_mix_dict[country], savedir=f'{parameters.path.savedir}{country}/',
                                    name="ProdMix",
                                    freq=parameters.freq)
            if mix_dict is not None:
                saving.save_dataset(data=mix_dict[country], savedir=f'{parameters.path.savedir}{country}/', name=f"Mix",
                                    freq=parameters.freq)
            if prod_imp_dict is not None:
                imp = prod_imp_dict[country]
                for k in imp:
                    saving.save_dataset(data=imp[k], savedir=f'{parameters.path.savedir}{country}/',
                                        name=f'ProdImpact_{k.replace("_", "-")}', freq=parameters.freq)
            if imp_dict is not None:
                imp = imp_dict[country]
                for k in imp:
                    saving.save_dataset(data=imp[k], savedir=f'{parameters.path.savedir}{country}/',
                                        name=f'Impact_{k.replace("_", "-")}', freq=parameters.freq)


def localize_from_utc(data: pd.DataFrame, timezone: str = 'CET') -> pd.DataFrame:
    """Converts the index of a dataframe from UTC to another time zone.

    :param data: table with TimeIndex, assumed to be UTC
    :type data: `pandas.DataFrame`
    :param timezone: the time zone to convert to, defaults to 'CET'
    :return: a table with shifted TimeIndex to the right time zone
    :rtype: `pandas.DataFrame`
    """
    return data.tz_localize(tz='utc').tz_convert(tz=timezone).tz_localize(None)


def get_producing_mix_kwh(flows_df: pd.DataFrame, prod_mix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a relative production mix to absolute values (in kWh) for the target country.

    Parameters
    ----------
    flows_df: pd.DataFrame
        A dataframe containing the raw productions/imports/exports (in kWh) for each target.
    prod_mix_df: pd.DataFrame
        A dataframe containing the relative production mix of the target country.
    Returns
    -------
        A dataframe containing the production mix in kWh for the target country.
    """

    total_kwh = flows_df['production']
    power_df = prod_mix_df.copy()
    assert np.isclose(power_df.sum(axis=1).abs().max(), 1), "Production mix sum is not equal to 1"
    power_df = power_df.multiply(total_kwh, axis=0)
    return power_df


def get_consuming_mix_kwh(flows_df: pd.DataFrame, mix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a relative consumption mix to absolute values (in kWh) for the target country.

    Parameters
    ----------
    flows_df: pd.DataFrame
        A dataframe containing the raw productions/imports/exports (in kWh) for each target.
    mix_df: pd.DataFrame
        A dataframe containing the relative consumption mix of the target country.

    Returns
    -------
        A dataframe containing the consumption mix in kWh for the target country.
    """

    total_kwh = flows_df['production'] + flows_df['imports'] - flows_df['exports']
    power_df = mix_df.copy()
    assert np.isclose(power_df.sum(axis=1).abs().max(), 1), "Consumption mix sum is not equal to 1"
    power_df = power_df.multiply(total_kwh, axis=0)
    return power_df
