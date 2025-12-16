"""The `pipelines` module is the main module of `ecodynelec`.
It consists in a collection of high-level functions triggering
processes intricating most of the functions of the package.

This module contains:
    - execute: whole process from downloading / loading the data to the computation of impacts
    - get_inverted_matrix: process from downloading to track the electricity.
    - localize_from_utc: shifts the time-zone from results.
"""
import pandas as pd

from ecodynelec.checking import check_mapping
# +
####### Local modules
from ecodynelec.pipeline_functions import load_config, check_download, load_raw_prod_exchanges, get_mix, get_impacts, \
    translate_to_timezone, save_results, load_impact_matrix, get_flows_kwh
from ecodynelec.dynamic_impact import dynamic_impact
from ecodynelec.progress_info import ProgressInfo


# +


#############################################
# ############################################
# Easy Execution of the whole algorithm
# ############################################
# ############################################

# +


def execute(config, missing_mapping='error', is_verbose=False, progress_bar: ProgressInfo = None):
    """Executes the whole computation process, i.e. (1) downloads required data;
    (2) load auxiliary data; (3) load and correct Entso-E data; (4) compute the 
    electricity tracking; (5) computes the environmental impacts; (6) save and return.

    This function only returns the impacts of the electricity mix of the target countries (the intermediate
    results, relative mixes and mixes of production in kWh aren't saved or returned).
    See :py:func:`get_prod_mix_impacts` or :py:func:`get_inverted_matrix` for a function returning all the
    intermediate results.

    Parameters
    ----------
        config: ecodynelec.Parameter or str
            a set of configuration parameters to govern the computation,
            either as Parameter object or str pointing at a xlsx file.
        missing_mapping: str, default to 'error'
            strategy for handling producing units with not mapping.
            'error' (default) raises an error, 'worst' takes the highest impact value in
            the available set, 'unit' takes the highest impact value available from a 
            similar unit type, defaults to 'error'
        is_verbose: bool, default to False
            To display progress information
        progress_bar: ProgressInfo, default to None
            A progress bar to display the progress of the computation

    Returns
    -------
        dict of pd.DataFrame or dict of dict of pd.DataFrame:
            a collection of tables containing the dynamic impacts of 1kWh of electricity
            Note if there are multiple target countries, the data is returned in a dict of each target's impacts.
    """

    if progress_bar:
        progress_bar.set_max_value(10)
        progress_bar.progress('Load config...', 0)

    ###########################
    ###### PARAMETER & VERIF
    ######
    p = load_config(config)
    n_target = len(p.target)

    ###########################
    ###### DOWNLOAD FROM SERVER
    ######
    if progress_bar:
        progress_bar.progress('Download from ENTSO-E...')
    check_download(parameters=p, is_verbose=is_verbose)

    ###########################
    ###### LOAD DATASETS
    ######
    if progress_bar:
        progress_bar.progress('Load raw prods...')
    raw_prodExch = load_raw_prod_exchanges(parameters=p, is_verbose=is_verbose, progress_bar=progress_bar)

    if progress_bar:
        progress_bar.progress('Load impact matrix...')
    impact_matrix = load_impact_matrix(parameters=p, is_verbose=is_verbose)

    # Verify the adequacy between production and impacts
    check_mapping(mapping=impact_matrix, mix=raw_prodExch, strategy=missing_mapping)

    ########################
    ###### COMPUTE TRACKING (and add local residual)
    ######
    if progress_bar:
        progress_bar.progress('Compute mix tracking...')
    mix_dict = get_mix(p, raw_prodExch, is_verbose=is_verbose, progress_bar=progress_bar)

    ############################
    ###### COMPUTE ELEC IMPACTS
    ######
    if progress_bar:
        progress_bar.progress('Compute impacts...')
    imp_dict = get_impacts(mix_dict, impact_matrix, missing_mapping, is_verbose=is_verbose)

    ###############################
    ###### TRANSLATE INTO TIMEZONE
    ######
    if progress_bar:
        progress_bar.progress('Save data...')
    _, _, _, _, imp_dict = translate_to_timezone(p, imp_dict=imp_dict, is_verbose=is_verbose)

    ################################
    ####### SAVE DATA
    #######
    save_results(p, impact_matrix=impact_matrix, imp_dict=imp_dict, is_verbose=is_verbose)

    if progress_bar:
        progress_bar.progress('Done.')
    if is_verbose: print("done.")
    if n_target == 1:
        return imp_dict[p.target[0]]
    return imp_dict


########################################
# ######################################
# Aymeric's new pipeline
# ######################################
# ######################################
# -

def get_prod_mix_impacts(config, step_imp_memory, missing_mapping='error', is_verbose=False, progress_bar: ProgressInfo = None):
    """Executes the whole computation process, i.e. (1) downloads required data;
    (2) load auxiliary data; (3) load and correct Entso-E data; (4) compute the
    electricity tracking; (5) computes the environmental impacts; (6) save and return.

    This function returns:
    - the impacts of the electricity production mix of the target countries
    - the impacts of the electricity consumer mix of the target countries (returned by :py:func:`execute`)
    - the intermediate results, relative mixes per target country and total production/imports/exports in kWh

    Parameters
    ----------
        config: ecodynelec.Parameter or str
            a set of configration parameters to govern the computation,
            either as Parameter object or str pointing at a xlsx file.
        step_imp_memory: dict
            dictionnary containing the last STEP impact of the last year
        missing_mapping: str, default to 'error'
            strategy for handling producing units with not mapping.
            'error' (default) raises an error, 'worst' takes the highest impact value in
            the available set, 'unit' takes the highest impact value available from a
            similar unit type, defaults to 'error'
        is_verbose: bool, default to False
            To display progress information
        progress_bar: ProgressInfo, default to None
            A progress bar to display the progress of the computation

    Returns
    -------
        raw_prod_dict: pd.DataFrame or dict of pd.DataFrame
            A table containing the production, in kWh, for each electricity source for the target country (local and
            import sources).
            Note if there are multiple target countries, the data is returned in a dict of each target's production table
        prod_dict: pd.DataFrame or dict of pd.DataFrame
            A table containing the relative production mix of the target country, in %, for each production source
            (only local sources).
            Note if there are multiple target countries, the data is returned in a dict of each target's mix table.
        mix_dict: pd.DataFrame or dict of pd.DataFrame
            A table containing the relative consumption mix of the target country, in %, for each production source
            (local and import sources).
            Note if there are multiple target countries, the data is returned in a dict of each target's mix table.
        prod_imp_dict: dict of pd.DataFrame or dict of dict of pd.DataFrame
            A collection of tables containing the dynamic impacts of the production of 1kWh of electricity in the target country.
            Note if there are multiple target countries, the data is returned in a dict of each target's impacts.
        imp_dict: dict of pd.DataFrame or dict of dict of pd.DataFrame
            A collection of tables containing the dynamic impacts of 1kWh of consumed electricity in the target country.
            Note if there are multiple target countries, the data is returned in a dict of each target's impacts.
        step_imp_memory: dict containing the last STEP impact of the last year
    """

    if progress_bar:
        progress_bar.set_max_value(11)
        progress_bar.progress('Load config...', 0)

    ###########################
    ###### PARAMETER & VERIF
    ######
    p = load_config(config)
    n_target = len(p.target)

    ###########################
    ###### DOWNLOAD FROM SERVER
    ######
    if progress_bar:
        progress_bar.progress('Download from ENTSO-E...')
    check_download(parameters=p, is_verbose=is_verbose)

    ###########################
    ###### LOAD DATASETS
    ######
    if progress_bar:
        progress_bar.progress('Load raw prods...')
    raw_prodExch = load_raw_prod_exchanges(parameters=p, is_verbose=is_verbose, progress_bar=progress_bar)

    if progress_bar:
        progress_bar.progress('Load impact matrix...')
    impact_matrix = load_impact_matrix(parameters=p, is_verbose=is_verbose)

    # Verify the adequacy between production and impacts
    check_mapping(mapping=impact_matrix, mix=raw_prodExch, strategy=missing_mapping)

    ########################
    ###### COMPUTE TRACKING (and add local residual)
    ######
    if progress_bar:
        progress_bar.progress('Compute mix tracking...')
    prod_mix_dict, mix_dict = get_mix(p, raw_prodExch, return_prod_mix=True, is_verbose=is_verbose, progress_bar=progress_bar)

    ############################
    ###### COMPUTE ELEC IMPACTS
    ######
    if progress_bar:
        progress_bar.progress('Compute impacts...')
    prod_imp_dict = get_impacts(prod_mix_dict, impact_matrix, missing_mapping, is_verbose=is_verbose)
    imp_dict = get_impacts(mix_dict, impact_matrix, missing_mapping, is_verbose=is_verbose)
    if p.start.year >= 2023:
        if p.dynamic_impact:
            prod_imp_dict, imp_dict, step_imp = dynamic_impact(prod_imp_dict, imp_dict, raw_prodExch, prod_mix_dict,
                                                    mix_dict, impact_matrix, step_imp_memory, p, is_verbose=is_verbose)
            step_imp_memory[p.start.year] = step_imp
    ##########################
    ####### COMPUTE MIX IN kWh
    #######
    if progress_bar:
        progress_bar.progress('Compute mix in kWh...')
    # Drop non-production lines of the mix (i.e. the first part of the mix matrix)
    for mix in mix_dict.keys():
        prod_mix_dict[mix] = prod_mix_dict[mix].drop(
            prod_mix_dict[mix].loc[:,
            [k.startswith('Mix') and not k.endswith('Other') for k in prod_mix_dict[mix].columns]],
            axis=1).astype('float32')
        prod_mix_dict[mix] = prod_mix_dict[mix] / prod_mix_dict[mix].sum(axis=1).values.reshape(-1, 1)
        mix_dict[mix] = mix_dict[mix].drop(
            mix_dict[mix].loc[:, [k.startswith('Mix') and not k.endswith('Other') for k in mix_dict[mix].columns]],
            axis=1).astype('float32')
    flows_dict = get_flows_kwh(p, raw_prodExch)

    ###############################
    ###### TRANSLATE INTO TIMEZONE
    ######
    if progress_bar:
        progress_bar.progress('Save data...')
    flows_dict, prod_mix_dict, mix_dict, prod_imp_dict, imp_dict = translate_to_timezone(p, flows_dict=flows_dict,
                                                                          prod_mix_dict=prod_mix_dict, mix_dict=mix_dict,
                                                                          prod_imp_dict=prod_imp_dict, imp_dict=imp_dict,
                                                                          is_verbose=is_verbose)

    ################################
    ####### SAVE DATA
    #######
    save_results(p, flows_dict=flows_dict, impact_matrix=impact_matrix, prod_mix_dict=prod_mix_dict, mix_dict=mix_dict,
                 prod_imp_dict=prod_imp_dict, imp_dict=imp_dict, is_verbose=is_verbose)

    if progress_bar:
        progress_bar.progress('Done.')
    if is_verbose: print("done.")
    if n_target == 1:
        return flows_dict[p.target[0]], prod_mix_dict[p.target[0]], mix_dict[p.target[0]], prod_imp_dict[p.target[0]], imp_dict[p.target[0]], step_imp_memory
    return flows_dict, prod_mix_dict, mix_dict, prod_imp_dict, imp_dict, step_imp_memory


########################################
# ######################################
# Get inverted matrix
# ######################################
# ######################################
# -
def get_inverted_matrix(config, is_verbose=False, progress_bar: ProgressInfo = None):
    """Triggers the computation process until the electricity tracking to return the
    electricity mix in all involved countries. No data saving is involved.
    For CH, the local residual is not added to the mix, even is enabled.

    Parameters
    ----------
        config: ecodynelec.Parameter or str
            a set of configration parameters to govern the computation,
            either as Parameter object or str pointing at a xlsx file.
        is_verbose: bool, default to False
            To display progress information
        progress_bar: ProgressInfo, default to None
            A progress bar to display the progress of the computation

    Returns
    -------
        mix_matrix: list of `pandas.DataFrame`
            A collection of tables containing the decomposition of 1kWh of electricity
    """

    if progress_bar:
        progress_bar.set_max_value(7)
        progress_bar.progress('Load config...', 0)

    ###########################
    ###### PARAMETER & VERIF
    ######
    p = load_config(config)

    ###########################
    ###### DOWNLOAD FROM SERVER
    ######
    if progress_bar:
        progress_bar.progress('Download from ENTSO-E...')
    check_download(parameters=p, is_verbose=is_verbose)

    ###########################
    ###### LOAD DATASETS
    ######
    if progress_bar:
        progress_bar.progress('Load raw prods...')
    raw_prod_exch = load_raw_prod_exchanges(parameters=p, is_verbose=is_verbose,
                                                          progress_bar=progress_bar)

    ########################
    ###### COMPUTE TRACKING
    ######
    if progress_bar:
        progress_bar.progress('Compute mix tracking...')
    mix_matrix = get_mix(p, raw_prod_exch, return_matrix=True, is_verbose=is_verbose, progress_bar=progress_bar)

    if progress_bar:
        progress_bar.progress('Done.')
    if is_verbose: print("done.")
    return mix_matrix
