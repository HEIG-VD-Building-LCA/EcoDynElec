import pandas


# +

##########################
# #########################
# Save impact vector
# #########################
# #########################
# -

def save_impact_vector(impact_matrix, savedir, cst_import=False, residual=False):
    """Function to save the impact matrix.

    Parameters
    ----------
        impact matrix: pandas.DataFrame
            the table with the impact factors
        savedir: str
            the directory where to save
        cst_import: bool, default to False
            if constant exchange impacts are considered
        residual: bool, default to False
            if a residual is considered
    """
    add_on = ""
    if cst_import: add_on += "_CstImp"
    if residual: add_on += "_Res"
    file_name = f"Unit_Impact_Vector{add_on}.csv"
    
    impact_matrix.to_csv(savedir + file_name, index=True)


# +
##########################
# #########################
# Save Dataset
# #########################
# #########################
# -

def save_dataset(data, savedir, name, target=None, freq='H'):
    """Function to save the datasets with information of the frequency.
    
    Parameters
    ----------
        data: pandas.DataFrame
            the dataset to save
        savedir: str
            the directory where to save
        name: str
            the name of the file (excluding extension and frequency info)
        target: str, default to None
            tag of target country, to be added to the name if given.
        freq: str, default to 'H'
            the frequency
    """
    ### Formating the time extension
    tPass = {'15min':'15min','30min':'30min',"H":"hour","D":"day",'d':'day','W':"week",
             "w":"week","MS":"month","M":"month","YS":"year","Y":"year"}
    as_target = "" if target is None else f"_{target}"
    
    ### Saving
    data.to_csv(savedir+f"{name}{as_target}_{tPass[freq]}.csv",index=True, float_format='%.3g')
