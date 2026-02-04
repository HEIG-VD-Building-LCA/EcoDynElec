"""The module `parameter` contains the parameter classes allowing
the management of parameters useful for in `pipelines` module and
to extract parameters from a spreadsheet.

The module contains the following classes:
    - Parameter: main parameter class
    - Filepath: handles the paths to files and diverse required data
    - Server: handles the connection to ENTSO-E server
"""
# +
import numpy as np
import pandas as pd
import os
import warnings

from ecodynelec.checking import check_frequency




# +
## Parameter
# -

class Parameter():
    """Parameter object adapted to the execution of the algorithm.
    
    Attributes
    ----------
        path: str
            FilePath object containing information about path to different documents.
        server: str
            Server object containing information about connections to ENTSO-E server.
        ctry: list
            the (sorted) list of countries to include
        target: str
            the target country where to compute the mix and impact.
        start: datetime
            starting date (utc)
        end: datetime
            ending date (utc)
        freq: str
            the time step (15min, 30min, H, d, W, M or Y)
        timezone: str
            the timezone to convert data to at the end of computation
        cst_imports: bool
            boolean to consider a constant impact for the imports
        sg_imports: bool
            boolean to replace Entso exchanges by SwissGrid exchanges
        net_exchanges: bool
            boolean to consider net exchanges at each border (i.e. no bidirectional transfer
            within one time step)
        residual_global: bool
            to include a residual for CH
        data_cleaning: bool
            to enable automatic data cleaning / filling
        ch_enr_model_path: str
            Path to the CH renewable energy production data, exported using EcoDynElec-Enr-Model. When residual global is True, this is used to replace the ``Residual_Other_CH`` category.
        storage_path: str
            Path to the CH storage data.
        dynamic_impact: bool
            To enable dynamic impact calculation

    Methods
    -------
        from_excel(excel):
            reads the values from an xlsx spreadsheet. Method is also included in
            initialization, thus no need to apply if the object is declared with
            `Parameter(excel="paht/to/file.xlsx")`.
        __setattr__:
            ensures that new values for attribures are formated correctly
        __repr__:
            allows visualization via `print()`
    """
    _is_frozen = False # Class attribute to prevent new attributes

    def __init__(self, excel=None):
        """Gather all necessary information to parametrize the execution of diverse
        functions of the module `ecodynelec.easy_use`.

        Parameters
        ----------
            excel: str, default is None
                path to .xlsx spreadsheet containing parameter information
        """
        self.path = Filepath()
        self.server = Server()

        self.ctry = sorted(["CH","FR","IT","DE","CZ","AT"])
        self.target = ["CH"]

        self.start = pd.to_datetime("2017-02-01 00:00", yearfirst=True) # first considered date
        self.end = pd.to_datetime("2017-02-01 23:00", yearfirst=True) # last considered date
        self.freq = 'H'
        self.timezone = 'UTC'

        self.cst_imports = False
        self.sg_imports = False
        self.net_exchanges = False
        self.network_losses = False
        self.residual_local = False
        self.residual_global = False
        self.data_cleaning = True
        self.dynamic_impact = True
        self.interpolated_stock = True

        self.ch_enr_model_path = None
        self.storage_path = None

        self.uvek_data = False

        if excel is not None: # Initialize with an excel file
            self.from_excel(excel)

        self._is_frozen = True # Freeze the list of attributes

    def __repr__(self):
        text = {}
        attributes = ["ctry","target","start","end","freq","timezone","cst_imports","net_exchanges",
                      "network_losses","sg_imports", "residual_local", "residual_global", 'data_cleaning', 'uvek_data']
        for a in attributes:
            text[a] = getattr(self, a)

        return ( "\n".join( [f"{a} --> {text[a]}" for a in text] )
                + f"\n\n{self.path} \n{self.server}" )

    def __setattr__(self, name, value):
        _booleans = ["cst_imports","sg_imports","net_exchanges","network_losses",
                    "residual_global","data_cleaning", "uvek_data"] # Define boolean variables

        if np.logical_and(self._is_frozen, not hasattr(self, name)):
            raise AttributeError(f"'parameter' object has no attribute '{name}'")
        elif name in ['start','end']:
            super().__setattr__(name, pd.to_datetime(value, yearfirst=True)) # set as time
        elif name == 'ctry':
            super().__setattr__(name, sorted(value)) # always keep sorted
        elif name in ['freq','frequency']:
            check_frequency(value)
            if value in ['Y','M']: # Start of Month or Year only.
                super().__setattr__(name, value+"S")
            else: super().__setattr__(name, value)
        elif name in _booleans:
            super().__setattr__(name, bool(value))
        elif name in ['path','server']:
            self._set_subclass(name, value)
        elif name == 'residual_local':
            # residual_local isn't supported anymore with the new local production system
            # by default, we now set residual_global to True and set the resulting residual to be local
            if bool(value):
                print("Warning: 'residual_local' is deprecated. Use 'residual_global' instead.")
                print('Automatically setting "residual_global" to True.')
                self.residual_global = True
            else:
                super().__setattr__(name, bool(value))
        else:
            super().__setattr__(name, value) # otherwise just set value

    def _set_subclass(self, name, value):
        match = [("path",Filepath),('server',Server)]
        if any([ ((name==n)&(isinstance(value, v))) for n,v in match]):
            super().__setattr__(name, value)
        else:
            raise TypeError(f"{name} attribute can not be of instance {type(value)}")

    def _dates_from_excel(self, array):
        adapt = lambda x: ("0" if x<10 else "") + str(x)
        date = array.fillna(0)
        if date.sum()==0: return None
        if len(date)<5: date = pd.Series( np.concatenate([ date, [1,1,0,0][len(date)-5:] ]) )
        return "{0}-{1}-{2} {3}:{4}".format(*date.apply(adapt).values)

    def _set_to_None(self):
        "Turn NaN attributes into None (e.g. from Excel, empty cells turns into NaN)"
        attributes = [a for a in dir(self) if ((not a.startswith("_"))&(not callable( getattr(self, a) )))]
        for a in attributes:
            if np.all( pd.isna(getattr(self, a)) ): setattr( self, a, None )

    def from_excel(self, excel):
        """Extract parameters information from a .xlsx spreadsheet.

        Parameters
        ----------
            excel: str
                path to a .xlsx spreadsheet
        """
        param_excel = pd.read_excel(excel, sheet_name="Parameter", index_col=0, header=None, dtype='O')

        self.ctry = np.sort(param_excel.loc["countries"].dropna().values)
        self.target = param_excel.loc['target'].iloc[0]
        if isinstance(self.target, str): self.target = [self.target]

        self.start = self._dates_from_excel(param_excel.loc['start'])
        self.end = self._dates_from_excel(param_excel.loc['end'])
        self.freq = param_excel.loc['frequency'].iloc[0]
        self.timezone = param_excel.loc['timezone'].iloc[0]

        self.cst_imports = param_excel.loc['constant exchanges'].iloc[0]
        self.sg_imports = param_excel.loc['exchanges from swissGrid'].iloc[0]
        self.net_exchanges = param_excel.loc['net exchanges'].iloc[0]
        self.network_losses = param_excel.loc['network losses'].iloc[0]
        self.residual_local = param_excel.loc['residual local'].iloc[0]
        self.residual_global = param_excel.loc['residual global'].iloc[0]
        self.data_cleaning = param_excel.loc['data cleaning'].iloc[0]

        if 'CH energy model path' in param_excel.index:
            self.ch_enr_model_path = param_excel.loc['CH energy model path'].iloc[0]

        self.path = self.path.from_excel(excel)
        self.server = self.server.from_excel(excel)

        self._set_to_None()
        return self


# +
## Filepath
# -

class Filepath():
    """Collection of `ecodynelec` parameters specifically related to data to be loaded from local machine.
    
    Attributes
    ----------
        rootdir: str
            root directory of the experiment (highest common folder). Useful mainly within the class.
        generation: str
            directory containing generation files from ENTSO-E database
        exchanges: str
            directory containing cross-border flow  files from ENTSO-E database
        savedir: str
            directory where to save the results. Default: None (no saving)
        ui_vector: str
            file with the impact matrix in a directly usable format as .csv
            (impact per kWh produced for each production unit)
        mapping: str
            file with the mapping as a spreadsheet (impact per kWh produced for each production unit)
        neighbours: str
            file gathering the list of neighbours of each european country
        gap: str
            file with estimations of the nature of the residual
        swissGrid: str
            file with production and cross-border flows from Swiss Grid
        networkLosses: str
            file with estimation of the power grid losses.

    Methods
    -------
        from_excel(excel):
            reads the values from an xlsx spreadsheet.
        __setattr__:
            ensures that new values for attribures are formated correctly
        __repr__:
            allows visualization via `print()`
    """
    _is_frozen = False # Class attribute to prevent new attributes

    def __init__(self, excel=None):
        """Gather parameters about local data files for the execution of diverse
        functions of the module `ecodynelec.easy_use`.

        Parameters
        ----------
            excel: str, default is None
                path to .xlsx spreadsheet containing parameter information
        """
        self.generation = None
        self.exchanges = None
        self.savedir = None

        self.ui_vector = None
        self.ui_network = None
        self.uvek_ui_vector = None
        self.mapping = None
        self.neighbours = None
        self.gap = None
        self.swissGrid = None
        self.networkLosses = None

        if excel is not None: # Initialize with an excel file
            self.from_excel(excel)

        self._is_frozen = True # Freeze the list of attributes

    def __repr__(self):
        attributes = ["generation","exchanges","savedir",
                      "ui_vector","mapping","neighbours","gap","swissGrid",
                      "networkLosses", "ui_network", "uvek_ui_vector"]
        text = ""
        for a in attributes:
            text += f"Filepath to {a} --> {getattr(self, a)}\n"
        return text

    def __setattr__(self, name, value):
        if np.logical_and(self._is_frozen, not hasattr(self, name)):
            raise AttributeError(f"'parameter.path' object has no attribute '{name}'")
        elif pd.isna(value):
            super().__setattr__(name, None) # set an empty info
        elif os.path.isdir(r"{}".format(value)):
            super().__setattr__(name, os.path.abspath(r"{}".format(value))+"/")
        elif os.path.isfile(r"{}".format(value)):
            super().__setattr__(name, os.path.abspath(r"{}".format(value)))
        elif np.logical_and(not self._is_frozen, name=='_is_frozen'):
            super().__setattr__(name, value)
        else:
            if name in ('generation','exchanges','savedir'): # Create them and send a warning
                msg = f"Unidentified {name} directory {os.path.abspath(value)}. It was created as new empty directory."
                warnings.warn(msg, FileNotFoundWarning)
                os.makedirs( os.path.abspath(value) ) # Create the folder
                super().__setattr__(name, os.path.abspath(r"{}".format(value))+"/") # Create
            else:
                raise FileNotFoundError(f'Unidentified file or directory: {os.path.abspath(value)}')

    def from_excel(self, excel):
        """Extract parameters information from a .xlsx spreadsheet.

        Parameters
        ----------
            excel: str
                path to a .xlsx spreadsheet
        """
        param_excel = pd.read_excel(excel, sheet_name="Filepath", index_col=0, header=None)

        self.generation = param_excel.loc['generation directory'].iloc[0]
        self.exchanges = param_excel.loc['exchange directory'].iloc[0]
        self.savedir = param_excel.loc['saving directory'].iloc[0]

        self.ui_vector = param_excel.loc['UI vector'].iloc[0]
        self.mapping = param_excel.loc['mapping file'].iloc[0]
        self.neighbours = param_excel.loc['neighboring file'].iloc[0]
        self.gap = param_excel.loc['gap file'].iloc[0]
        self.swissGrid = param_excel.loc['file swissGrid'].iloc[0]
        self.networkLosses = param_excel.loc['file grid losses'].iloc[0]

        return self



# +
## SERVER
# -

class Server():
    """Server object allowing to parametrize the automatic downloading of data files.
    
    Attributes
    ----------
        useServer: bool, default to False
            to download from server or to skip this time consuming step.
        removeUnused: bool, default to False
            to clear the target directory from non-related files.
        host: str, default to "sftp-transparency.entsoe.eu"
            specify the host database for the connection
        port: int, deafult to 22
            specify the port to use
        username: str, default to None
            the username for the account to connect with
        password: str, default to None
            the password of the account. No encryption on the `ecodynelec` end. The credentials 
            (`username` and `password`) are asked during the execution if missing from this
            class. Best practice for occasional use of the download functionality is to enter
            your credentials only during the main execution and not via this class or in spreadsheet.
        _nameGenerationFile: str, default to "AggregatedGenerationPerType_16.1.B_C.csv"
            general naming of files to download for unit generation on the ENTSO-E server. This
            excludes the first part of the files names, which involve the year and month.
        _nameExchangesFile: str, default to "PhysicalFlows_12.1.G.csv"
            general naming of files to download for phisical cross-border flows on the ENTSO-E server.
            This excludes the first part of the files names, which involve the year and month.
        _remoteGenerationDir: str, default to "/TP_export/AggregatedGenerationPerType_16.1.B_C/"
            path to files of interest for unit generation on the ENTSO-E server.
        _remoteExchangesDir: str, default to "/TP_export/PhysicalFlows_12.1.G/"
            path to files of interest for phisical cross-border flows on the ENTSO-E server.

    Methods
    -------
        from_excel(excel):
            reads the values from an xlsx spreadsheet.
        __setattr__:
            ensures that new values for attribures are formated correctly
        __repr__:
            allows visualization via `print()`
    """
    _is_frozen = False # Class attribute to prevent new attributes

    def __init__(self, excel=None):
        """Gather downloading parameters to configurate the execution of diverse
        functions of the module `ecodynelec.easy_use`.

        Parameters
        ----------
            excel: str, default is None
                path to .xlsx spreadsheet containing parameter information
        """
        # Root of file names on server
        self._nameGenerationFile = "AggregatedGenerationPerType_16.1.B_C.csv"
        self._nameExchangesFile = "PhysicalFlows_12.1.G.csv"

        # Directory with files
        self._remoteGenerationDir = "/TP_export/AggregatedGenerationPerType_16.1.B_C/"
        self._remoteExchangesDir = "/TP_export/PhysicalFlows_12.1.G/"

        # Information about the server connection
        self.useServer = False
        self.removeUnused = False
        self.host = "sftp-transparency.entsoe.eu"
        self.port = 22

        # Connection login
        self.username = None
        self.password = None # Preferable to ask for it. May be interesting to store if we can encrypt.

        if excel is not None: # Initialize with an excel file
            self.from_excel(excel)

        self._is_frozen = True # Freeze the list of attributes


    def __repr__(self):
        attributes = ['useServer','host','port','username','password','removeUnused',
                      '_remoteGenerationDir','_remoteExchangesDir']
        text = ""
        for a in attributes:
            if a!='password': text += f"Server for {a} --> {getattr(self, a)}\n"
            elif isinstance( getattr(self, a), str ): text += f"Server for {a} --> {'*'*len(a)}\n"
            else: text += f"Server for {a} --> \n"
        return text

    def __setattr__(self, name, value):
        if np.logical_and(self._is_frozen, not hasattr(self, name)):
            raise AttributeError(f"'parameter.server' object has no attribute '{name}'")
        elif pd.isna(value):
            if name in ['useServer','removeUnused']:
                super().__setattr__(name, False) # set False
            else: super().__setattr__(name, None) # set an empty info
        elif name in ['useServer','removeUnused']:
            if value in [0,'False','No','',' ','-','/']:
                super().__setattr__(name, False)
            elif value in [1,'True','Yes','',' ','-','/']:
                super().__setattr__(name, True)
            else: raise TypeError(f"{name} attribute can not be {value}")
        else:
            super().__setattr__(name, value) # Set the value


    def from_excel(self, excel):
        """Extract parameters information from a .xlsx spreadsheet.

        Parameters
        ----------
            excel: str
                path to a .xlsx spreadsheet
        """
        param_excel = pd.read_excel(excel, sheet_name="Server", index_col=0, header=None)

        self.host = param_excel.loc['host'].iloc[0]
        self.port = param_excel.loc['port'].iloc[0]
        self.username = param_excel.loc['username'].iloc[0]
        self.password = param_excel.loc['password'].iloc[0]
        self.useServer = param_excel.loc['use server'].iloc[0]
        self.removeUnused = param_excel.loc['remove unused'].iloc[0]

        return self



# +
## WARNING CLASS
# -

class FileNotFoundWarning(UserWarning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)