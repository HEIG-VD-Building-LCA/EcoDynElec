Overview
===============

This section details the connection between EcoDynElec and data of different nature. This page gives an overview of the different data and data sources. Some fields require more information and a specific page is dedicated to them. For most of it, a default solution is provided with EcoDynElec and this documentation only matters for further contributions to the development of the software.







ENTSO-E Electricity Data
*************************
EcoDynElec was built around electricity data from the `ENTSO-E <https://transparency.entsoe.eu>`__, specifically the *Generation per production type* and the *Cross-Border Physical Flows*.

**No ENTSO-E data is provided with EcoDynElec**, however the tool allows all necessary transformation, from the `download of required data <https://ecodynelec.readthedocs.io/en/latest/examples/downloading.html>`__ directly from the dataset, to the `data cleaning <https://ecodynelec.readthedocs.io/en/latest/data_input/data_cleaning.html>`__, to the `cross referencing <https://ecodynelec.readthedocs.io/en/latest/data_input/residual.html>`__, before performing the `electricity tracking <https://ecodynelec.readthedocs.io/en/latest/structure/tracking.html>`__ and computation of `environmental impacts <https://ecodynelec.readthedocs.io/en/latest/structure/impacts.html>`__. Specifically, the `data cleaning <https://ecodynelec.readthedocs.io/en/latest/data_input/data_cleaning.html>`__ procedure of electricity data is detailed in a dedicated page.








Mapping and LCA Impacts
************************

EcoDynElec computes the LCA impacts of electricity based on the electricity mix and LCA data of the electricity generation in each involved country. The data can be provided to EcoDynElec via an adapted spreadsheet or additional file. The spreadsheet `mapping_template.xlsx <https://github.com/LESBAT-HEIG-VD/EcoDynElec/blob/main/support_files/mapping_template.xlsx>`__ is a blank template hosting the structure and equations required to transform data from LCA databases and adapt them to the electricity data from ENTSO-E.

A specific page was dedicated to the topic of `LCA data <https://ecodynelec.readthedocs.io/en/latest/data_input/lca_data.html>`__ and its usage in EcoDynElec.

The data itself is **not provided per default**, for licensing reasons. Thus extracting data from LCA database and softwares is left to the user. The user can directly fill the template and provide it to EcoDynElec, as some functions of EcoDynElec were specifically tailored to extract relevant information from excel spreadsheets with this exact formatting.

However, **for the software to being fully operational**, a default `Unit Impact file <https://github.com/LESBAT-HEIG-VD/EcoDynElec/blob/main/ecodynelec/data/Unit_Impact_Vector.csv>`__ was created. This file contains, for 6 countries only (AT, CH, DE, FR, IT, CZ) and 4 impact categories, the results of LCA data processing *through the mapping spreadsheet*. If no LCA data is provided by the user, EcoDynElec will try and use this default file. More information is provided in the `LCA data <https://ecodynelec.readthedocs.io/en/latest/data_input/lca_data.html>`__ page as well.







Configuration spreadsheet
**********************************
EcoDynElec was designed to being used in Python. Thus each computation needs its configuration. The `configuration <https://ecodynelec.readthedocs.io/en/latest/data_input/parameters.html>`__ page goes over all available options. The procedures to set the configuration was extensively described in dedicated tutorials, one relying `solely on Python <https://ecodynelec.readthedocs.io/en/latest/examples/with_python.html>`__ and the other relying on a `configuration spreadsheet <https://ecodynelec.readthedocs.io/en/latest/examples/with_spreadsheet.html>`__, for which templates can be downloaded from the `GitHub repository <https://github.com/LESBAT-HEIG-VD/EcoDynElec/tree/main/examples>`__.

The configuration is considered as *a necessary input data* and it **requires the user input**: no "*complete*" default setting is provided, only some elements have default values and/or refer to default data files as mentioned in this page.








Residual model
********************

The need for considering a production residual in Switzerland comes from a difference in global electricity generation reported in SFOE annual reports and on the ENTSO-E databases. Information from the SFOE comes from Table 23 (p29) of each annual `SFOE Electricity Statistics <https://www.bfe.admin.ch/bfe/en/home/supply/statistics-and-geodata/energy-statistics/electricity-statistics.html>`_ report. The `Generation Residual <https://ecodynelec.readthedocs.io/en/latest/data_input/residual.html>`__ page further develops this the modeling hypotheses.

A default file is used for Switzerland, with all sources explained. Thus the the user is not required to provide its own data. The default file `Share_residual.csv <https://github.com/LESBAT-HEIG-VD/EcoDynElec/blob/main/ecodynelec/data/Share_residual.csv>`__ covers 2016 to 2024 and is used by EcoDynElec, based on the `Residual_model.xlsx <https://github.com/LESBAT-HEIG-VD/EcoDynElec/blob/main/support_files/Residual_model.xlsx>`__ file. The latter contains the modeling hypotheses and calculation based on the SFOE statistics, while the former only considers the result and is stored within the files of inside the package. The SFOE statistics are usually published yearly during the 3rd week of June.








SwissGrid
*************
Swiss Grid information contains global 15min frequency information about the electric grid in Switzerland. Files can be downloaded on SwissGrid website under rubric `transmission <https://www.swissgrid.ch/en/home/operation/grid-data/transmission.html>`_. Several information is used at different steps of the computation.

The exchanges between Switzerland and their neighbor countries (Cross Border Exchange, columns K-R) are used as exchanges at the Swiss borders instead of ENTSO-E data if requested by the user.

The total energy production is compared with the sum of Swiss production in ENTSO-E databases to evaluate amount of residual energy. This amount of residual is then coupled with relative information from the ``Residual_model`` file.

The default file `SwissGrid_total.csv <https://github.com/LESBAT-HEIG-VD/EcoDynElec/blob/main/ecodynelec/data/SwissGrid_total.csv>`__ is used if no information is specified. This file is a concatenation of all files of the Swiss electricity operator from the period 2016 - 2024. The user is not required to provide its own data.








SFOE Data
*************
Distribution losses from the electric network can only be considered only for Swizerland in the current version of EcoDynElec. It relies on `SFOE <https://www.bfe.admin.ch/bfe/en/home.html>`__ data.
The data used consists in tables of monthly statistics from the Swiss Federal Office of Energy (SFOE), so far covering 2012 to 2024. These values are obtained in Table A-1a (p47-p50) from each annual `SFOE Electricity Statistics <https://www.bfe.admin.ch/bfe/en/home/supply/statistics-and-geodata/energy-statistics/electricity-statistics.html>`_ report. Only the column "*Pertes*" is used in the current version of EcoDynElec, though additional information is kept for possible use extension.

A default file is used for Switzerland, with all sources explained. Thus the the user is not required to provide its own data. The default file `SFOE_data.csv <https://github.com/LESBAT-HEIG-VD/EcoDynElec/blob/main/ecodynelec/data/SFOE_data.csv>`__ is used by EcoDynElec, it directly reports the data provided in the SFOE statistics. The SFOE statistics are usually published yearly during the 3rd week of June.








Distribution losses
**********************************
The current implementation of EcoDynElec allows to consider distribution losses in Switzerland only, i.e. in the case where Switzerland is the target country. Per default, data from table A-1 (p.47-48) of the yearly `SFOE electricity statistics reports <https://www.bfe.admin.ch/bfe/en/home/supply/statistics-and-geodata/energy-statistics/electricity-statistics.html>`__ is used to infer these losses. This data is a monthly estimate of diverse electricity indicators, including distribution losses in Switzerland (column *Verluste - Perts*).

For the user to include distribution losses, the field ``network_losses`` of the main section of the `configuration procedure <https://ecodynelec.readthedocs.io/en/latest/data_input/parameters.html#main-parameters>`__ must be set to ``True``. No other input is required to use with the current data coverage (i.e. 2012 - 2024). To include different data, or to update the current one, see the SFOE_data.csv file, both in the `software files <https://github.com/LESBAT-HEIG-VD/EcoDynElec/blob/main/ecodynelec/data/SFOE_data.csv>`__ and the `support files <https://github.com/LESBAT-HEIG-VD/EcoDynElec/blob/main/support_files/SFOE_data.csv>`__. In the current implementation, the added file must have the very same structure to be loaded through the ``preprocessing.auxilary.load_grid_losses`` function and being then used in the ``tracking`` module.

The `computation tracking <https://ecodynelec.readthedocs.io/en/latest/structure/tracking.html>`__ results in a matrix, in which the first few columns correspond to the mix of production sources for every involved country. To extract only information relative to the chosen *target country*, this resulting matrix is multiplied by a vector, called *Functional Unit Vector* (FU vector) consisting in a list of zeros, with one `1` at the location corresponding to the chosen target country. When the **distribution losses** are considered, the single value in the FU vector is greater than 1, as the production *to deliver 1 kWh* to a customer in the target country is necessarily greater than 1 kWh. The distribution losses data allows to calculate what the value in the FU vector should be.







Neighbor
*************
List for all European countries of their direct neighbors, i.e. directly linked through an active power connection. Information obtained from the `ENTSO-E website <https://transparency.entsoe.eu/transmission-domain/physicalFlow/show>`_.

A default file is used by EcoDynElec, this field does not require further inputs from the user.