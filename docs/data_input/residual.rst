Production Residual
=====================

For Switzerland, additional data are gathered since the ENTSO-E data do not inform on 100% of the production means (i.e. less tracking of medium- or low-voltage production such as run-off river hydropower). Information from Swissgrid, the local Transmission System Operation (TSO), is used to adjust the exchange flows at the Swiss borders, and to add the missing domestic production not covered by ENTSO-E data.

Estimate
---------

The current implementation of EcoDynElec includes the possibility to consider a generation residual for Switzerland only. This choice is a simplification justified by preliminary studies:
    1. the discrepancy of production in Switzerland between the data provided by ENTSO-E and by Swissgrid is significant (5-15% most of the time). Although this discrepancy gradually diminishes, probably due to more information being reported to the ENTSO-E, it remains significant at the time of the publication of EcoDynElec.
    2. due to its central location in Europe, Switzerland is a country through which large amounts of electricity transit. The core assumption of EcoDynElec is that electricity mixes are homogeneous within regions (i.e. countries at the current level of details) and that cross-border exchanges result in mixing these. It is thus of main importance that production mixes of central countries are accurately represented
    3. data crossing between ENTSO-E and the local TSO of multiple countries (incl. AT, DE and FR) was conducted and no significant discrepancy was found.
    

Comparing data from ENTSO-E and Swissgrid allow to quantify the production residual, i.e. to determine how much energy is missing in the ENTSO-E data. However Swissgrid data does not provide the origin of production.

To identify the composition of the Swiss production residual, data from the SFOE annual reports is compared to ENTSO-E data. The SFOE data is a daily total for one weekday, one Saturday and one Sunday per week, with details on the generation source. Information from the SFOE comes from Table 23 (p29) of each annual `SFOE Electricity Statistics <https://www.bfe.admin.ch/bfe/en/home/supply/statistics-and-geodata/energy-statistics/electricity-statistics.html>`_ report. The SFOE statistics are usually published yearly during the 3rd week of June.

The model to compare data and estimate the residual first compares ENTSO-E and SwissGrid data to quantify the production gap down to a resolution of 15min. Then the daily comparison between ENTSO-E and SFOE mixes estimates the composition of the gap for each day of the computation period. This estimate of the composition is then assumed fixed during the whole day for each day, and thus can be applied to the sub-hourly quantitative estimate of the residual.

In the current version of EcoDynElec, the residual is found and used as a mix of small hydroelectricity (60-80%) and a mix of other sources mostly driven by PV and small thermal plants, both renewable (biomass, wastes) and non-renewable (e.g. electricity from co-generation). The methodology for comparing the SFOE data and the ENTSO-E data (composition of the mix) is detailed in the `Residual_model.xlsx <https://github.com/LESBAT-HEIG-VD/EcoDynElec/raw/main/support_files/Residual_model.xlsx>`__ file. Currently EcoDynElec covers the Swiss production gap over 2016 to 2024.

A default file is used for Switzerland, with all sources explained. Thus the the user is not required to provide its own data. The default file `Share_residual.csv <https://github.com/LESBAT-HEIG-VD/EcoDynElec/blob/main/ecodynelec/data/Share_residual.csv>`__ is used by EcoDynElec, based on the `Residual_model.xlsx <https://github.com/LESBAT-HEIG-VD/EcoDynElec/blob/main/support_files/Residual_model.xlsx>`__ file.



Implementation
---------------
The production residual can be considered in two different ways: locally and globally.

Global consideration
~~~~~~~~~~~~~~~~~~~~~
By "global", what is meant is, that this production residual is included as *another set of production units* and their generation can be exchanged with other countries, as well as the production of any other generation unit type. This can be activated by turning on the `residual_global` element of the main section of the `configuration procedure <https://ecodynelec.readthedocs.io/en/latest/data_input/parameters.html#main-parameters>`__. Considering the residual globally requires the residual to being estimated during the early stages, ***before*** electricity tracking. Switzerland must be included among the countries involved for this feature to work.

Local consideration
~~~~~~~~~~~~~~~~~~~~~
By "local", what is meant is, that this production residual is included as *a local set of production units* and their generation can ***not*** be exchanged with other countries. This can be activated by turning on the `residual_local` element of the main section of the `configuration procedure <https://ecodynelec.readthedocs.io/en/latest/data_input/parameters.html#main-parameters>`__. Considering the residual locally requires the residual to being added to the electricity mix ***after*** electricity tracking. Switzerland must be *the target country* for this feature to work.
