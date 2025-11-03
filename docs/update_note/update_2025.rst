2025 EcoDynElec Update
======================

This section describes the main updates and changes made to the datasets and scripts of the EcoDynElec project in 2025.

----

SwissGrid
---------

**Data type:**
Quarter-hourly data covering **net and raw electricity consumption** in Switzerland, **national generation**, and **cross-border exchanges** with several European countries.

**Downloading:**
Manual download from `SwissGrid <https://www.swissgrid.ch/fr/home/operation/grid-data/generation.html>`_.
Download the ``.xlsx`` or ``.xls`` files for **all available years** (e.g., 2016 → 20XX).

**Update:**
The data update is performed by executing the ``update_all`` function located in ``ecodynelec/update.py``.

**Important:** you must **specify the path** to the folder containing **all yearly files** (every downloaded year).
The module will not work properly if one or more yearly files are missing.

**Expected folder structure:**

.. code-block:: text

    EcoDynElec/
    ├── ...
    ├── support_files/
    │   └── swissgrid_data/
    │       ├── EnergieUebersichtCH-2016.xlsx
    │       ├── EnergieUebersichtCH-2017.xlsx
    │       ├── ...
    │       └── EnergieUebersichtCH-2024.xlsx


----

OFEN (SFOE)
------------

**Data type:**
Monthly data on national electricity production (hydropower, nuclear, thermal, renewable, etc.).
Since 2024, *wind* and *solar photovoltaic* production (for years 2020–2024) have been added.

**Downloading:**
Available on `SFOE – Swiss Electricity Statistics <https://www.bfe.admin.ch/bfe/fr/home/approvisionnement/statistiques-et-geodonnees/statistiques-de-lenergie/statistique-de-l-electricite.html/#kw-101126>`_.
Make sure to save the files following the structure below:

.. code-block:: text

    EcoDynElec/
    ├── ...
    ├── support_files/
    │   └── ofen_data/
    │       ├── 2022.pdf
    │       ├── 2023.pdf
    │       ├── 2024.pdf
    │       └── ...

**Update:**
The ``sfoe_extracting.py`` module automates the update of SFOE PDF files for ``SFOE_data.csv``.
Currently, only the 2022 and 2024 reports are processed by the function.
If 2025 data become available, the ``SFOE_extracting`` function must be adapted accordingly.
It is also essential to download all PDF files from **2022 onward** to ensure the proper functioning of the application.
The SFOE data are also used by the ``auxiliary.py`` module to determine the Swiss residual share.
The ``read_ofen_pdf_file`` function has been **optimized and updated** for the **2023 and 2024 reports**.
**Make sure to extend or adapt it accordingly for future years.**

----

ENTSO-E
-------
**Data type:**
Quarter-hourly data:
- ``AggregatedGenerationPerType``
- ``PhysicalFlows``

**Downloading:**
Data are retrieved via **FileZilla**, following the procedure described in `Downloading ENTSO-E data <https://ecodynelec.readthedocs.io/en/latest/examples/downloading.html>`__

.. warning::

    The variables ``Energy_storage_IT`` and ``Other_renewable_IT`` have been newly added to the ENTSO-E datasets.
    Previously, these categories were ignored, and their corresponding impact values in ``Unit_Impact_Vector.csv`` were set to 0.
    It is now necessary to **update the associated impact values** according to the model’s requirements.


----

Energy Charts
-------------

**Data type:**
Daily data (in GWh) for Swiss electricity production — including hydropower, biogas, biomass, waste incineration, wind, photovoltaic, and other sources.

**Downloading:**
Available on `Energy Charts <https://www.energy-charts.info/charts/energy/chart.htm?l=fr&c=CH>`_
Make sure to save the files following the structure below:

.. code-block:: text

    EcoDynElec/
    ├── ...
    ├── support_files/
    │   └── ec_data/
    │       ├── energy-charts_Production_nette_d'électricité_en_Suisse_2020.csv
    │       ├── energy-charts_Production_nette_d'électricité_en_Suisse_2021.csv
    │       ├── energy-charts_Production_nette_d'électricité_en_Suisse_2023.csv
    │       └── ...

----

Pronovo
-------

**Data type:**
Quarter-hourly production data by technology (PV, wind, biomass, hydropower).
Since 2023, data are provided on a **quarterly basis**, with a distinction between **A+** (auxiliary consumption) and **A-** (actual production).
In **EcoDynElec**, only the values labeled with **A−** are used.

**Downloading:**
Available on `Pronovo <https://pronovo.ch/downloads/Lastgangprofile_nach_Technologien/Archiv/>`__
Make sure to save the files following the structure below:

.. code-block:: text

    EcoDynElec/
    ├── prod_2020
    ├── prod_2021/
    │       ├── Profils_courbe_de_charge_01.2020.csv
    │       ├── Profils_courbe_de_charge_02.2020.csv
    │       └── ...
    ├── prod_2023
    │       ├── 202303_CH_Total_Quartal_def.csv
    │       ├── 202306_CH_Total_Quartal_def.csv
    │       └── ...
    ├── ...

----

Residual model update procedure
-----------------------------------------------

To update the ``Residual_model.xlsx`` file:

1. Run the ``extract_ofen_typical_days_for_residual`` function separately.
   This function internally calls the updated ``read_ofen_pdf_file``.
   It will generate a **dictionary output** for the selected year,
   which must be **converted into a DataFrame** and saved as a ``.csv`` file.

2. Open the generated CSV file and **copy its content** next to the existing data
   in ``Residual_model.xlsx`` located in ``support_files/``.

3. Repeat the same procedure for ENTSO-E data using the
   ``extract_entsoe_daily_generation_for_residuals`` function.
   Copy and paste the new data in the same way.

4. Ensure that all relevant **Excel formulas** have been extended
   to cover the new data range — particularly those related to:
   - the **ENTSO-E mapping**,
   - **comparison: percentage difference**, and
   - **comparison: absolute difference**.
   These calculations are used when generating the ``Share_residual.csv`` file.

5. Once everything has been verified, run the ``update_all`` function.
   It will automatically export the updated data from ``Residual_model.xlsx``
   into ``Share_residual.csv``.

----

Timezone alignment and end-of-year gap
--------------------------------------

The **Swissgrid**, **Pronovo**, and **Energy Charts** datasets are originally provided in **GMT+1**
(local time, without explicit timezone information). To ensure consistency and interoperability
across all sources, these time series are systematically converted to **UTC**.

However, this alignment introduces a one-hour forward shift in the timestamps. Unless the data
extends slightly beyond December 31st (e.g., into January 1st of the following year), this
shift creates a missing hour at the very end of the last year (from 31/12 23:45 to 01/01 00:45 in UTC).

This gap does not impact the functioning of the application. Nevertheless, starting from the 2025
update, a few additional data points from early **2025** have been downloaded and integrated into
the processing pipeline for **Swissgrid**, **Pronovo**, and **Energy Charts**, ensuring complete
yearly coverage.
