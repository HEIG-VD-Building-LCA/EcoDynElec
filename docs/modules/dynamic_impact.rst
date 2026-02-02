Dynamic Storage Modeling (Pumped Storage)
=========================================

This section details the implementation of the dynamic storage module, designed to refine the environmental assessment of Pumped Storage Hydropower (STEP/PSH) stations. Unlike previous approaches based on static mixes, this module calculates the actual carbon intensity of turbined water based on the pumping history.

Methodology
-----------

The modeling relies on two distinct layers to distinguish naturally sourced water (neutral) from pumped water (grid impact):

1.  **Physical Layer (Mass Balance)**: Hourly reconstruction of the reservoir's composition.
2.  **Environmental Layer (Carbon Tracking)**: Recursive calculation of the stock's carbon intensity.

The implemented recursive equation is as follows:

.. math::

    I_{STEP}(t) = f_{renouv}(t) \cdot I_{mix}(t) + (1 - f_{renouv}(t)) \cdot I_{STEP}(t-1)

Where:
*   :math:`I_{STEP}(t)` is the carbon intensity of the turbined electricity.
*   :math:`I_{mix}(t)` is the carbon intensity of the Swiss consumption mix at the time of pumping.
*   :math:`f_{renouv}(t)` is the renewal factor of the pumped stock.

Data Sources
------------

Activating this module requires specific data replacing or complementing standard ENTSO-E flows:

*   **Pronovo**: Use of raw flows to distinguish Injection (Turbining) from Withdrawal (Pumping). This corrects the bias of "net" production.
*   **Energy Charts**: Fill levels of Swiss reservoirs (interpolated to the hour) via the file ``Storage.csv``.

Software Implementation
-----------------------

The integration of dynamic storage affects several modules of the EcoDynElec package:

*   **`Dynamic impact` Module**: A new engine module that handles the physical modeling of the reservoir and the environmental calculation.
*   **`Extracting` & `Loading` Modules**: Adapted to ingest raw Pronovo flows and stock levels.
*   **`Parameter` Module**: Addition of a "flag" (selector) allowing the user to enable or disable the dynamic calculation.

Usage
-----

To enable dynamic storage, the corresponding parameter must be defined in the configuration:

.. code-block:: python

    config.dynamic_storage = True  # Activation example

Impact on Results
-----------------

The dynamic approach highlights the flexibility role of Pumped Storage stations. Results show that turbined water often has a lower carbon footprint than static assumptions (typically fixed at 80 gCO2eq/kWh) because pumping predominantly occurs during off-peak hours with low carbon intensity.

.. automodule:: ecodynelecg.dynamic_impact
   :members:
   :undoc-members:
   :show-inheritance:


Main Calculation Functions
--------------------------

.. autofunction:: ecodynelec.dynamic_impact.dynamic_impact

.. autofunction:: ecodynelec.dynamic_impact.dynamic_storage_shares


Data Loading and Processing
---------------------------

.. autofunction:: ecodynelec.dynamic_impact.load_concat_files

.. autofunction:: ecodynelec.dynamic_impact.interpolate_storage_hourly