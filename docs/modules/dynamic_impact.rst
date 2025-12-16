Dynamic Impact Calculation
==========================

Module dedicated to the computation of the time-dependent environmental impact of Pumped Hydro Storage (PHS/STEP).
It implements a recursive mass-balance algorithm to track the "carbon content" of the water stock based on historical pumping and natural inflows.

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