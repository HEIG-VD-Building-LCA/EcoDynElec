sfoe_extracting
============================================

.. automodule:: ecodynelec.preprocessing.sfoe_extracting
   :members:
   :undoc-members:
   :show-inheritance:

.. _SFOE-data-downloading:

SFOE data downloading :
-------------------------------------------

SFOE data
'''''''''''''

SFOE data can be found on `SFOE <https://www.bfe.admin.ch/bfe/fr/home/approvisionnement/statistiques-et-geodonnees/statistiques-de-lenergie/statistique-de-l-electricite.html/>`_ under the "Electricity statistics" section. Only pdf from 2022 to later are needed.

The downloaded .pdf files should then be placed in a 'support_files/ofen_data' and rename with the year "20XX.pdf"
For example:

.. code-block:: text

    EcoDynElec/
    ├── ...
    ├── support_files/
    │   ├── ofen_data/
    │       ├── 2022.pdf
    │       ├── 2023.pdf
    │       └── ...
