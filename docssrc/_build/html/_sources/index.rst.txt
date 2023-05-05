.. Viper Core documentation master file, created by
   sphinx-quickstart on Tue Oct 12 22:51:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPARCSpy - spatially resolved CRISPR screening in python
========================================================

SPARCSpy is a scalable toolkit to analyse SPARCS datasets. The python implementation efficiently segments individual cells, generates single-cell datasets  and provides tools for the efficient deep learning classification of their phenotypes and subsequent excision using Laser Microdissection.


.. toctree::
   :maxdepth: 2
   :caption: Ecosystem:

   pages/ecosystem/

.. toctree::
   :maxdepth: 2
   :caption: Command Line Tools:
   
   pages/vipercmd
   
.. toctree::
   :maxdepth: 2
   :caption: Analysis Pipeline:
   
   pages/pipeline/introduction
   pages/pipeline/project
   pages/pipeline/segmentation
   pages/pipeline/extraction
   pages/pipeline/classification
   pages/pipeline/selection
   
.. toctree::
   :maxdepth: 1
   :caption:  Module Documentation:
   
   pages/module/processing 
   pages/module/pipeline
   pages/module/ml
   