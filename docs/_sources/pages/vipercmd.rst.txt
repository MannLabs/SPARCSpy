command line arguments
***********************

viper-split
=======================
.. argparse::
   :module: vipercmd.viper-split
   :func: _generate_parser
   :prog: viper-split
   

Manipulate existing single cell hdf5 datasets.
viper-split can be used for splitting, shuffleing and compression / decompression.

Examples:
    Splitting with shuffle and compression:
    ::
        viper-split single_cells.h5 -r -c -o train.h5 0.9 -o test.h5 0.05 -o validate.h5 0.05
    
    Shuffle
    ::
        viper-split single_cells.h5 -r -o single_cells.h5 1.0

    Compression
    ::
        viper-split single_cells.h5 -c -o single_cells.h5 1.0

    Decompression
    ::
        viper-split single_cells.h5 -o single_cells.h5 1.0

   
viper-stat
=======================
.. argparse::
   :module: vipercmd.viper-stat
   :func: generate_parser
   :prog: viper-stat
   
Manipulate existing single cell hdf5 datasets.
viper-split can be used for splitting, shuffleing and compression / decompression.

Examples:
    Show progress in a folder containing multiple datasets
    ::
        viper-stat
        
    Result:
    ::
        Viper-stat collecting information. This can take some time...
       slide000           True        731,468        72.6GiB
               single_cells.h5        729,775        30.5GiB
       slide001           True        755,358        69.3GiB
               single_cells.h5        753,277        30.4GiB
   