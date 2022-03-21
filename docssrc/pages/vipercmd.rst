viper-split
====================
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
====================
.. argparse::
   :module: vipercmd.viper-stat
   :func: generate_parser
   :prog: viper-stat
   
get information on the status of viper projects.

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


viper-cleanup
====================
.. argparse::
   :module: vipercmd.viper-stat
   :func: generate_parser
   :prog: viper-cleanup
   
List intermediate files contained in viper projects that can be deleted to free up disk-space.
Can be run in dry-run to only list found files before deleteting.

Examples:
    Show found files in a folder containing multiple datasets
    ::
        viper-cleanup -n True .
        
    Result:
    ::
        Searching for intermediate files that can be deleted, this may take a moment...
        
        ProjectA
        Found the following files to delete:
        ('~/ProjectA/segmentation/input_image.h5', '42.6GiB')

        Found the following directories to delete:
        ('~/ProjectA/segmentation/tiles/7', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/6', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/3', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/11', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/19', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/14', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/17', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/5', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/2', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/4', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/1', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/13', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/0', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/16', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/20', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/9', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/12', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/15', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/10', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/8', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/18', '7.7GiB')
        Rerun with -n False to remove these files
        
        ProjectB
        Found the following files to delete:
        ('~/ProjectA/segmentation/input_image.h5', '42.6GiB')

        Found the following directories to delete:
        ('~/ProjectA/segmentation/tiles/7', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/6', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/3', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/11', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/19', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/14', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/17', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/5', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/2', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/4', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/1', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/13', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/0', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/16', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/20', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/9', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/12', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/15', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/10', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/8', '7.7GiB')
        ('~/ProjectA/segmentation/tiles/18', '7.7GiB')
        Rerun with -n False to remove these files
    
    Delete intermediate files in folder containing multiple datasets
    ::
        viper-cleanup -n False .
        
    Result:
    ::
        Searching for intermediate files that can be deleted, this may take a moment...

        ProjectA
        Deleting files...
        Deleted files with a total storage size of 200.6GiB

        ProjectB
        Deleting files...
        Deleted files with a total storage size of 200.6GiB