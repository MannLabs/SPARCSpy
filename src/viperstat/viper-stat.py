#!/usr/bin/env python

import sys, getopt
import argparse
import os
from tabulate import tabulate

def main():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Scan directory for viper projects.')
    
    # Required positional argument
    parser.add_argument('search_directory', type=str,nargs='?',
                        help='directory containing viper projects')
    
    args = parser.parse_args()
    
    
    if args.search_directory is None:
        search_directory = os.getcwd()
        
    else:
        try:
            search_directory = os.path.abspath(args.search_directory)
        except:
            print("search directory not a valid path")
    
    current_level_directories = [(os.path.join(search_directory, name),name) for name in os.listdir(search_directory)]
    
    table = []
    for path, dir_name in current_level_directories:
        
        config_name = "config.yml"
        is_project_dir = os.path.isfile(os.path.join(path,config_name))
        
        if is_project_dir:            
            segmentation_name = "segmentation/segmentation.h5"
            segmentation_finished = os.path.isfile(os.path.join(path,segmentation_name))
            
            segmentation_classes_name = "segmentation/classes.csv"
            classes_exist = os.path.isfile(os.path.join(path,segmentation_classes_name))
            
            if classes_exist:
                segmentation_classes = sum(1 for line in open(os.path.join(path,segmentation_classes_name)))
            else:
                segmentation_classes = 0
                
            extraction_name = "extraction/data/single_cells.h5"
            extraction_finished = os.path.isfile(os.path.join(path,extraction_name))
            
            if extraction_finished:
                extraction_size = sizeof_fmt(os.stat(os.path.join(path,extraction_name)).st_size)
            else:
                extraction_size = "-"
            
            table.append([dir_name, segmentation_finished, "{:,}".format(segmentation_classes), extraction_finished, extraction_size])
            
            
            
            
    if len(table) > 0:
        print(tabulate(table, headers=["Project", "Segmentation", "Cells", "Extraction", "Size"],stralign="right"))
    else:
        print("No projects found")
             

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

if __name__ == "__main__":
    main()