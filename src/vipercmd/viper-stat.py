#!/usr/bin/env python

import sys, getopt
import argparse
import os

from tabulate import tabulate
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool
from colorama import init
from colorama import Fore, Back, Style
import h5py

def generate_parser():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Scan directory for viper projects.')
    
    # Required positional argument
    parser.add_argument('search_directory', type=str,nargs='?',
                        help='directory containing viper projects')
    
    parser.add_argument("-t","--threads", type=int, default=8, help="number of threads")
    
    parser.add_argument("-r","--recursion", type=int, default=5, help="levels of recursion")
    return parser
    
def main():
   
    print(f"Viper-stat collecting information. This can take some time...")
    
    
    parser = generate_parser()
    args = parser.parse_args()
    
    global num_threads 
    num_threads= args.threads
    
    if args.search_directory is None:
        search_directory = os.getcwd()
        
    else:
        try:
            search_directory = os.path.abspath(args.search_directory)
        except:
            print("search directory not a valid path")
        
    table = scan_directory(args.recursion, search_directory)
    
    # check if any projects were found
    if len(table) > 0:
        
        table.sort(key=lambda x: x[0])
        
        for line in table:
            print_project(line)
    else:
        print("No projects found")
def print_project(line):
    
    
    out_line = []
    Fore.RED + 'You can colorize a single line.' + Style.RESET_ALL
    out_line.append('\033[1m' + str(line[0]).rjust(15))
    
    color = Fore.GREEN if line[1] else Fore.RED
    
    out_line.append(color + str(line[1]).rjust(15) + Style.RESET_ALL)
    out_line.append('\033[1m' + str(line[2]).rjust(15) + Style.RESET_ALL)
    out_line.append('\033[1m' + str(line[3]).rjust(15) + Style.RESET_ALL)
    print("".join(out_line))
    
    pad = " "*15
    
    line[-1].sort(key=lambda x: x[0])
    
    for extract in line[-1]:
        print(Fore.BLUE +pad + str(extract[0]).rjust(15)+str(extract[1]).rjust(15)+str(extract[2]).rjust(15)+ Style.RESET_ALL)

    

def scan_directory(levels_left, path):
    
    HDF_FILETYPES = ["hdf", "hf", "h5"]

    if levels_left > 0:

        config_name = "config.yml"
        is_project_dir = os.path.isfile(os.path.join(path,config_name))

        if is_project_dir:     

            dir_name = os.path.basename(path)
            
            project_size = get_dir_size(path)

        

            segmentation_name = "segmentation/segmentation.h5"
            segmentation_finished = os.path.isfile(os.path.join(path,segmentation_name))

            segmentation_classes_name = "segmentation/classes.csv"
            classes_exist = os.path.isfile(os.path.join(path,segmentation_classes_name))

            if classes_exist:
                segmentation_classes = sum(1 for line in open(os.path.join(path,segmentation_classes_name)))
            else:
                segmentation_classes = 0
                
            extraction_data = "extraction/data"
            extraction_dir = os.path.join(path,extraction_data)
            
            extractions = []
            if os.path.isdir(extraction_dir):
            
                current_level_files = [ name for name in os.listdir(extraction_dir) if os.path.isfile(os.path.join(extraction_dir, name))]


                for i, file in enumerate(current_level_files):
                    filetype = file.split(".")[-1]
                    filename = file.split(".")[0]

                    if filetype in HDF_FILETYPES:

                        file_path = os.path.join(extraction_dir,file)

                        size = sizeof_fmt(os.stat(os.path.join(extraction_dir,file)).st_size)
                        length = get_dataset_length(os.path.join(extraction_dir,file))

                        extractions.append((file, length, size))

            


            return [[dir_name, segmentation_finished, "{:,}".format(segmentation_classes), sizeof_fmt(project_size),extractions]]

        else:        

            # iterate all subfolders
            current_level_directories = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

            with Pool(max_workers=num_threads) as pool:
                projects = pool.map(partial(scan_directory, levels_left-1),current_level_directories)
            
            return list(flatten(projects))

def get_dir_size(path):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total

def get_dataset_length(path):
    
    try:
        input_hdf = h5py.File(path, 'r')
        index_handle = input_hdf.get('single_cell_index')
        length = len(index_handle)
        input_hdf.close()
        return "{:,}".format(length) 
    except:
        return "denied"
    
from collections.abc import Iterable
# adapted from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(l):
    for el in l:
        if isinstance(el, Iterable):
            if len(el)>0:
                yield el[0]
        else:
            # pool.map might return None on subfolders
            if el is not None:
                yield el
            
# https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

if __name__ == "__main__":
    init()
    main()