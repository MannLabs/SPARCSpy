#!/usr/bin/env python

import sys, getopt
import argparse
import os

from tabulate import tabulate
import h5py
import numpy as np
import random
from multiprocessing import Pool

def _generate_parser():
    # instantiate the parser
    parser = argparse.ArgumentParser(description='Manipulate existing single cell hdf5 datasets.')
    
    # define arguments
    parser.add_argument('input_dataset', type=str,
                        help='input dataset which should be split')
    
    parser.add_argument("-o","--output", action="append", nargs=2, help='Output definition <name> <length>. For example -o test.h5 0.9 or or -o test.h5 1000. If the sum of all lengths is <= 1, it is interpretated as fraction. Else its used as absolute value')
    parser.add_argument("-r","--random", default=False, action='store_true', help="shuffle single cells randomly")
    
    parser.add_argument("-t","--threads", type=int, default=4, help="number of threads")
    
    parser.add_argument("-c","--compression", default=False, action='store_true', help="use lzf compression")
    
    return parser
    
def _main():
    
    """
        :param calibration_points: Calibration coordinates in the form of :math:`(3, 2)`.
        :type calibration_points: :class:`numpy.array`, optional
    """
    
    
    parser = generate_parser()
    args = parser.parse_args()
    
    
    global input_name 
    input_name = args.input_dataset
    
    global compression_type
    compression_type = "lzf" if args.compression else None
    
    input_hdf = h5py.File(args.input_dataset, 'r')
    index_handle = input_hdf.get('single_cell_index')
    data_handle = input_hdf.get('single_cell_data')

    
    num_datasets = index_handle.shape[0]
    
    global mapping 
    mapping = list(range(num_datasets)) 
    if args.random:
        random.shuffle(mapping)
        
    print(f"shuffle indices: {args.random}")
    print(f"compression: {args.compression}")
    
    print(f"{args.input_dataset} contains {num_datasets} samples")
    
    fraction_sum = sum([float(el[1]) for el in args.output])
    
    absolute = False
    if fraction_sum > 1:
        absolute = True
        
    if absolute and fraction_sum > num_datasets:
        print("number of output samples exceeds input samples")
        return
    
    plan = []
    
    if not absolute:
        for pair in args.output:
            name = pair[0]
            fraction = float(pair[1])
            plan.append((name,round(fraction*num_datasets)))
    else:
        plan = [(pair[0],round(float(pair[1]))) for pair in args.output]
    
    slices = []
    start = 0
    
    for name, length in plan:
        print(f"{name} with {length} samples")
        slices.append((name, length, slice(start,length+start)))
        start = length
       
    print(f"\n=== starting parallel execution with {args.threads} threads ===")
    with Pool(processes=args.threads) as pool:
        x = pool.map(_write_new_list,slices)
    
    input_hdf.close()
    
def _write_new_list(param):
    name, length, section = param
    
    input_hdf = h5py.File(input_name, 'r')
    input_index = input_hdf.get('single_cell_index')
    input_data = input_hdf.get('single_cell_data')
    
    num_channels = input_data.shape[1]
    image_size = input_data.shape[2]
    
    output_hdf = h5py.File(name, 'w')
    output_hdf.create_dataset('single_cell_index', (length,2) ,dtype="uint32")
    output_hdf.create_dataset('single_cell_data', (length,
                                               num_channels,
                                               image_size,
                                               image_size),
                                               chunks=(1,1,image_size,image_size),
                                               dtype="float16",
                                               compression=compression_type)
    
    output_index = output_hdf.get('single_cell_index')
    output_data = output_hdf.get('single_cell_data')
    
    
    for i, index in enumerate(mapping[section]):
        ix = input_index[index]
        
        #reindex index element
        ix[0] = i
        
        
        output_index[i] = ix
        
        
        data = input_data[index]
        output_data[i] = data
  
        
        if i % 10000 == 0:
            print(f"{name}: {i} samples written")
        
    output_hdf.close()    
    input_hdf.close()

def _sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

if __name__ == "__main__":

    _main()