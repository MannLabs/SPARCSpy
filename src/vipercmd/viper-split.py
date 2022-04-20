#!/usr/bin/env python

import sys, getopt
import argparse
import os

from tabulate import tabulate
import h5py
import numpy as np
import random
from pandas import read_csv
from math import ceil
from multiprocessing import Pool
from functools import partial
from numbers import Number
from tqdm import tqdm

def generate_parser():
    # instantiate the parser
    parser = argparse.ArgumentParser(description='Manipulate existing single cell hdf5 datasets.')
    
    # define arguments
    parser.add_argument('input_dataset', type=str,
                        help='input dataset which should be split')
    
    parser.add_argument("-o","--output", action="append", nargs=2, help='Output definition <name> <length/csv>. For example -o test.h5 0.9 or -o test.h5 1000 or -o test.h5 /path/to/csv. If the sum of all lengths is <= 1, it is interpretated as fraction. Else its used as absolute value. If a csv path is given it automatically reads the number of cells from the csv path.')
    parser.add_argument("-r","--random", default=False, action='store_true', help="shuffle single cells randomly")
    
    parser.add_argument("-t","--threads", type=int, default=4, help="number of threads")
    
    parser.add_argument("-c","--compression", default=False, action='store_true', help="use lzf compression")

    return parser
    
def _main():
    
    """
    """
    
    parser = generate_parser()
    args = parser.parse_args()
    
    global input_name 
    input_name = args.input_dataset
    
    global compression_type
    compression_type = "lzf" if args.compression else None
    
    with h5py.File(args.input_dataset, 'r') as f:
        index_handle = f.get('single_cell_index')
        index_hdf5 = f['single_cell_index'][:]
        data_handle = f.get('single_cell_data')
        num_datasets = index_handle.shape[0]

    #check to see if any of the output files contain csv
    csv_status = _check_csv(args.output)
    
    global mapping 
    mapping = list(range(num_datasets)) 

    if args.random:
        if csv_status is not None:
            print(f"Random set to true in function call but csv file paths passed. To select cell ids from csv random needs to be set to false.")
            return
        else:
            random.shuffle(mapping)

    if csv_status:
        fraction_sum = sum([float(_get_length(el)) for el in args.output])
    else:    
        fraction_sum = sum([float(el[1]) for el in args.output])
    
    absolute = False
    if fraction_sum > 1:
        absolute = True
        
    if absolute and fraction_sum > num_datasets:
        print("number of output samples exceeds input samples")
        return
    
    print(f"shuffle indices: {args.random}")
    print(f"compression: {args.compression}")
    print(f"csv status: {csv_status}")
    print(f"absolute: {absolute}")
    
    print(f"{args.input_dataset} contains {num_datasets} samples")
    
    plan = []
    slices = []
    start = 0

    if not absolute:
        #this will only ever return true if there are no csv files in the list
        # so no need to account for csv files here
        for pair in args.output:
            name = pair[0]
            fraction = float(pair[1])
            plan.append((name,round(fraction*num_datasets)))
            
            for name, length in plan:
                print(f"{name} with {length} samples")
                slices.append((name, length, slice(start,length+start)))
                start += length

    else:  
        for pair in args.output:
            if _check_csv([pair]):
                name = pair[0]
                csv_path = pair[1]
                length = _get_length(pair)

                print(f"Collecting cellids from csv file for {name}.")
                print(f"\n=== starting parallel execution of cellid collection with {args.threads} threads ===")

                #matches cell ids to index location in hdf5 in batches for easier parallel processing
                batchsize = 1000

                #calculate number of batches
                number_of_batches = ceil(num_datasets/batchsize)
                
                #generate start and end index of each batch
                i = 0
                indexes = []
                for ix in range(number_of_batches):
                    if i+batchsize <= num_datasets:
                        indexes.append((i, i+batchsize))
                        i += batchsize  
                    else:
                        indexes.append((i, num_datasets))

                ids = []
                with Pool(processes=args.threads) as pool:
                    num_tasks = len(indexes)
                    for i, _ in enumerate(pool.imap_unordered(partial(_filter_classes, index_hdf5 = index_hdf5, classes_keep_path = csv_path), indexes)):
                        ids.append(_)
                        sys.stderr.write('\rdone {0:%}'.format(i/num_tasks))
                    #ids = pool.map(partial(_filter_classes, hdf5_input_path = input_name, classes_keep_path = csv_path), indexes)

                ids = _flatten(list(ids))
                slices.append((name, length, ids))

            else:
                plan = [(pair[0],round(float(pair[1]))) for pair in args.output]
                
                for name, length in plan:
                    print(f"{name} with {length} samples")
                    slices.append((name, length, slice(start,length+start)))
                    start = length     

    print(f"\n=== starting parallel execution of generation of new hdf5s with {args.threads} threads ===")
    with Pool(processes=args.threads) as pool:
        x = pool.map(partial(_write_new_list, input_hdf5 = input_name, compression_type = args.compression), slices)
    # print(len(slices))
    print("\n")
    #for param in slices:
    #    _write_new_list(param = param,  input_hdf5 = input_name, compression_type = args.compression)
    
def _write_new_list(param, input_hdf5, compression_type):
    name, length, section = param
    
    input_hdf = h5py.File(input_hdf5, 'r')
    input_index = np.array(input_hdf.get('single_cell_index'))
    input_data = input_hdf.get('single_cell_data')
    
    num_channels = input_data.shape[1]
    image_size = input_data.shape[2]
    
    output_hdf = h5py.File(name, 'w')
    output_hdf.create_dataset('single_cell_index', (length, 2) ,dtype="uint32")
    output_hdf.create_dataset('single_cell_data', (length,
                                                    num_channels,
                                                    image_size,
                                                    image_size),
                                                    chunks=(1, 1, image_size,image_size),
                                                    dtype="float16",
                                                    compression=compression_type)
    
    output_index = output_hdf.get('single_cell_index')
    output_data = output_hdf.get('single_cell_data')

    output_index = output_hdf.get('single_cell_index')
    output_data = output_hdf.get('single_cell_data')

    #actually get the values we want to extract
    print(section[0:10])
    index_to_get = input_index[section]
    
    #now sort the index according to order in old hdf5
    #this needs to be done since otherwise we cant get the images from the hdf5
    index_to_get = np.array(sorted(index_to_get, key=lambda x: x[0]))
    
    #create new index with reset position index for new hdf5
    _, z = zip(*index_to_get)
    index_new = np.array(list(zip(np.array(range(0, len(z))), np.array(z))))
    
    batchsize = 1000
    number_of_batches = ceil(len(index_new)/batchsize)

    i = 0
    indexes = []
    for ix in range(number_of_batches):
        if i+batchsize <= index_new.shape[0]:
            indexes.append((i, i+batchsize))
            i += batchsize  
        else:
            indexes.append((i, index_new.shape[0]))

    i = 0   
    for start, end in indexes:
        print(f"{name}: {i} samples written")
        #get index location in old hdf5 file
        ids, _ = zip(*index_to_get[start:end])
        #read images from old hdf5 at specified locations
        images = input_hdf["single_cell_data"][list(ids), :, :]
        
        #write out both images and index to new file
        output_data[start:end] = images
        output_index[start:end] = index_new[start:end]
        i += batchsize
    
    input_hdf.close()
    output_hdf.close()

    # for i, index in enumerate(mapping[section]):
    #     ix = input_index[index]
        
    #     #reindex index element
    #     ix[0] = i
        
    #     output_index[i] = ix
        
    #     data = input_data[index]
    #     output_data[i] = data
  
    #     if i % 10000 == 0:
    #         print(f"{name}: {i} samples written")
        
    # output_hdf.close()    
    # input_hdf.close()

def _flatten(l):
    _list = []
    for el in l:
        if isinstance(el, list):
            _list = _list + el
    return(_list)  

def _get_length(arg):
    name = arg[0]
    length = arg[1]

    if isinstance(length, Number):
        return length
    else:
        csv = read_csv(length, index_col = None)
        length = csv.shape[0]
        return length

def _check_csv(args):
    csv = False

    for arg in args:
        name = arg[0]
        length = arg[1]

        if isinstance(length, Number):
            continue
        else:
            csv = True

    return csv

def _sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def _filter_classes(ix_ranges, index_hdf5, classes_keep_path):
    start, end = ix_ranges

    #open hdf5 and get indexes
    #this needs to be performed again in this function because hdf5 file handles can not be pickled and otherwise
    #this function can not be run in parallel processes to speedup the runtime

    #with h5py.File(hdf5_input_path, "r") as f
    #    index_hdf5 = f['single_cell_index'][:]

    #open classes to keep
    classes_keep = read_csv(classes_keep_path, index_col = None)
    classes_keep = classes_keep.iloc[:, 0].values.tolist()

    #convert classes to keep to set to make comparisions more efficient
    classes_keep = set(classes_keep)

    indexes, cell_ids = zip(*index_hdf5[start:end])
    keep = [x in classes_keep for x in cell_ids]

    #hdf5.close()

    return(keep)

if __name__ == "__main__":

    _main()