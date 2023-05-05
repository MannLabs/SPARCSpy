from datetime import datetime
from operator import index
import os
import numpy as np
import pandas as pd
import csv
from functools import partial
from multiprocessing import Pool
import h5py
from tqdm import tqdm
from itertools import compress

from skimage.filters import gaussian
from skimage.morphology import disk, dilation

from scipy.ndimage import binary_fill_holes

from vipercore.processing.segmentation import numba_mask_centroid, _return_edge_labels
from vipercore.processing.utils import plot_image, flatten
from vipercore.processing.preprocessing import percentile_normalization, MinMax
from vipercore.pipeline.base import ProcessingStep

import uuid
import shutil
import timeit

import _pickle as cPickle
from matplotlib.pyplot import imshow, figure

class HDF5CellExtraction(ProcessingStep):

    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_DATA_FILE = "single_cells.h5"
    DEFAULT_SEGMENTATION_DIR = "segmentation"
    DEFAULT_SEGMENTATION_FILE = "segmentation.h5"
    DEFAULT_DATA_DIR = "data"
    CLEAN_LOG = False

    #new parameters to make workflow adaptable to other types of projects
    channel_label = "channels"
    segmentation_label = "labels"
    
    def __init__(self,
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
        if not os.path.isdir(self.directory):
                os.makedirs(self.directory)

        base_directory = self.directory.replace("/extraction", "")

        self.input_segmentation_path = os.path.join(base_directory, self.DEFAULT_SEGMENTATION_DIR, self.DEFAULT_SEGMENTATION_FILE)
        self.filtered_classes_path = os.path.join(base_directory, self.DEFAULT_SEGMENTATION_DIR, "classes.csv")
        self.output_path = os.path.join(self.directory, self.DEFAULT_DATA_DIR, self.DEFAULT_DATA_FILE)

        #extract required information for generating datasets
        self.get_compression_type()
        self.get_classes_path()

        self.save_index_to_remove = []
        
        """class can be initiated to create a WGA extraction workfow

        :param config: Configuration for the extraction passed over from the :class:`pipeline.Dataset`
        :type config: dict

        :param string: Directiory for the extraction log and results. is created if not existing yet
        :type config: string
        
        :param debug: Flag used to output debug information and map images
        :type debug: bool, default False
        
        :param overwrite: Flag used to recalculate all images, not yet implemented
        :type overwrite: bool, default False
        """
                     
    def get_compression_type(self):
        self.compression_type = "lzf" if self.config["compression"] else None
        return(self.compression_type)

    def get_classes_path(self):
        self.classes_path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR, "classes.csv")
        return self.classes_path

    def get_channel_info(self):
        with h5py.File(self.input_segmentation_path, 'r') as hf:

            hdf_channels = hf.get(self.channel_label)
            hdf_labels = hf.get(self.segmentation_label)

            if len(hdf_channels.shape) == 3:
                self.n_channels_input = hdf_channels.shape[0]
            elif len(hdf_labels.shape) == 4:
                self.n_channels_input = hdf_channels.shape[1]

            self.log(f"Using channel label {hdf_channels}")
            self.log(f"Using segmentation label {hdf_labels}")

            if len(hdf_labels.shape) == 3:
                self.n_segmentation_channels = hdf_labels.shape[0]
            elif len(hdf_labels.shape) == 4:
                self.n_segmentation_channels = hdf_labels.shape[1]

            self.n_channels_output = self.n_segmentation_channels + self.n_channels_input

    def get_output_path(self):
        self.extraction_data_directory = os.path.join(self.directory, self.DEFAULT_DATA_DIR)
        return self.extraction_data_directory    

    def setup_output(self):
        self.extraction_data_directory = os.path.join(self.directory, self.DEFAULT_DATA_DIR)
        if not os.path.isdir(self.extraction_data_directory):
            os.makedirs(self.extraction_data_directory)
            self.log("Created new data directory " + self.extraction_data_directory)   
    
    def parse_remapping(self):
        self.remap = None
        if "channel_remap" in self.config:
            char_list = self.config["channel_remap"].split(",")
            self.log("channel remap parameter found:")
            self.log(char_list)
            
            self.remap = [int(el.strip()) for el in char_list]

    def get_classes(self, filtered_classes_path):
        self.log(f"Loading filtered classes from {filtered_classes_path}")
        cr = csv.reader(open(filtered_classes_path,'r'))
        filtered_classes = [int(el[0]) for el in list(cr)]

        self.log("Loaded {} filtered classes".format(len(filtered_classes)))
        filtered_classes = np.unique(filtered_classes) #make sure they are all unique
        self.log("After removing duplicates {} filtered classes remain.".format(len(filtered_classes)))

        class_list = list(filtered_classes)
        if 0 in class_list: class_list.remove(0)
        self.num_classes = len(class_list)

        return(class_list)
    
    def generate_save_index_lookup(self, class_list):
        lookup = pd.DataFrame(index = class_list)
        return(lookup)
    
    def verbalise_extraction_info(self):
        #print some output information
        self.log(f"Extraction Details:")
        self.log(f"--------------------------------")
        self.log(f"Input channels: {self.n_channels_input}")
        self.log(f"Input labels: {self.n_segmentation_channels}")
        self.log(f"Output channels: {self.n_channels_output}")
        self.log(f"Number of classes to extract: {self.num_classes}")
    
    def _get_arg(self, class_list):
        results = list(zip(range(len(class_list)), class_list))
        return(results)

    def _initialize_tempmmap_array(self):
        #define as global variables so that this is also avaialable in other functions
        global _tmp_single_cell_data, _tmp_single_cell_index

        self.single_cell_index_shape = (self.num_classes,2)
        self.single_cell_data_shape = (self.num_classes,
                                        self.n_channels_output,
                                        self.config["image_size"],
                                        self.config["image_size"]) 
        
        #import tempmmap module and reset temp folder location
        from alphabase.io import tempmmap
        TEMP_DIR_NAME = tempmmap.redefine_temp_location(self.config["cache"])

        #generate container for single_cell_data
        _tmp_single_cell_data = tempmmap.array(shape = self.single_cell_data_shape, dtype = np.float16)
        _tmp_single_cell_index  = tempmmap.array(shape = self.single_cell_index_shape, dtype = np.int64)

        self.TEMP_DIR_NAME = TEMP_DIR_NAME

    def _transfer_tempmmap_to_hdf5(self):
        global _tmp_single_cell_data, _tmp_single_cell_index   
        
        self.log(f"number of cells too close to image edges to extract: {len(self.save_index_to_remove)}")
        _tmp_single_cell_data = np.delete(_tmp_single_cell_data, self.save_index_to_remove, axis=0)
        _tmp_single_cell_index = np.delete(_tmp_single_cell_index, self.save_index_to_remove, axis=0)
        _, cell_ids = _tmp_single_cell_index[:].T
        _tmp_single_cell_index[:] = list(zip(list(range(len(cell_ids))), cell_ids))

        self.log(f"Transferring extracted single cells to .hdf5")

        with h5py.File(self.output_path, 'w') as hf:
            hf.create_dataset('single_cell_index', data = _tmp_single_cell_index[:], dtype=np.int64) #increase to 64 bit otherwise information may become truncated
            self.log("index created.")
            hf.create_dataset('single_cell_data', data = _tmp_single_cell_data[:], 
                                                    chunks= (1,
                                                    1,
                                                    self.config["image_size"],
                                                    self.config["image_size"]),
                                            compression=self.compression_type,
                                            dtype=np.float32)  
        
        #delete tempobjects (to cleanup directory)
        self.log(f"Tempmmap Folder location {self.TEMP_DIR_NAME} will now be removed.")
        shutil.rmtree(self.TEMP_DIR_NAME, ignore_errors=True)

        del self.TEMP_DIR_NAME, _tmp_single_cell_data, _tmp_single_cell_index 
    
    def _write_cell(self, i, file, output_data, output_index):
        filetype = file.split(".")[-1]
        filename = file.split(".")[0]
  
        if filetype == "npy":
            class_id = int(filename)

            if i % 10000 == 0:
                self.log(f"Collecting dataset {i}")

            current_cell = os.path.join(self.extraction_cache, file)
            output_data[i] = np.load(current_cell)
            output_index[i] = np.array([i, class_id])
            index_addition = 1

        else:
            self.log(f"Non .npy file found {filename}, output hdf might contain missing values")
            index_addition = 0
        
        return(index_addition)

    def _get_label_info(self, arg):
        save_index, index = arg
        # no additional labelling required
        return(save_index, index, None, None)   

    def _save_cell_info(self, save_index, index, image_index, label_info, stack):
        #save index is irrelevant for this
        #label info is None so just ignore for the base case
        #image_index is none so jsut ignore for the base case
        # np.save(os.path.join(self.extraction_cache,"{}.npy".format(index)), stack)
        global _tmp_single_cell_data, _tmp_single_cell_index

        #save single cell images
        _tmp_single_cell_data[save_index] = stack
        _tmp_single_cell_index[save_index] = [save_index, index]

    def _extract_classes(self, input_segmentation_path, px_center, arg):
        """
        Processing for each invidual cell that needs to be run for each center.
        """
        save_index, index, image_index, label_info = self._get_label_info(arg) #label_info not used in base case but relevant for flexibility for other classes
    
        #generate some progress output every 10000 cells
        #relevant for benchmarking of time
        if save_index % 100 == 0:
            self.log("Extracting dataset {}".format(save_index))

        with h5py.File(input_segmentation_path, 'r', 
                       rdcc_nbytes=self.config["hdf5_rdcc_nbytes"], 
                       rdcc_w0=self.config["hdf5_rdcc_w0"],
                       rdcc_nslots=self.config["hdf5_rdcc_nslots"]) as input_hdf:
        
            hdf_channels = input_hdf.get(self.channel_label)
            hdf_labels = input_hdf.get(self.segmentation_label)
            
            width = self.config["image_size"]//2

            image_width = hdf_channels.shape[-2] #adaptive to ensure that even with multiple stacks of input images this works correctly
            image_height = hdf_channels.shape[-1]
            n_channels = hdf_channels.shape[-3]
            
            window_y = slice(px_center[0]-width,px_center[0]+width)
            window_x = slice(px_center[1]-width,px_center[1]+width)
            
            if width < px_center[0] and px_center[0] < image_width-width and width < px_center[1] and px_center[1] < image_height-width:
                # mask 0: nucleus mask
                if image_index is None:
                    nuclei_mask = hdf_labels[0, window_y, window_x]
                else:
                    nuclei_mask = hdf_labels[image_index, 0, window_y, window_x]

                nuclei_mask = np.where(nuclei_mask == index, 1, 0)
                nuclei_mask_extended = gaussian(nuclei_mask, preserve_range=True, sigma=5)
                nuclei_mask = gaussian(nuclei_mask, preserve_range=True, sigma=1)

                # channel 0: nucleus
                if image_index is None:
                    channel_nucleus = hdf_channels[0, window_y, window_x]
                else:
                    channel_nucleus = hdf_channels[image_index, 0, window_y, window_x]
                
                channel_nucleus = percentile_normalization(channel_nucleus, 0.001, 0.999)
                channel_nucleus = channel_nucleus * nuclei_mask_extended
                channel_nucleus = MinMax(channel_nucleus)

                if n_channels >= 2:
                    
                    # mask 1: cell mask
                    if image_index is None:
                        cell_mask = hdf_labels[1,window_y,window_x]
                    else:
                        cell_mask = hdf_labels[image_index, 1,window_y,window_x]

                    cell_mask = np.where(cell_mask == index,1,0).astype(int)
                    cell_mask = binary_fill_holes(cell_mask)

                    cell_mask_extended = dilation(cell_mask,footprint=disk(6))

                    cell_mask =  gaussian(cell_mask,preserve_range=True,sigma=1)   
                    cell_mask_extended = gaussian(cell_mask_extended,preserve_range=True,sigma=5)

                    # channel 3: cellmask
                    
                    if image_index is None:
                        channel_wga = hdf_channels[1, window_y, window_x]
                    else:
                        channel_wga = hdf_channels[image_index, 1,window_y,window_x]

                    channel_wga = percentile_normalization(channel_wga)
                    channel_wga = channel_wga*cell_mask_extended
                
                if n_channels == 1:
                    required_maps = [nuclei_mask, channel_nucleus]
                else:
                    required_maps = [nuclei_mask, cell_mask, channel_nucleus, channel_wga]
                
                #extract variable feature channels
                feature_channels = []

                if image_index is None:
                    if hdf_channels.shape[0] > 2:  
                        for i in range(2, hdf_channels.shape[0]):
                            feature_channel = hdf_channels[i, window_y, window_x]   
                            feature_channel = percentile_normalization(feature_channel)
                            feature_channel = feature_channel*cell_mask_extended
                            feature_channel = MinMax(feature_channel)
                            
                            feature_channels.append(feature_channel)
        
                else:
                    if hdf_channels.shape[1] > 2:
                        for i in range(2, hdf_channels.shape[1]):
                            feature_channel = hdf_channels[image_index, i, window_y, window_x]
                            feature_channel = percentile_normalization(feature_channel)
                            feature_channel = feature_channel*cell_mask_extended
                            feature_channel = MinMax(feature_channel)
                            
                            feature_channels.append(feature_channel)
                
                channels = required_maps + feature_channels
                stack = np.stack(channels, axis=0).astype("float16")
                
                if self.remap is not None:
                    stack = stack[self.remap]

                self._save_cell_info(save_index, index, image_index, label_info, stack) #to make more flexible for new datastructures with more labelling info
            else:
                self.save_index_to_remove.append(save_index)

    def process(self, input_segmentation_path, filtered_classes_path):
        # is called with the path to the segmented image
        
        self.get_channel_info() # needs to be called here after the segmentation is completed
        self.setup_output()
        self.parse_remapping()
        
        # setup cache
        self.uuid = str(uuid.uuid4())
        self.extraction_cache = os.path.join(self.config["cache"],self.uuid)
        if not os.path.isdir(self.extraction_cache):
            os.makedirs(self.extraction_cache)
            self.log("Created new extraction cache " + self.extraction_cache)
            
        self.log("Started extraction")
        self.log("Loading segmentation data from {input_segmentation_path}")
        
        hf = h5py.File(input_segmentation_path, 'r')
        hdf_channels = hf.get(self.channel_label)
        hdf_labels = hf.get(self.segmentation_label)

        self.log(f"Using channel label {hdf_channels}")
        self.log(f"Using segmentation label {hdf_labels}")
        self.log("Finished loading channel data " + str(hdf_channels.shape))
        self.log("Finished loading label data " + str(hdf_labels.shape))

        # Calculate centers
        self.log("Checked class coordinates")
        
        center_path = os.path.join(self.directory, "center.pickle")
        cell_ids_path = os.path.join(self.directory, "_cell_ids.pickle")

        if os.path.isfile(center_path) and os.path.isfile(cell_ids_path) and not self.overwrite:
            self.log("Cached version found, loading")
            with open(center_path, "rb") as input_file:
                center_nuclei = cPickle.load(input_file)
                px_centers = np.round(center_nuclei).astype(int)
            with open(cell_ids_path, "rb") as input_file:
                _cell_ids = cPickle.load(input_file)
        else:
            self.log("Started class coordinate calculation")
            center_nuclei, length, _cell_ids = numba_mask_centroid(hdf_labels[0], debug=self.debug)
            px_centers = np.round(center_nuclei).astype(int)
            self.log("Finished class coordinate calculation")
            with open(center_path, "wb") as output_file:
                cPickle.dump(center_nuclei, output_file)
            with open(cell_ids_path, "wb") as output_file:
                cPickle.dump(_cell_ids, output_file)
            with open(os.path.join(self.directory,"length.pickle"), "wb") as output_file:
                cPickle.dump(length, output_file)
                
            del length

        class_list = self.get_classes(filtered_classes_path)
        lookup_saveindex = self.generate_save_index_lookup(class_list)           
        
        #make into set to improve computational efficiency
        #needs to come after generating lookup index otherwise it will throw an error message
        class_list = set(class_list)
        
        #filter cell ids found using center into those that we actually want to extract
        _cell_ids = list(_cell_ids)
        filter = [x in class_list for x in _cell_ids]

        px_centers = np.array(list(compress(px_centers, filter)))
        _cell_ids = list(compress(_cell_ids, filter))

        # setup cache
        self._initialize_tempmmap_array()
        
        #start extraction
        self.verbalise_extraction_info()

        self.log(f"Starting extraction of {self.num_classes} classes")
        start = timeit.default_timer()

        # f = partial(self._extract_classes, input_segmentation_path)
        # args = list(zip([lookup_saveindex.index.get_loc(x) for x in _cell_ids], _cell_ids))

        # with Pool(processes = self.config["threads"]) as pool:
        #     x = list(tqdm(pool.imap(f, px_centers, args), length = len(args)))
        #     pool.close()
        #     pool.join()
        #     print("multiprocessing done.")
        
        for px_center, cell_id in tqdm(zip(px_centers, _cell_ids), desc = "extracting classes"):
            save_index = lookup_saveindex.index.get_loc(cell_id)
            self._extract_classes(input_segmentation_path, px_center,  (save_index, cell_id))

        stop = timeit.default_timer()

        #calculate duration
        duration = stop - start
        rate = self.num_classes/duration

        #generate final log entries
        self.log(f"Finished extraction in {duration:.2f} seconds ({rate:.2f} cells / second)")
        self.log("Collect cells")

        #make into set to improve computational efficiency
        #transfer results to hdf5
        self._transfer_tempmmap_to_hdf5()
        self.log("Finished cleaning up cache")

class TimecourseHDF5CellExtraction(HDF5CellExtraction):
    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_DATA_FILE = "single_cells.h5"
    DEFAULT_SEGMENTATION_DIR = "segmentation"
    DEFAULT_SEGMENTATION_FILE = "input_segmentation.h5"

    DEFAULT_DATA_DIR = "data"
    CLEAN_LOG = False
    
    #new parameters to make workflow adaptable to other types of projects
    channel_label = "input_images"
    segmentation_label = "segmentation"

    def __init__(self, 
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

    def get_labelling(self):
        with h5py.File(self.input_segmentation_path, 'r') as hf:
            self.label_names = hf.get("label_names")[:]
            self.n_labels = len(self.label_names)

    def _get_arg(self, class_list):
        #need to extract ID for each cellnumber
        #generate lookuptable where we have all cellids for each tile id
        with h5py.File(self.input_segmentation_path, "r") as hf:
            labels = hf.get("labels").asstr()[:]
            classes = hf.get("classes")

            results = pd.DataFrame(columns = ["tileids", "cellids"], index = range(labels.shape[0]))
        
            self.log({"Extracting classes from each Segmentation Tile."})
            #should be updated later when classes saved in segmentation automatically 
            # currently not working because of issue with datatypes
            
            for i, tile_id in zip(labels.T[0], labels.T[1]):
                #dirty fix for some strange problem with some of the datasets
                #FIX THIS
                if i == "":
                    continue
                cellids =list(classes[int(i)])
                if 0 in cellids:
                    cellids.remove(0)
                results.loc[int(i), "cellids"] = cellids
                results.loc[int(i), "tileids"] = tile_id

        #map each cell id to tile id and generate a tuple which can be passed to later functions
        return_results = [[(xset, i, results.loc[i, "tileids"]) for i, xset in enumerate(results.cellids)]]
        return_results = flatten(return_results)
        #required output format: [(class_ids, image_index1, label_id), (class_ids, image_index2, label_id)]
        return(return_results)

    def _get_label_info(self, arg):
        save_index, index, image_index, label_info = arg
        return(save_index, index, image_index, label_info)  

    def _initialize_tempmmap_array(self):
        #define as global variables so that this is also avaialable in other functions
        global _tmp_single_cell_data, _tmp_single_cell_index
        
        #import tempmmap module and reset temp folder location
        from alphabase.io import tempmmap
        TEMP_DIR_NAME = tempmmap.redefine_temp_location(self.config["cache"])

        #generate datacontainer for the single cell images
        column_labels = ['index', "cellid"] + list(self.label_names.astype("U13"))[1:]
        self.single_cell_index_shape =  (self.num_classes, len(column_labels))
        self.single_cell_data_shape = (self.num_classes,
                                                    self.n_channels_output,
                                                    self.config["image_size"],
                                                    self.config["image_size"])

        #generate container for single_cell_data
        print(self.single_cell_data_shape)
        _tmp_single_cell_data = tempmmap.array(self.single_cell_data_shape, dtype = np.float16)

        #generate container for single_cell_index
        #cannot be a temmmap array with object type as this doesnt work for memory mapped arrays
        #dt = h5py.special_dtype(vlen=str)
        _tmp_single_cell_index  = np.empty(self.single_cell_index_shape, dtype = "<U64") #need to use U64 here otherwise information potentially becomes truncated
        
        #_tmp_single_cell_index  = tempmmap.array(self.single_cell_index_shape, dtype = "<U32")

        self.TEMP_DIR_NAME = TEMP_DIR_NAME

    def _transfer_tempmmap_to_hdf5(self):
        global _tmp_single_cell_data, _tmp_single_cell_index   

        self.log(f"number of cells too close to image edges to extract: {len(self.save_index_to_remove)}")
        _tmp_single_cell_data = np.delete(_tmp_single_cell_data, self.save_index_to_remove, axis=0)
        _tmp_single_cell_index = np.delete(_tmp_single_cell_index, self.save_index_to_remove, axis=0)

        #extract information about the annotation of cell ids
        column_labels = ['index', "cellid"] + list(self.label_names.astype("U13"))[1:]
        
        self.log("Creating HDF5 file to save results to.")
        with h5py.File(self.output_path, 'w') as hf:
            #create special datatype for storing strings
            dt = h5py.special_dtype(vlen=str)

            #save label names so that you can always look up the column labelling
            hf.create_dataset('label_names', data = column_labels, chunks=None, dtype = dt)
            
            #generate index data container
            hf.create_dataset('single_cell_index_labelled', _tmp_single_cell_index.shape , chunks=None, dtype = dt)
            single_cell_labelled = hf.get("single_cell_index_labelled")
            single_cell_labelled[:] = _tmp_single_cell_index[:]

            hf.create_dataset('single_cell_index', (_tmp_single_cell_index.shape[0], 2), dtype="uint64")           

            hf.create_dataset('single_cell_data',data =  _tmp_single_cell_data,
                                                chunks=(1,
                                                        1,
                                                        self.config["image_size"],
                                                        self.config["image_size"]),
                                                compression=self.compression_type,
                                                dtype="float16")
            
        self.log(f"Transferring exracted single cells to .hdf5")
        with h5py.File(self.output_path, 'a') as hf:
            #need to save this index seperately since otherwise we get issues with the classificaiton of the extracted cells
            index = _tmp_single_cell_index[:, 0:2]
            _, cell_ids = index.T
            index = np.array(list(zip(range(len(cell_ids)), cell_ids)))
            index[index == ""] = "0" 
            index = index.astype("uint64")
            hf["single_cell_index"][:] = index

        #delete tempobjects (to cleanup directory)
        self.log(f"Tempmmap Folder location {self.TEMP_DIR_NAME} will now be removed.")
        shutil.rmtree(self.TEMP_DIR_NAME, ignore_errors=True)

        del _tmp_single_cell_data, _tmp_single_cell_index, self.TEMP_DIR_NAME 

    def _save_cell_info(self, index, cell_id, image_index, label_info, stack):
        global _tmp_single_cell_data, _tmp_single_cell_index
        #label info is None so just ignore for the base case
        
        #save single cell images
        _tmp_single_cell_data[index] = stack
        # print("index:", index)
        # import matplotlib.pyplot as plt
        
        # for i in stack:
        #         plt.figure()
        #         plt.imshow(i)
        #         plt.show()

        #get label information
        with h5py.File(self.input_segmentation_path, "r") as hf:
            labelling = hf.get("labels").asstr()[image_index][1:]
            save_value = [str(index), str(cell_id)]
            save_value = np.array(flatten([save_value, labelling]))

            _tmp_single_cell_index[index] = save_value

            #double check that its really the same values
            if _tmp_single_cell_index[index][2] != label_info:
                self.log("ISSUE INDEXES DO NOT MATCH.")
                self.log(f"index: {index}")
                self.log(f"image_index: {image_index}")
                self.log(f"label_info: {label_info}")
                self.log(f"index it should be: {_tmp_single_cell_index[index][2]}")
    
    def process(self, input_segmentation_path, filtered_classes_path):
    # is called with the path to the segmented image
        
        self.get_labelling()
        self.get_channel_info()
        self.setup_output()
        self.parse_remapping()

        complete_class_list = self.get_classes(filtered_classes_path)
        arg_list = self._get_arg(complete_class_list)
        lookup_saveindex = self.generate_save_index_lookup(complete_class_list)

        # setup cache
        self._initialize_tempmmap_array()

        #start extraction
        self.log("Starting extraction.")
        self.verbalise_extraction_info()

        with  h5py.File(self.input_segmentation_path, 'r') as hf:
            start = timeit.default_timer()

            self.log(f"Loading segmentation data from {self.input_segmentation_path}")
            hdf_labels = hf.get(self.segmentation_label)

            for arg in tqdm(arg_list):
                cell_ids, image_index, label_info = arg 
                # print("image index:", image_index)
                # print("cell ids", cell_ids)
                # print("label info:", label_info)
                
                input_image = hdf_labels[image_index, 0, :, :]

                #check if image is an empty array
                if np.all(input_image==0):
                    self.log(f"Image with the image_index {image_index} only contains zeros. Skipping this image.")
                    print(f"Error: image with the index {image_index} only contains zeros!! Skipping this image.")
                    continue
                else:
                    center_nuclei, _, _cell_ids = numba_mask_centroid(input_image, debug=self.debug)

                    if center_nuclei is not None:
                        px_centers = np.round(center_nuclei).astype(int)
                        _cell_ids = list(_cell_ids)

                        # #plotting results for debugging
                        # import matplotlib.pyplot as plt
                        # plt.figure(figsize = (10, 10))
                        # plt.imshow(hdf_labels[image_index, 1, :, :])
                        # plt.figure(figsize = (10, 10))
                        # plt.imshow(hdf_labels[image_index, 0, :, :])
                        # y, x = px_centers.T
                        # plt.scatter(x, y, color = "red", s = 5)
                        
                        #filter lists to only include those cells which passed the final filters (i.e remove border cells)
                        filter = [x in cell_ids for x in _cell_ids]
                        px_centers = np.array(list(compress(px_centers, filter)))
                        _cell_ids = list(compress(_cell_ids, filter))

                        # #plotting results for debugging
                        # y, x = px_centers.T
                        # plt.scatter(x, y, color = "blue", s = 5)
                        # plt.show()

                        for cell_id, px_center in zip(_cell_ids, px_centers):
                            save_index = lookup_saveindex.index.get_loc(cell_id)
                            self._extract_classes(input_segmentation_path, px_center,  (save_index, cell_id, image_index, label_info))
                    else:
                        self.log(f"Image with the image_index {image_index} doesn't contain any cells. Skipping this image.")
                        print(f"Error: image with the index {image_index} doesn't contain any cells!! Skipping this image.")
                        continue

            stop = timeit.default_timer()

        duration = stop - start
        rate = self.num_classes/duration
        self.log(f"Finished parallel extraction in {duration:.2f} seconds ({rate:.2f} cells / second)")
        
        self.log("Collect cells")
        self._transfer_tempmmap_to_hdf5()
        self.log("Extraction completed.")

class SingleCellExtraction:

    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_DATA_DIR = "data"
    CLEAN_LOG = False
    
    def __init__(self, 
                 config, 
                 folder_path, 
                 debug = False, 
                 overwrite = False,
                 intermediate_output = False):

        """class can be used to create a DAPI - WGA extraction workfow
        
        Args:

            config (dict): Config file which is passed by the Project class when called. Is loaded from the project based on the name of the class.
        
            directory (str): Directory which should be used by the processing step. A subdirectory of the project directory is passed by the project class when called. The directory will be newly created if it does not exist yet.
            
            intermediate_output (bool, optional, default ``False``): When set to True intermediate outputs will be saved where applicable.
                
            debug (bool, optional, default ``False``): When set to True debug outputs will be printed where applicable. 
                
            overwrite (bool, optional, default ``True``): When set to True, the processing step directory will be delted and newly created when called.
            
        """
        self.debug = debug
        self.overwrite = overwrite
        self.config = config
        self.intermediate_output = intermediate_output
        
        
        
        # Create segmentation directory
        self.extraction_directory = folder_path
        if not os.path.isdir(self.extraction_directory):
            os.makedirs(self.extraction_directory)
            self.log("Created new directory " + self.extraction_directory)
        
        self.extraction_data_directory = os.path.join(folder_path, self.DEFAULT_DATA_DIR)
        if not os.path.isdir(self.extraction_data_directory):
            
            os.makedirs(self.extraction_data_directory)
            self.log("Created new data directory " + self.extraction_data_directory)
            
        # Set up log and clean old log
        if self.CLEAN_LOG:
            log_path = os.path.join(self.extraction_directory, self.DEFAULT_LOG_NAME)
            if os.path.isfile(log_path):
                os.remove(log_path)
        
    def get_output_path(self):
        return self.extraction_data_directory            
                
    def get_timestamp(self):
        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")  
        return "[" + dt_string + "] "
    
    def log(self, message):
        
        log_path = os.path.join(self.extraction_directory, self.DEFAULT_LOG_NAME)
        
        with open(log_path, "a") as myfile:
            myfile.write(self.get_timestamp() + message +" \n")
        
    def __call__(self, input_segmentation_path, filtered_classes_path):
        # is called with the path to the segmented image
        
        self.log("Started extraction")
        self.log("Loading segmentation data")
        
        hf = h5py.File(input_segmentation_path, 'r')
        hdf_channels = hf.get('channels')
        hdf_labels = hf.get('labels')
        
        self.log("Finished loading channel data " + str(hdf_channels.shape))
        self.log("Finished loading label data " + str(hdf_labels.shape))
        
        self.log("Loading filtered classes")
        cr = csv.reader(open(filtered_classes_path,'r'))
        filtered_classes = [int(el[0]) for el in list(cr)]
        self.log("Loaded {} filtered classes".format(len(filtered_classes)))
        
        
        # Calculate centers
        self.log("Checked class coordinates")
        
        center_path = os.path.join(self.extraction_directory,"center.pickle")
        
        if os.path.isfile(center_path):
            
            self.log("Cached version found, loading")
            with open(center_path, "rb") as input_file:
                center_nuclei = cPickle.load(input_file)
                
                
        else:
            self.log("Started class coordinate calculation")
            center_nuclei, length, coords = numba_mask_centroid(hdf_labels[0], debug=self.debug)
            

            with open(center_path, "wb") as output_file:
                cPickle.dump(center_nuclei, output_file)

            with open(os.path.join(self.extraction_directory,"length.pickle"), "wb") as output_file:
                cPickle.dump(length, output_file)

            with open(os.path.join(self.extraction_directory,"coords.pickle"), "wb") as output_file:
                cPickle.dump(coords, output_file)
                
            del length
            del coords
        
        
        px_center = np.round(center_nuclei).astype(int)
        
        #return
        
        self.log("Started parallel extraction")
        f = partial(self.extract_classes, px_center, input_segmentation_path)

        with Pool(processes=self.config["threads"]) as pool:
            x = pool.map(f,filtered_classes)
            
        self.log("Finished parallel extraction")
        
        
    def extract_classes(self, center_list, input_segmentation_path, arg):
        
        if arg % 10000 == 0:
            self.log("Saved dataset {}".format(arg))
        
        if arg == 0:
            return

        index = arg-1
        px_center = center_list[index]
        
        hf = h5py.File(input_segmentation_path, 'r')
        hdf_channels = hf.get('channels')
        hdf_labels = hf.get('labels')

        #channel labelling scheme that needs to be kept
        #dapi = channels[0]
        #cyotosl = channels[1]
        #additional_channels = channels[2:]
        #labels_nuclei = segmentation[0]
        #labels_cell = segmentation[1]

        image_width = hdf_channels.shape[1]
        image_height = hdf_channels.shape[2]
        n_channels = hdf_channels.shape[0]

        width = self.config["image_size"]//2
        
        window_y = slice(px_center[0]-width,px_center[0]+width)
        window_x = slice(px_center[1]-width,px_center[1]+width)
        
        channels = []
        if width < px_center[0] and px_center[0] < image_width-width and width < px_center[1] and px_center[1] < image_height-width:
            
            
            # mask 1: nucleus mask
            nuclei_mask = hdf_labels[0,window_y,window_x]
            
            nuclei_mask = np.where(nuclei_mask == index+1, 1,0)

            nuclei_mask_extended = gaussian(nuclei_mask,preserve_range=True,sigma=5)
            nuclei_mask = gaussian(nuclei_mask,preserve_range=True,sigma=1)

            
            if self.debug:
                plot_image(nuclei_mask, size=(5,5), save_name=os.path.join(self.extraction_data_directory,"{}_0".format(index)), cmap="Greys") 
            channels.append(nuclei_mask)

            # mask 2: cell mask
            if n_channels >= 2:
                cell_mask = hdf_labels[1,window_y,window_x]
                cell_mask = np.where(cell_mask == index+1,1,0).astype(int)
                cell_mask = binary_fill_holes(cell_mask)

                cell_mask_extended = dilation(cell_mask,footprint=disk(6))

                cell_mask =  gaussian(cell_mask,preserve_range=True,sigma=1)   
                cell_mask_extended = gaussian(cell_mask_extended,preserve_range=True,sigma=5)

                if self.debug:
                    plot_image(cell_mask, size=(5,5), save_name=os.path.join(self.extraction_data_directory,"{}_1".format(index)), cmap="Greys") 
                
                channels.append(cell_mask)

            # channel 1: nucleus
            channel_nucleus = hdf_channels[0,window_y,window_x]
            channel_nucleus = percentile_normalization(channel_nucleus)
            channel_nucleus = channel_nucleus *nuclei_mask_extended

            channel_nucleus = MinMax(channel_nucleus)

            if self.debug:
                plot_image(channel_nucleus, size=(5,5), save_name=os.path.join(self.extraction_data_directory,"{}_2".format(index)))

            channels.append(channel_nucleus)

            # channl 2: cytosol marker
            if n_channels >= 2:
                channel_cytosol= hdf_channels[1,window_y,window_x]
                channel_cytosol = percentile_normalization(channel_cytosol)
                channel_cytosol = channel_cytosol*cell_mask_extended

                channel_cytosol = MinMax(channel_cytosol)

                if self.debug:
                    plot_image(channel_cytosol, size=(5,5), save_name=os.path.join(self.extraction_data_directory,"{}_3".format(index)))

                channels.append(channel_cytosol)

            # channel 3+: marker channels
            if n_channels >= 3:
                for id in range(2, n_channels):
                    channel_additional = hdf_channels[id, window_y, window_x]
                    channel_additional = percentile_normalization(channel_additional)
                    channel_additional = channel_additional*cell_mask_extended

                    if self.debug:
                        plot_image(channel_additional, size=(5,5), save_name=os.path.join(self.extraction_data_directory,"{}_4".format(index)))

                    channels.append(channel_additional)
            
                    design = np.stack(channels, axis=-1).astype("float16")
            
            np.save(os.path.join(self.extraction_data_directory,"{}.npy".format(index)),design)
            
class HDF5CellExtractionOld:

    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_DATA_FILE = "single_cells.h5"
    DEFAULT_DATA_DIR = "data"
    CLEAN_LOG = False
    
    def __init__(self, 
                 config, 
                 folder_path, 
                 debug=False, 
                 overwrite=False,
                 intermediate_output = True):
        
        """class can be initiated to create a WGA extraction workfow

        :param config: Configuration for the extraction passed over from the :class:`pipeline.Dataset`
        :type config: dict

        :param string: Directiory for the extraction log and results. is created if not existing yet
        :type config: string
        
        :param debug: Flag used to output debug information and map images
        :type debug: bool, default False
        
        :param overwrite: Flag used to recalculate all images, not yet implemented
        :type overwrite: bool, default False
        """
        self.debug = debug
        self.overwrite = overwrite
        self.config = config
        self.intermediate_output = intermediate_output
        
        # Set up log and clean old log
        if self.CLEAN_LOG:
            log_path = os.path.join(self.extraction_directory, self.DEFAULT_LOG_NAME)
            if os.path.isfile(log_path):
                os.remove(log_path)
        
        # Create segmentation directory
        self.extraction_directory = folder_path
        if not os.path.isdir(self.extraction_directory):
            
            os.makedirs(self.extraction_directory)
            self.log("Created new directory " + self.extraction_directory)
            
        self.extraction_data_directory = os.path.join(folder_path, self.DEFAULT_DATA_DIR)
        if not os.path.isdir(self.extraction_data_directory):
            os.makedirs(self.extraction_data_directory)
            self.log("Created new data directory " + self.extraction_data_directory)
            
        
        
    def get_output_path(self):
        return self.extraction_data_directory            
                
    def get_timestamp(self):
        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")  
        return "[" + dt_string + "] "
    
    def log(self, message):
        
        log_path = os.path.join(self.extraction_directory, self.DEFAULT_LOG_NAME)
        
        with open(log_path, "a") as myfile:
            myfile.write(self.get_timestamp() + message +" \n")
            
        if self.debug:
            print(self.get_timestamp() + message)
        
    def __call__(self, input_segmentation_path, filtered_classes_path):
        # is called with the path to the segmented image
        
        # setup cache
        self.uuid = str(uuid.uuid4())
        self.extraction_cache = os.path.join(self.config["cache"],self.uuid)
        if not os.path.isdir(self.extraction_cache):
            os.makedirs(self.extraction_cache)
            self.log("Created new extraction cache " + self.extraction_cache)
            
        self.log("Started extraction")
        self.log("Loading segmentation data")
        
        hf = h5py.File(input_segmentation_path, 'r')
        hdf_channels = hf.get('channels')
        hdf_labels = hf.get('labels')
        
        
        self.log("Finished loading channel data " + str(hdf_channels.shape))
        self.log("Finished loading label data " + str(hdf_labels.shape))
        
        self.log("Loading filtered classes")
        cr = csv.reader(open(filtered_classes_path,'r'))
        filtered_classes = [int(el[0]) for el in list(cr)]
        self.log("Loaded {} filtered classes".format(len(filtered_classes)))
        
        
        # Calculate centers
        self.log("Checked class coordinates")
        
        center_path = os.path.join(self.extraction_directory,"center.pickle")
        if os.path.isfile(center_path) and not self.overwrite:
            
            self.log("Cached version found, loading")
            with open(center_path, "rb") as input_file:
                center_nuclei = cPickle.load(input_file)
                px_center = np.round(center_nuclei).astype(int)
                
        else:
            self.log("Started class coordinate calculation")
            center_nuclei, length = numba_mask_centroid(hdf_labels[0], debug=self.debug)
            px_center = np.round(center_nuclei).astype(int)
            self.log("Finished class coordinate calculation")
            with open(center_path, "wb") as output_file:
                cPickle.dump(center_nuclei, output_file)

            with open(os.path.join(self.extraction_directory,"length.pickle"), "wb") as output_file:
                cPickle.dump(length, output_file)
                
            del length
        
        # parallel execution
        
        class_list = list(filtered_classes)
        # Zero class contains background
        
        if 0 in class_list: class_list.remove(0)
            
        num_classes = len(class_list)
        
        num_channels = 5
        
        self.log(f"Started parallel extraction of {num_classes} classes")
        start = timeit.default_timer()
        
        f = partial(self._extract_classes, px_center, input_segmentation_path)

        with Pool(processes=self.config["threads"]) as pool:
            x = pool.map(f,class_list)
            
        
        stop = timeit.default_timer()
        duration = stop - start
        rate = num_classes/duration
        self.log(f"Finished parallel extraction in {duration:.2f} seconds ({rate:.2f} cells / second)")
        self.log("Collect cells")
        # collect cells
        
        # create empty hdf5
        current_level_files = [ name for name in os.listdir(self.extraction_cache) if os.path.isfile(os.path.join(self.extraction_cache, name))]
        num_classes = len(current_level_files)
        
        output_path = os.path.join(self.extraction_data_directory, self.DEFAULT_DATA_FILE)
        hf = h5py.File(output_path, 'w')
        
        hf.create_dataset('single_cell_index', (num_classes,2) ,dtype="uint32")
        hf.create_dataset('single_cell_data', (num_classes,
                                                   num_channels,
                                                   self.config["image_size"],
                                                   self.config["image_size"]),
                                               chunks=(1,
                                                       1,
                                                       self.config["image_size"],
                                                       self.config["image_size"]),
                                               compression=self.compression_type,
                                               dtype="float16")
        output_index = hf.get('single_cell_index')
        output_data = hf.get('single_cell_data')
        
        i = 0
        for file in current_level_files:
            
            filetype = file.split(".")[-1]
            filename = file.split(".")[0]
  
            if filetype == "npy":

                
                class_id = int(filename)

                if i % 10000 == 0:
                    self.log(f"Collecting dataset {i}")


                current_cell = os.path.join(self.extraction_cache, file)
                output_data[i] = np.load(current_cell)
                output_index[i] = np.array([i,class_id])
                i+=1 

                
                    
            else:
                self.log(f"Non .npy file found {filename}, output hdf might contain missing values")
                        
        
        hf.close()
        self.log("Finished collection")
        
        # remove cache
        
        
        self.log("Cleaning up cache")
        shutil.rmtree(self.extraction_cache)
        self.log("Finished cleaning up cache")
        
        
    def _extract_classes(self, center_list, input_segmentation_path, arg):
        if arg % 10000 == 0:
            self.log("Extracting dataset {}".format(arg))
            
        index = arg
        
        px_center = center_list[index]
        
        input_hdf = h5py.File(input_segmentation_path, 'r', 
                       rdcc_nbytes=self.config["hdf5_rdcc_nbytes"], 
                       rdcc_w0=self.config["hdf5_rdcc_w0"],
                       rdcc_nslots=self.config["hdf5_rdcc_nslots"])
        
        hdf_channels = input_hdf.get('channels')
        hdf_labels = input_hdf.get('labels')
        
        width = self.config["image_size"]//2
        
        image_width = hdf_channels.shape[1]
        image_height = hdf_channels.shape[2]
        
        window_y = slice(px_center[0]-width,px_center[0]+width)
        window_x = slice(px_center[1]-width,px_center[1]+width)
        
        # check for boundaries
        if width < px_center[0] and px_center[0] < image_width-width and width < px_center[1] and px_center[1] < image_height-width:
            # channel 0: nucleus mask
            nuclei_mask = hdf_labels[0,window_y,window_x]
            
            nuclei_mask = np.where(nuclei_mask == index, 1,0)

            nuclei_mask_extended = gaussian(nuclei_mask,preserve_range=True,sigma=5)
            nuclei_mask = gaussian(nuclei_mask,preserve_range=True,sigma=1)

            """
            if self.debug:
                plot_image(nuclei_mask, size=(5,5), save_name=os.path.join(self.extraction_directory,"{}_0".format(index)), cmap="Greys") 
            """
            # channel 1: cell mask
            
            cell_mask = hdf_labels[1,window_y,window_x]
            cell_mask = np.where(cell_mask == index,1,0).astype(int)
            cell_mask = binary_fill_holes(cell_mask)

            cell_mask_extended = dilation(cell_mask,footprint=disk(6))

            cell_mask =  gaussian(cell_mask,preserve_range=True,sigma=1)   
            cell_mask_extended = gaussian(cell_mask_extended,preserve_range=True,sigma=5)

            """
            if self.debug:
                plot_image(cell_mask, size=(5,5), save_name=os.path.join(self.extraction_directory,"{}_1".format(index)), cmap="Greys") 
            """
            
            # channel 2: nucleus
            channel_nucleus = hdf_channels[0,window_y,window_x]
            channel_nucleus = percentile_normalization(channel_nucleus)
            channel_nucleus = channel_nucleus *nuclei_mask_extended

            channel_nucleus = MinMax(channel_nucleus)
            
            """
            if self.debug:
                plot_image(channel_nucleus, size=(5,5), save_name=os.path.join(self.extraction_directory,"{}_2".format(index)))
            """

            # channel 3: golgi
            channel_golgi = hdf_channels[1,window_y,window_x]
            channel_golgi = percentile_normalization(channel_golgi)
            channel_golgi = channel_golgi*cell_mask_extended

            channel_golgi = MinMax(channel_golgi)
            
            """
            if self.debug:
                plot_image(channel_golgi, size=(5,5), save_name=os.path.join(self.extraction_directory,"{}_3".format(index)))
            """
            
            # channel 4: cellmask
            channel_wga = hdf_channels[2,window_y,window_x]
            channel_wga = percentile_normalization(channel_wga)
            channel_wga = channel_wga*cell_mask_extended
            
            """
            if self.debug:
                plot_image(channel_wga, size=(5,5), save_name=os.path.join(self.extraction_directory,"{}_4".format(index)))
            """
            
            stack = np.stack([nuclei_mask,cell_mask,channel_nucleus,channel_golgi,channel_wga], axis=0).astype("float16")
            np.save(os.path.join(self.extraction_cache,"{}.npy".format(index)),stack)
            
        input_hdf.close()      
