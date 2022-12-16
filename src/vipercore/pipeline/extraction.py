from datetime import datetime
from operator import index
import os
import numpy as np
import pandas as pd
import csv
from functools import partial
from multiprocessing import Pool
import h5py

from skimage.filters import gaussian
from skimage.morphology import disk, dilation

from scipy.ndimage import binary_fill_holes

from vipercore.processing.segmentation import numba_mask_centroid
from vipercore.processing.utils import plot_image, flatten
from vipercore.processing.deprecated import normalize, MinMax
from vipercore.pipeline.base import ProcessingStep

import uuid
import shutil
import timeit

import _pickle as cPickle

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
        self.get_channel_info()
        
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
        self.log("Loading filtered classes")
        cr = csv.reader(open(filtered_classes_path,'r'))
        filtered_classes = [int(el[0]) for el in list(cr)]

        self.log("Loaded {} filtered classes".format(len(filtered_classes)))
        filtered_classes = np.unique(filtered_classes) #make sure they are all unique
        self.log("After removing duplicates {} filtered classes remain.".format(len(filtered_classes)))

        class_list = list(filtered_classes)
        if 0 in class_list: class_list.remove(0)
        self.num_classes = len(class_list)

        return(class_list)
    
    def verbalise_extraction_info(self):
        #print some output information
        self.log(f"Extraction Details:")
        self.log(f"--------------------------------")
        self.log(f"Input channels: {self.n_channels_input}")
        self.log(f"Input labels: {self.n_segmentation_channels}")
        self.log(f"Output channels: {self.n_channels_output}")
        self.log(f"Number of classes to extract: {self.num_classes}")
    
    def _get_arg(class_list, hf):
        #no modification required in base version so do not need to use hf
        return class_list

    def _create_hdf5(self):

        hf = h5py.File(self.output_path, 'w')
        hf.create_dataset('single_cell_index', (self.num_classes,2) ,dtype="uint32")
        hf.create_dataset('single_cell_data', (self.num_classes,
                                                   self.n_channels,
                                                   self.config["image_size"],
                                                   self.config["image_size"]),
                                               chunks=(1,
                                                       1,
                                                       self.config["image_size"],
                                                       self.config["image_size"]),
                                               compression=self.compression_type,
                                               dtype="float16")   
        
        #return result so that we can continue working with it
        return(hf)
    
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
        index = arg
        # no additional labelling required
        return(None, index, None, None)   

    def _save_cell_info(self, save_index, index, image_index, label_info, stack):
        #save index is irrelevant for this
        #label info is None so just ignore for the base case
        #image_index is none so jsut ignore for the base case
        np.save(os.path.join(self.extraction_cache,"{}.npy".format(index)), stack)

    def _extract_classes(self, input_segmentation_path, px_center, arg):
        """
        Processing for each invidual cell that needs to be run for each center.
        """
        print(arg)
        save_index, index, image_index, label_info = self._get_label_info(arg) #label_info not used in base case but relevant for flexibility for other classes
        
        #generate some progress output every 10000 cells
        #relevant for benchmarking of time
        if index % 10000 == 0:
            self.log("Extracting dataset {}".format(index))

        with h5py.File(input_segmentation_path, 'r', 
                       rdcc_nbytes=self.config["hdf5_rdcc_nbytes"], 
                       rdcc_w0=self.config["hdf5_rdcc_w0"],
                       rdcc_nslots=self.config["hdf5_rdcc_nslots"]) as input_hdf:
        
            hdf_channels = input_hdf.get(self.channel_label)
            hdf_labels = input_hdf.get(self.segmentation_label)
            
            width = self.config["image_size"]//2

            image_width = hdf_channels.shape[-2] #adaptive to ensure that even with multiple stacks of input images this works correctly
            image_height = hdf_channels.shape[-1]
            
            window_y = slice(px_center[0]-width,px_center[0]+width)
            window_x = slice(px_center[1]-width,px_center[1]+width)
        
            # check for boundaries
            if width < px_center[0] and px_center[0] < image_width-width and width < px_center[1] and px_center[1] < image_height-width:
                
                # channel 0: nucleus mask
                if image_index is None:
                    nuclei_mask = hdf_labels[0, window_y, window_x]
                else:
                    nuclei_mask = hdf_labels[image_index, 0,window_y, window_x]
                #print(nuclei_mask[60:68,60:68])
                
                nuclei_mask = np.where(nuclei_mask == index, 1,0)
                nuclei_mask_extended = gaussian(nuclei_mask,preserve_range=True,sigma=5)
                nuclei_mask = gaussian(nuclei_mask,preserve_range=True,sigma=1)

                #if self.debug:
                #    plot_image(nuclei_mask, size=(5,5)), save_name=os.path.join(self.directory,"{}_0".format(index)), cmap="Greys") 
                
                # channel 1: cell mask
                if image_index is None:
                    cell_mask = hdf_labels[1,window_y,window_x]
                else:
                    cell_mask = hdf_labels[image_index, 1,window_y,window_x]

                #print(cell_mask[60:68,60:68])
                plot_image(cell_mask, size=(5,5))
                cell_mask = np.where(cell_mask == index,1,0).astype(int)
                cell_mask = binary_fill_holes(cell_mask)

                cell_mask_extended = dilation(cell_mask,footprint=disk(6))

                cell_mask =  gaussian(cell_mask,preserve_range=True,sigma=1)   
                cell_mask_extended = gaussian(cell_mask_extended,preserve_range=True,sigma=5)

                #if self.debug:
                #    plot_image(cell_mask, size=(5,5), save_name=os.path.join(self.directory,"{}_1".format(index)), cmap="Greys") 
                
                # channel 2: nucleus
                if image_index is None:
                    channel_nucleus = hdf_channels[0, window_y, window_x]
                else:
                    channel_nucleus = hdf_channels[image_index, 0, window_y, window_x]
                
                channel_nucleus = normalize(channel_nucleus)
                channel_nucleus = channel_nucleus * nuclei_mask_extended
                channel_nucleus = MinMax(channel_nucleus)
            
                #if self.debug:
                #    plot_image(channel_nucleus, size=(5,5), save_name=os.path.join(self.directory,"{}_2".format(index)))
                
                # channel 3: cellmask
                
                if image_index is None:
                    channel_wga = hdf_channels[1, window_y, window_x]
                else:
                    channel_wga = hdf_channels[image_index, 1,window_y,window_x]

                channel_wga = normalize(channel_wga)
                channel_wga = channel_wga*cell_mask_extended
                
                required_maps = [nuclei_mask, cell_mask, channel_nucleus, channel_wga]
                
                
                feature_channels = []

                if image_index is None:
                    if hdf_channels.shape[0] > 2:
                        for i in range(2, len(hdf_channels)):
                            feature_channel = hdf_channels[i,window_y,window_x]
                            feature_channel = normalize(feature_channel)
                            feature_channel = feature_channel*cell_mask_extended
                            feature_channel = MinMax(feature_channel)
                            
                            feature_channels.append(feature_channel)
                else:
                    if hdf_channels.shape[1] > 2:  
                        feature_channel = hdf_channels[image_index, i,window_y,window_x]   
                        feature_channel = normalize(feature_channel)
                        feature_channel = feature_channel*cell_mask_extended
                        feature_channel = MinMax(feature_channel)
                        
                        feature_channels.append(feature_channel)  
                
                channels = required_maps+feature_channels
                stack = np.stack(channels, axis=0).astype("float16")
                
                if self.remap is not None:
                    stack = stack[self.remap]

                self._save_cell_info(save_index, index, image_index, label_info, stack) #to make more flexible for new datastructures with more labelling info

    def process(self, input_segmentation_path, filtered_classes_path):
        # is called with the path to the segmented image
        
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
        if os.path.isfile(center_path) and not self.overwrite:
            
            self.log("Cached version found, loading")
            with open(center_path, "rb") as input_file:
                center_nuclei = cPickle.load(input_file)
                px_center = np.round(center_nuclei).astype(int)
                
        else:
            self.log("Started class coordinate calculation")
            center_nuclei, length, _ = numba_mask_centroid(hdf_labels[0], debug=self.debug)
            px_center = np.round(center_nuclei).astype(int)
            self.log("Finished class coordinate calculation")
            with open(center_path, "wb") as output_file:
                cPickle.dump(center_nuclei, output_file)

            with open(os.path.join(self.directory,"length.pickle"), "wb") as output_file:
                cPickle.dump(length, output_file)
                
            del length

        class_list = self.get_classes(self, filtered_classes_path)

        #start extraction
        self.verbalise_extraction_info()

        self.log(f"Started parallel extraction of {self.num_classes} classes")
        start = timeit.default_timer()
        
        f = partial(self._extract_classes, input_segmentation_path)
        arg = self._get_arg(class_list, hf) #get arg to pass to _extract_classes (this depends on output dataformat)

        with Pool(processes=self.config["threads"]) as pool:
            x = pool.map(f, px_center, arg)
            
        stop = timeit.default_timer()
        duration = stop - start
        rate = num_classes/duration
        self.log(f"Finished parallel extraction in {duration:.2f} seconds ({rate:.2f} cells / second)")
        self.log("Collect cells")
        # collect cells
        
        # create empty hdf5 -> relocate to seperate function for more flexibility
        output_path = os.path.join(self.extraction_data_directory, self.DEFAULT_DATA_FILE)

        #get size of dataset that needs to be created
        current_level_files = [ name for name in os.listdir(self.extraction_cache) if os.path.isfile(os.path.join(self.extraction_cache, name))]
        num_classes = len(current_level_files)
        
        hf = self._create_hdf5(output_path, self.compression_type, num_classes, num_channels) #THIS IS SPECIFIC FUNCTION FOR DIFFERENT DATASET TYPES
        output_index = hf.get('single_cell_index')
        output_data = hf.get('single_cell_data')
        
        i = 0
        for file in current_level_files:
            result = self._write_cell(i, file, output_index, output_data)
            i += result #update index depending on what was done
                          
        hf.close()
        self.log("Finished collection")
        
        # remove cache
        self.log("Cleaning up cache")
        shutil.rmtree(self.extraction_cache)
        self.log("Finished cleaning up cache")

class TimecourseHDF5CellExtraction(HDF5CellExtraction):
    DEFAULT_SEGMENTATION_FILE = "input_segmentation.h5"
    channel_label = "input_images"
    segmentation_label = "segmentation"
    def __init__(self, 
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.get_labelling()

    def get_labelling(self):
        with h5py.File(self.input_segmentation_path, 'r') as hf:
            self.label_names = hf.get("label_names")[:]
            self.n_labels = len(self.label_names)

    def _get_arg(self, class_list):
        #need to extract ID for each cellnumber
        #generate lookuptable where we have all cellids for each tile id
        with h5py.File(self.input_segmentation_path, "r") as hf:
            labels = hf.get("labels")[:].astype("U13")
            segmentation = hf.get(self.segmentation_label)

            results = pd.DataFrame(columns = ["tileids", "cellids"], index = range(labels.shape[0]))
        
            self.log({"Extracting classes from each Segmentation Tile."})
            #should be updated later when classes saved in segmentation automatically 
            # currently not working because of issue with datatypes
            
            for i, tile_id in zip(labels.T[0], labels.T[1]):
                cellids = list(np.unique(segmentation[int(i)][0]))
                cellids.remove(0)
                results.loc[int(i), "cellids"] = cellids
                results.loc[int(i), "tileids"] = tile_id

        #map each cell id to tile id and generate a tuple which can be passed to later functions
        return_results = [[(cellid, i, results.loc[i, "tileids"]) for i, xset in enumerate(results.cellids) if cellid in set(xset)] for cellid in class_list]
        return_results = flatten(return_results)
        #required output format: [(class_id, image_index, label_id), (class_id, image_index, label_id)]
        return(return_results)

    def _get_label_info(self, arg):
        save_index, index, image_index, label_info = arg
        return(save_index, index, image_index, label_info)  

    def generate_save_index_lookup(self, class_list):
        lookup = pd.DataFrame(index = class_list)
        return(lookup)

    def _create_hdf5(self):
        #extract information about the annotation of cell ids
        column_labels = ['index', "cellid"] + list(self.label_names.astype("U13"))[1:]
        
        with h5py.File(self.output_path, 'w') as hf:
            #create special datatype for storing strings
            dt = h5py.special_dtype(vlen=str)

            #save label names so that you can always look up the column labelling
            hf.create_dataset('label_names', data = column_labels, chunks=None, dtype = dt)
            
            #generate index data container
            self.single_cell_index_shape =  (self.num_classes, len(column_labels))
            print(column_labels)
            hf.create_dataset('single_cell_index', self.single_cell_index_shape , chunks=None, dtype = dt)
            
            #generate datacontainer for the single cell images
            self.single_cell_data_shape = (self.num_classes,
                                                    self.n_channels_output,
                                                    self.config["image_size"],
                                                    self.config["image_size"])

            hf.create_dataset('single_cell_data', self.single_cell_data_shape,
                                                chunks=(1,
                                                        1,
                                                        self.config["image_size"],
                                                        self.config["image_size"]),
                                                compression=self.compression_type,
                                                dtype="float16")

    def _initialize_tempmmap_array(self):
        #define as global variables so that this is also avaialable in other functions
        global _tmp_single_cell_data, _tmp_single_cell_index
        
        #import tempmmap module and reset temp folder location
        from alphabase.io import tempmmap
        TEMP_DIR_NAME = tempmmap.redefine_temp_location(self.config["cache"])

        #generate container for single_cell_data
        _tmp_single_cell_data = tempmmap.array(self.single_cell_data_shape,dtype = np.float16)

        #generate container for single_cell_index
        #cannot be a temmmap array with object type as this doesnt work for memory mapped arrays
        dt = h5py.special_dtype(vlen=str)
        _tmp_single_cell_index  = np.empty(self.single_cell_index_shape, dtype = "<U32")
        
        #_tmp_single_cell_index  = tempmmap.array(self.single_cell_index_shape, dtype = "<U32")

        self.TEMP_DIR_NAME = TEMP_DIR_NAME

    def _transfer_tempmmap_to_hdf5(self):
        global _tmp_single_cell_data, _tmp_single_cell_index

        self.log(f"Transferring exracted single cells to .hdf5")
        with h5py.File(self.output_path, 'a') as hf:
            hf["single_cell_indexes"][:] = _tmp_single_cell_index
            hf["single_cell_data"][:] = _tmp_single_cell_data

        #delete tempobjects (to cleanup directory)
        self.log(f"Tempmmap Folder location {self.TEMP_DIR_NAME} will now be removed.")
        shutil.rmtree(self.TEMP_DIR_NAME, ignore_errors=True)

        del _tmp_single_cell_data, _tmp_single_cell_index, self.TEMP_DIR_NAME 

    def _save_cell_info(self, index, cell_id, image_index, label_info, stack):
        #label info is None so just ignore for the base case
        _tmp_single_cell_data[index] = stack

        #get label information
        with h5py.File(self.input_segmentation_path, "r") as hf:
            save_value = np.array(flatten([str(index), str(cell_id), hf.get("labels")[image_index].astype('U13')[1:]]))

            _tmp_single_cell_index[index][:] = save_value

            #double check that its really the same values
            if _tmp_single_cell_index[index][1] != label_info:
                self.log("ISSUE INDEXES DO NOT MATCH.")
                self.log(f"index: {index}")
                self.log(f"image_index: {image_index}")
                self.log(f"label_info: {label_info}")
    
    def process(self, input_segmentation_path, filtered_classes_path):
    # is called with the path to the segmented image
        
        self.setup_output()
        self.parse_remapping()

        complete_class_list = self.get_classes(filtered_classes_path)
        arg_list = self._get_arg(complete_class_list)
        lookup_saveindex = self.generate_save_index_lookup(complete_class_list)

        # setup cache
        self._create_hdf5()
        self._initialize_tempmmap_array()

        #start extraction
        self.log("Starting extraction.")
        self.verbalise_extraction_info()
        
        with  h5py.File(self.input_segmentation_path, 'r') as hf:
            start = timeit.default_timer()

            self.log(f"Loading segmentation data from {self.input_segmentation_path}")
            hdf_channels = hf.get(self.channel_label)
            hdf_labels = hf.get(self.segmentation_label)

            for arg in arg_list:
                cell_ids, image_index, label_info = arg 
                print(cell_ids)
                center_nuclei, _, _cell_ids = numba_mask_centroid(hdf_labels[image_index, 0, :, :], debug=self.debug)
                px_centers = np.round(center_nuclei).astype(int)
                _cell_ids = list(_cell_ids)
                for cell_id in cell_ids:
                    save_index = lookup_saveindex.index.get_loc(cell_id)
                    px_center = px_centers[_cell_ids.index(cell_id)]
                    self._extract_classes(input_segmentation_path, px_center,  (save_index, cell_id, image_index, label_info))
                print("done")
            stop = timeit.default_timer()

        duration = stop - start
        rate = self.num_classes/duration
        self.log(f"Finished parallel extraction in {duration:.2f} seconds ({rate:.2f} cells / second)")
        
        self.log("Collect cells")
        self._transfer_tempmmap_to_hdf5(self)
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

        #dapi = channels[0]
        #golgi = channels[1]
        #wga = channels[2]
        #labels_nuclei = segmentation[0]
        #labels_cell = segmentation[1]

        image_width = hdf_channels.shape[1]
        image_height = hdf_channels.shape[2]

        width = self.config["image_size"]//2
        
        window_y = slice(px_center[0]-width,px_center[0]+width)
        window_x = slice(px_center[1]-width,px_center[1]+width)
        
        if width < px_center[0] and px_center[0] < image_width-width and width < px_center[1] and px_center[1] < image_height-width:
            
            
            # channel 0: nucleus mask
            nuclei_mask = hdf_labels[0,window_y,window_x]
            
            nuclei_mask = np.where(nuclei_mask == index+1, 1,0)

            nuclei_mask_extended = gaussian(nuclei_mask,preserve_range=True,sigma=5)
            nuclei_mask = gaussian(nuclei_mask,preserve_range=True,sigma=1)

            
            if self.debug:
                plot_image(nuclei_mask, size=(5,5), save_name=os.path.join(self.extraction_data_directory,"{}_0".format(index)), cmap="Greys") 

            # channel 1: cell mask
            
            cell_mask = hdf_labels[1,window_y,window_x]
            cell_mask = np.where(cell_mask == index+1,1,0).astype(int)
            cell_mask = binary_fill_holes(cell_mask)

            cell_mask_extended = dilation(cell_mask,footprint=disk(6))

            cell_mask =  gaussian(cell_mask,preserve_range=True,sigma=1)   
            cell_mask_extended = gaussian(cell_mask_extended,preserve_range=True,sigma=5)



            if self.debug:
                plot_image(cell_mask, size=(5,5), save_name=os.path.join(self.extraction_data_directory,"{}_1".format(index)), cmap="Greys") 

            # channel 2: nucleus
            channel_nucleus = hdf_channels[0,window_y,window_x]
            channel_nucleus = normalize(channel_nucleus)
            channel_nucleus = channel_nucleus *nuclei_mask_extended

            channel_nucleus = MinMax(channel_nucleus)

            if self.debug:
                plot_image(channel_nucleus, size=(5,5), save_name=os.path.join(self.extraction_data_directory,"{}_2".format(index)))


            # channel 3: golgi
            channel_golgi = hdf_channels[1,window_y,window_x]
            channel_golgi = normalize(channel_golgi)
            channel_golgi = channel_golgi*cell_mask_extended

            channel_golgi = MinMax(channel_golgi)

            if self.debug:
                plot_image(channel_golgi, size=(5,5), save_name=os.path.join(self.extraction_data_directory,"{}_3".format(index)))

            # channel 4: cellmask
            channel_wga = hdf_channels[2,window_y,window_x]
            channel_wga = normalize(channel_wga)
            channel_wga = channel_wga*cell_mask_extended

            if self.debug:
                plot_image(channel_wga, size=(5,5), save_name=os.path.join(self.extraction_data_directory,"{}_4".format(index)))

            design = np.stack([nuclei_mask,cell_mask,channel_nucleus,channel_golgi,channel_wga], axis=-1).astype("float16")
            

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
            channel_nucleus = normalize(channel_nucleus)
            channel_nucleus = channel_nucleus *nuclei_mask_extended

            channel_nucleus = MinMax(channel_nucleus)
            
            """
            if self.debug:
                plot_image(channel_nucleus, size=(5,5), save_name=os.path.join(self.extraction_directory,"{}_2".format(index)))
            """

            # channel 3: golgi
            channel_golgi = hdf_channels[1,window_y,window_x]
            channel_golgi = normalize(channel_golgi)
            channel_golgi = channel_golgi*cell_mask_extended

            channel_golgi = MinMax(channel_golgi)
            
            """
            if self.debug:
                plot_image(channel_golgi, size=(5,5), save_name=os.path.join(self.extraction_directory,"{}_3".format(index)))
            """
            
            # channel 4: cellmask
            channel_wga = hdf_channels[2,window_y,window_x]
            channel_wga = normalize(channel_wga)
            channel_wga = channel_wga*cell_mask_extended
            
            """
            if self.debug:
                plot_image(channel_wga, size=(5,5), save_name=os.path.join(self.extraction_directory,"{}_4".format(index)))
            """
            
            stack = np.stack([nuclei_mask,cell_mask,channel_nucleus,channel_golgi,channel_wga], axis=0).astype("float16")
            np.save(os.path.join(self.extraction_cache,"{}.npy".format(index)),stack)
            
        input_hdf.close()      
