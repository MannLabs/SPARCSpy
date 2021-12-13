from datetime import datetime
import os
import numpy as np
import csv
from functools import partial
from multiprocessing import Pool
import h5py

from skimage.filters import gaussian
from skimage.morphology import disk, dilation

from scipy.ndimage import binary_fill_holes


from vipercore.processing.segmentation import numba_mask_centroid
from vipercore.processing.utils import plot_image
from vipercore.processing.deprecated import normalize, MinMax
from vipercore.pipeline.base import ProcessingStep

import uuid
import shutil
import timeit

import _pickle as cPickle



class HDF5CellExtraction(ProcessingStep):
    
    
    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_DATA_FILE = "single_cells.h5"
    DEFAULT_DATA_DIR = "data"
    CLEAN_LOG = False
    
    def __init__(self, 
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
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
            
        
        
            
        
    def get_output_path(self):
        self.extraction_data_directory = os.path.join(self.directory, self.DEFAULT_DATA_DIR)
        return self.extraction_data_directory            
                
        
    def process(self, input_segmentation_path, filtered_classes_path):
        # is called with the path to the segmented image
        
        self.extraction_data_directory = os.path.join(self.directory, self.DEFAULT_DATA_DIR)
        if not os.path.isdir(self.extraction_data_directory):
            os.makedirs(self.extraction_data_directory)
            self.log("Created new data directory " + self.extraction_data_directory)
        
        # parse remapping
        self.remap = None
        if "channel_remap" in self.config:
            char_list = self.config["channel_remap"].split(",")
            self.log("channel remap parameter found:")
            self.log(char_list)
            
            self.remap = [int(el.strip()) for el in char_list]
        
        # setup cache
        self.uuid = str(uuid.uuid4())
        self.extraction_cache = os.path.join(self.config["cache"],self.uuid)
        if not os.path.isdir(self.extraction_cache):
            os.makedirs(self.extraction_cache)
            self.log("Created new extraction cache " + self.extraction_cache)
            
        self.log("Started extraction")
        self.log("Loading segmentation data from {input_segmentation_path}")
        
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
        
        center_path = os.path.join(self.directory,"center.pickle")
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

            with open(os.path.join(self.directory,"length.pickle"), "wb") as output_file:
                cPickle.dump(length, output_file)
                
            del length
        
        # parallel execution
        
        class_list = list(filtered_classes)
        # Zero class contains background
        
        if 0 in class_list: class_list.remove(0)
            
        num_classes = len(class_list)
        
        num_channels = len(hdf_channels) + len(hdf_labels)
        
        self.log(f"Input channels: {len(hdf_channels)}")
        self.log(f"Input labels: {len(hdf_labels)}")
        self.log(f"Output channels: {num_channels}")
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
        
        compression_type = "lzf" if self.config["compression"] else None
        
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
                                               compression=compression_type,
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

            cell_mask_extended = dilation(cell_mask,selem=disk(6))

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
            
                
            # channel 3: cellmask
            channel_wga = hdf_channels[1,window_y,window_x]
            channel_wga = normalize(channel_wga)
            channel_wga = channel_wga*cell_mask_extended
            
            
            required_maps = [nuclei_mask, cell_mask, channel_nucleus, channel_wga]
            
            feature_channels = []
            
            for i in range(2,len(hdf_channels)):
                feature_channel = hdf_channels[i,window_y,window_x]
                feature_channel = normalize(feature_channel)
                feature_channel = feature_channel*cell_mask_extended

                feature_channel = MinMax(feature_channel)
                
                feature_channels.append(feature_channel)
            
           
            channels = required_maps+feature_channels
            stack = np.stack(channels, axis=0).astype("float16")
            
            if self.remap is not None:
                stack = stack[self.remap]
                
            np.save(os.path.join(self.extraction_cache,"{}.npy".format(index)),stack)
            
        input_hdf.close()

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

            cell_mask_extended = dilation(cell_mask,selem=disk(6))

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
        
        compression_type = "lzf" if self.config["compression"] else None
        
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
                                               compression=compression_type,
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

            cell_mask_extended = dilation(cell_mask,selem=disk(6))

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
