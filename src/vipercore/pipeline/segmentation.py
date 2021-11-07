
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import skfmm
import csv
import h5py
from functools import partial
from multiprocessing import Pool
import shutil
import warnings

from skimage.filters import gaussian, median
from skimage.morphology import binary_erosion, disk, dilation
from skimage.segmentation import watershed
from skimage.color import label2rgb

from vipercore.processing.segmentation import segment_local_tresh, mask_centroid, contact_filter, size_filter, shift_labels
from vipercore.processing.utils import plot_image

from vipercore.pipeline.base import ProcessingStep

class Segmentation(ProcessingStep):
    """Segmentation helper class used for creating segmentation workflows.
    Attributes:
        maps (dict(str)): Segmentation workflows based on the :class:`.Segmentation` class can use maps for saving and loading checkpoints and perform. Maps can be numpy arrays 
        
        DEFAULT_OUTPUT_FILE (str, default ``segmentation.h5``)
        DEFAULT_FILTER_FILE (str, default ``classes.csv``)
        PRINT_MAPS_ON_DEBUG (bool, default ``False``)
        
        identifier (int, default ``None``): Only set if called by :class:`ShardedSegmentation`. Unique index of the shard.
            
        window (list(tuple), default ``None``): Only set if called by :class:`ShardedSegmentation`. Defines the window which is assigned to the shard. The window will be applied to the input. The first element refers to the first dimension of the image and so on. For example use ``[(0,1000),(0,2000)]`` To crop the image to `1000 px height` and `2000 px width` from the top left corner.

        input_path (str, default ``None``): Only set if called by :class:`ShardedSegmentation`. Location of the input hdf5 file. During sharded segmentation the :class:`ShardedSegmentation` derived helper class will save the input image in form of a hdf5 file. This makes the input image available for parallel reading by the segmentation processes.
        
    Example:
        .. code-block:: python
        
            def process(self):
                # two maps are initialized
                self.maps = {"map0": None,
                             "map1": None}
                
                # its checked if the segmentation directory already contains these maps and they are then loaded. The index of the first map which has not been found is returned. It indicates the step where computation needs to resume
                current_step = self.load_maps_from_disk()
                
                if current_step <= 0:
                    # do stuff and generate map0
                    self.save_map("map0")
                    
                if current_step <= 1:
                    # do stuff and generate map1
                    self.save_map("map1")
                            
    """
    DEFAULT_OUTPUT_FILE = "segmentation.h5"
    DEFAULT_FILTER_FILE = "classes.csv"
    PRINT_MAPS_ON_DEBUG = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.identifier = None
        self.window = None
        self.input_path = None
        
    
    def initialize_as_shard(self, 
                            identifier, 
                            window,
                            input_path):
        """Initialize Segmentation Step with further parameters needed for federated segmentation. 
        
        Important:
        
            This function is intented for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.
            
        Args:
            identifier (int): Unique index of the shard.
            
            window (list(tuple)): Defines the window which is assigned to the shard. The window will be applied to the input. The first element refers to the first dimension of the image and so on. For example use ``[(0,1000),(0,2000)]`` To crop the image to `1000 px height` and `2000 px width` from the top left corner.
            
            input_path (str): Location of the input hdf5 file. During sharded segmentation the :class:`ShardedSegmentation` derived helper class will save the input image in form of a hdf5 file. This makes the input image available for parallel reading by the segmentation processes.
        
        """
        
        self.identifier = identifier
        self.window = window
        self.input_path = input_path
        

    def call_as_shard(self):
        """Wrapper function for calling a sharded segmentation.
        
        Important:
        
            This function is intented for internal use by the :class:`ShardedSegmentation` helper class. In most cases it is not relevant to the creation of custom segmentation workflows.
        
        """
        
        hf = h5py.File(self.input_path, 'r')
        hdf_input = hf.get('channels')
        input_image = hdf_input[:,self.window[0],self.window[1]]
        hf.close()
            
        super().__call__(input_image)

    def save_segmentation(self, 
                          channels, 
                          labels, 
                          classes):
        
        """Saves the results of a segmentation at the end of the process.
        
        Args:
            channels (np.array): Numpy array of shape ``(height, width)`` or``(channels, height, width)``. Channels are all data which are saved as floating point values e.g. images.
            labels (np.array): Numpy array of shape ``(height, width)``. Labels are all data which are saved as integer values. These are mostly segmentation maps with integer values corresponding to the labels of cells.
            
            classes (list(int)): List of all classes in the labels array, which have passed the filtering step. All classes contained in this list will be extracted.
        
        """
        self.log("saving segmentation")
        
        # size (C, H, W) is expected
        # dims are expanded in case (H, W) is passed
        
        channels = np.expand_dims(channels, axis=0) if len(channels.shape) == 2 else channels
        labels = np.expand_dims(labels, axis=0) if len(labels.shape) == 2 else labels
        
        map_path = os.path.join(self.directory, self.DEFAULT_OUTPUT_FILE)
        hf = h5py.File(map_path, 'w')
        
        hf.create_dataset('labels', data=labels, chunks=(1, self.config["chunk_size"], self.config["chunk_size"]))
        hf.create_dataset('channels', data=channels, chunks=(1, self.config["chunk_size"], self.config["chunk_size"]))
        hf.close()
        
        # save classes
        filtered_path = os.path.join(self.directory, self.DEFAULT_FILTER_FILE)
        
        to_write = "\n".join([str(i) for i in list(classes)])
        with open(filtered_path, 'w') as myfile:
            myfile.write(to_write)
            
        self.log("=== finished segmentation ===")
            
    def load_maps_from_disk(self):     
        
        """Tries to load all maps which were defined in ``self.maps`` and returns the current state of processing.
        
        Returns
            (int): Index of the first map which could not be loaded. An index of zero indicates that computation needs to start at the first map.
        
        """
        
        if not hasattr(self, "maps"):
            raise AttributeError("No maps have been defined. Therefore saving and loading of maps as checkpoints is not supported. Initialize maps in the process method of the segmentation like self.maps = {'map1': None,'map2': None}")
            
        # iterating over all maps
        for map_index, map_name in enumerate(self.maps.keys()):

            
            try:
                map_path = os.path.join(self.directory, "{}_{}_map.npy".format(map_index, map_name))

                if os.path.isfile(map_path):
                    map = np.load(map_path)
                    self.log("Loaded map {} {} from path {}".format(map_index, map_name, map_path))
                    self.maps[map_name] = map
                else:
                    self.log("No existing map {} {} found at path {}, new one will be created".format(map_index,map_name, map_path))
                    self.maps[map_name] = None
                    
            except:
                self.log("Error loading map {} {} from path {}".format(map_index,map_name, map_path))
                self.maps[map_name] = None
        
        # determine where to start based on precomputed maps and parameters
        # get index of lowest map which could not be loaded
        # reruslts in index of step where to start
                
        is_not_none = [el is not None for el in self.maps.values()]
        return np.argmin(is_not_none) if not all(is_not_none) else len(is_not_none)
    
    def save_map(self, map_name):
        """Saves newly computed map.
        
        Args
            map_name (str): name of the map to be saved, as defined in ``self.maps``.
            
        Example:
        
            .. code-block:: python
                
                # declare all intermediate maps
                self.maps = {"myMap": None}
                
                # load intermediate maps if possible and get current processing step
                current_step = self.load_maps_from_disk()
                
                if current_step <= 0:

                    # do some computations
    
                    self.maps["myMap"] = myNumpyArray
                    
                    # save map
                    self.save_map("myMap")
            
        
        """
        
        if self.maps[map_name] is None:
            self.log("Error saving map {}, map is None".format(map_name))
        else:
            map_index = list(self.maps.keys()).index(map_name)
            
            if self.intermediate_output:
                map_path = os.path.join(self.directory, "{}_{}_map.npy".format(map_index, map_name))
                np.save(map_path, self.maps[map_name])
                self.log("Saved map {} {} under path {}".format(map_index,map_name, map_path))
            
            # check if map contains more than one channel (3, 1024, 1024) vs (1024, 1024)
            
            if len(self.maps[map_name].shape) > 2:
                
                for i, channel in enumerate(self.maps[map_name]):
                    
                    channel_name = "{}_{}_{}_map.png".format(map_index, map_name,i)
                    channel_path = os.path.join(self.directory, channel_name)
                    
                    if self.debug and self.PRINT_MAPS_ON_DEBUG:
                        self.save_image(channel, save_name = channel_path)
            else:
                channel_name = "{}_{}_map.png".format(map_index, map_name)
                channel_path = os.path.join(self.directory, channel_name)
                
                if self.debug and self.PRINT_MAPS_ON_DEBUG:
                    self.save_image(self.maps[map_name], save_name = channel_path)
                
    def save_image(self, array, save_name="", cmap="magma",**kwargs):
        
        fig = plt.figure(frameon=False)
        fig.set_size_inches((10, 10))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(array, cmap=cmap,**kwargs)

        if save_name != "":
            plt.savefig(save_name)
            plt.show()
            plt.close()
    
        
        
class ShardedSegmentation(ProcessingStep):
    """object which can create log entries.
        
    Attributes:
        DEFAULT_OUTPUT_FILE (str, default ``segmentation.h5``): Default output file name for segmentations.
        
        DEFAULT_FILTER_FILE (str, default ``classes.csv``): Default file with filtered class IDs.
        
        DEFAULT_INPUT_IMAGE_NAME (str, default ``input_image.h5``): Default name for the input image, which is written to disk as hdf5 file.
        
        DEFAULT_SHARD_FOLDER (str, default ``tiles``): Date and time format used for logging.
    """
    DEFAULT_OUTPUT_FILE = "segmentation.h5"
    DEFAULT_FILTER_FILE = "classes.csv"
    DEFAULT_INPUT_IMAGE_NAME = "input_image.h5"
    DEFAULT_SHARD_FOLDER = "tiles"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not hasattr(self, "method"):
            raise AttributeError("No Segmentation method defined, please set attribute ``method``")
        
    def process(self, input_image):
        
        self.shard_directory = os.path.join(self.directory, self.DEFAULT_SHARD_FOLDER)
        
        if not os.path.isdir(self.shard_directory):
            os.makedirs(self.shard_directory)
            self.log("Created new shard directory " + self.shard_directory)
            
        self.save_input_image(input_image)
        # calculate sharding plan
        
        
        self.image_size = input_image.shape[1:]

        if self.config["shard_size"] >= np.prod(self.image_size):
            self.log("target size is equal or larger to input image. Sharding will not be used.")

            sharding_plan = [(slice(0,self.image_size[0]),slice(0,self.image_size[1]))]
        else:
            self.log("target size is smaller than input image. Sharding will be used.")
            sharding_plan = self.calculate_sharding_plan(self.image_size)
            
        shard_list = self.initialize_shard_list(sharding_plan)
        self.log(f"sharding plan with {len(sharding_plan)} elements generated, sharding with {self.config['threads']} threads begins")
        
        
        with Pool(processes=self.config["threads"]) as pool:
            x = pool.map(self.method.call_as_shard, shard_list)  
        self.log("Finished parallel segmentation")
        
        self.resolve_sharding(sharding_plan)
        
        self.log("=== finished segmentation === ")
        
            
    def initialize_shard_list(self, sharding_plan):
        _shard_list = []
        
        input_path = os.path.join(self.directory, self.DEFAULT_INPUT_IMAGE_NAME)
        
        for i, window in enumerate(sharding_plan):
            local_shard_directory = os.path.join(self.shard_directory,str(i))
            
            current_shard = self.method(
                 self.config,
                 local_shard_directory,
                 debug=self.debug,
                 overwrite = self.overwrite,
                intermediate_output = self.intermediate_output
            )
            
            current_shard.initialize_as_shard(i, window, input_path)
            
            _shard_list.append(current_shard)
        return _shard_list
        
    def save_input_image(self, input_image):
        path = os.path.join(self.directory, self.DEFAULT_INPUT_IMAGE_NAME)
        hf = h5py.File(path, 'w')
        hf.create_dataset('channels', data=input_image, chunks=(1, self.config["chunk_size"], self.config["chunk_size"]))
        hf.close()
        self.log(f"saved input_image: {path}")
        
    def calculate_sharding_plan(self, image_size):
        _sharding_plan = []
        side_size = np.floor(np.sqrt(int(self.config["shard_size"])))
        shards_side = np.round(image_size/side_size).astype(int)
        shard_size = image_size//shards_side
        
        self.log(f"input image {image_size[0]} px by {image_size[1]} px")
        self.log(f"target_shard_size: {self.config['shard_size']}")
        self.log(f"sharding plan:")
        self.log(f"{shards_side[0]} rows by {shards_side[1]} columns")
        self.log(f"{shard_size[0]} px by {shard_size[1]} px")

        for y in range(shards_side[0]):
            for x in range(shards_side[1]):

                last_row = y == shards_side[0]-1
                last_column = x == shards_side[1]-1

                lower_y = y * shard_size[0]
                lower_x = x * shard_size[1]

                upper_y = (y+1) * shard_size[0]
                upper_x = (x+1) * shard_size[1]

                if last_row: 
                    upper_y = image_size[0]

                if last_column: 
                    upper_x = image_size[1]

                shard = (slice(lower_y, upper_y), slice(lower_x, upper_x))
                _sharding_plan.append(shard)
        return _sharding_plan
    
    def resolve_sharding(self, sharding_plan):
        """
        The function iterates over a sharding plan and generates a new stitched hdf5 based segmentation.
        """
        
        self.log("resolve sharding plan")
        
        output = os.path.join(self.directory,self.DEFAULT_OUTPUT_FILE)
        
        label_size = (2,self.image_size[0],self.image_size[1])
        channel_size = (3,self.image_size[0],self.image_size[1])

        
        hf = h5py.File(output, 'w')
        
        hdf_labels = hf.create_dataset('labels', 
                          label_size, 
                          chunks=(1, self.config["chunk_size"], self.config["chunk_size"]), 
                          dtype = "int32")
        
        hdf_channels = hf.create_dataset('channels', 
                          channel_size, 
                          chunks=(1, self.config["chunk_size"], self.config["chunk_size"]),
                          dtype = "float16")
        
        
        class_id_shift = 0
        
        filtered_classes_combined = []
        edge_classes_combined = []
        for i, window in enumerate(sharding_plan):
            
            self.log(f"Stitching tile {i}")
            
            local_shard_directory = os.path.join(self.shard_directory,str(i))
            local_output = os.path.join(local_shard_directory,self.DEFAULT_OUTPUT_FILE)
            local_classes = os.path.join(local_shard_directory,"classes.csv")
            
            cr = csv.reader(open(local_classes,'r'))
            filtered_classes = [int(el[0]) for el in list(cr)]
            filtered_classes_combined += [class_id + class_id_shift for class_id in filtered_classes if class_id != 0]
            
            local_hf = h5py.File(local_output, 'r')
            local_hdf_channels = local_hf.get('channels')
            local_hdf_labels = local_hf.get('labels')
            
            
            shifted_map, edge_labels = shift_labels(local_hdf_labels, class_id_shift, return_shifted_labels=True)
            
            hdf_labels[:,window[0],window[1]] = shifted_map
            hdf_channels[:,window[0],window[1]]=local_hdf_channels
            
            edge_classes_combined += edge_labels
            class_id_shift += np.max(local_hdf_labels[0])
            
            local_hf.close()
            self.log(f"Finished stitching tile {i}")
        
        classes_after_edges = [item for item in filtered_classes_combined if item not in edge_classes_combined]

        self.log("Number of filtered classes combined after sharding:")
        self.log(len(filtered_classes_combined))
        
        self.log("Number of classes in contact with shard edges:")
        self.log(len(edge_classes_combined))
        
        self.log("Number of classes after removing shard edges:")
        self.log(len(classes_after_edges))
        
        # save newly generated class list
        # print filtered classes
        filtered_path = os.path.join(self.directory, self.DEFAULT_FILTER_FILE)
        to_write = "\n".join([str(i) for i in list(classes_after_edges)])
        with open(filtered_path, 'w') as myfile:
            myfile.write(to_write)
            
        # sanity check of class reconstruction   
        if self.debug:    
            all_classes = set(hdf_labels[:].flatten())
            if set(edge_classes_combined).issubset(set(all_classes)):
                self.log("Sharding sanity check: edge classes are a full subset of all classes")
            else:
                self.log("Sharding sanity check: edge classes are NOT a full subset of all classes.")
            
            plot_image(hdf_channels[0].astype(np.float64))
            plot_image(hdf_channels[1].astype(np.float64))
            plot_image(hdf_channels[2].astype(np.float64))

            image = label2rgb(hdf_labels[1],hdf_channels[0].astype(np.float64)/np.max(hdf_channels[0].astype(np.float64)),alpha=0.2, bg_label=0)
            plot_image(image)
        
        
        
        
        hf.close()      
        
        self.log("resolved sharding plan")