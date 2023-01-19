# -*- coding: utf-8 -*-
import warnings
import shutil
import os
import yaml
from PIL import Image
import PIL
import numpy as np
import sys

#packages for timecourse project
import pandas as pd
from cv2 import imread
import re
import h5py
from tqdm import tqdm

from vipercore.pipeline.base import Logable

class Project(Logable):
    """
    Project base class used to create a new viper project.
    
    Args:
        location_path (str): Path to the folder where to project should be created. The folder is created in case the specified folder does not exist.
        
        config_path (str, optional, default ""): Path pointing to a valid configuration file. The file will be copied to the project directory and renamed to the name specified in ``DEFAULT_CLASSIFICATION_DIR_NAME``. If no config is specified, the existing config in the project directory will be used, if possible. See the section configuration to find out more about the config file. 
        
        intermediate_output (bool, default ``False``): When set to True intermediate outputs will be saved where applicable.
            
        debug (bool, default ``False``): When set to True debug outputs will be printed where applicable. 
            
        overwrite (bool, default ``False``): When set to True, the processing step directory will be completely deleted and newly created when called.
            
        segmentation_f (Class, default ``None``): Class containing segmentation workflow.
            
        extraction_f (Class, default ``None``): Class containing extraction workflow.
            
        classification_f (Class, default ``None``): Class containing classification workflow.
            
        selection_f (Class, default ``None``): Class containing selection workflow.
            
    Attributes:
        DEFAULT_CONFIG_NAME (str, default "config.yml"): Default config name which is used for the config file in the project directory. This name needs to be used when no config is supplied and the config is manually created in the project folder.
        DEFAULT_SEGMENTATION_DIR_NAME (str, default "segmentation"): Default foldername for the segmentation process.
        DEFAULT_EXTRACTION_DIR_NAME (str, default "extraction"): Default foldername for the extraction process.
        DEFAULT_CLASSIFICATION_DIR_NAME (str, default "selection"): Default foldername for the classification process.
        DEFAULT_SELECTION_DIR_NAME (str, default "classification"): Default foldername for the selection process.
    
    """

    DEFAULT_CONFIG_NAME = "config.yml"
    DEFAULT_SEGMENTATION_DIR_NAME = "segmentation"
    DEFAULT_EXTRACTION_DIR_NAME = "extraction"
    DEFAULT_CLASSIFICATION_DIR_NAME = "classification"
    DEFAULT_SELECTION_DIR_NAME = "selection"
    
    # Project object is initialized, nothing is written to disk
    def __init__(self,    
                 location_path,
                 *args,
                 config_path =  "",
                 intermediate_output = False,
                 debug = False,
                 overwrite = False,
                 segmentation_f = None,
                 extraction_f = None,
                 classification_f = None,
                 selection_f = None, 
                 **kwargs
                 ):
        
        super().__init__(debug=debug)
        
        self.debug = debug
        self.overwrite = overwrite
        self.intermediate_output = intermediate_output
         
        self.segmentation_f = segmentation_f
        self.extraction_f = extraction_f
        self.classification_f = classification_f
        self.selection_f = selection_f,
        
        # PIL limit used to protect from large image attacks
        PIL.Image.MAX_IMAGE_PIXELS = 10000000000
        
        self.input_image = None
        self.config = None
        
        self.directory = location_path
        
        # handle location
        self.project_location = location_path
        
        # check if project dir exists and creates it if not
        if not os.path.isdir(self.project_location):
                os.makedirs(self.project_location)
        else:
            warnings.warn("Theres already a directory in the location path")
                
        # handle configuration file
        new_config_path = os.path.join(self.project_location,self.DEFAULT_CONFIG_NAME)
        
        if config_path == "":
            
            # Check if there is already a config file in the dataset folder in case no config file has been specified
            
            if os.path.isfile(new_config_path):
                self._load_config_from_file(new_config_path)
            
            else:
                warnings.warn(f"You will need to add a config named {self.DEFAULT_CONFIG_NAME} file manually to the dataset")
        
        else:
            if not os.path.isfile(config_path):
                raise ValueError("Your config path is invalid")
                
            else:       
                
                print("modifying config")
                if os.path.isfile(new_config_path):
                    os.remove(new_config_path) 
                    
                # The blueprint config file is copied to the dataset folder and renamed to the default name
                shutil.copy(config_path, new_config_path)
                self._load_config_from_file(new_config_path)
                
        
        # === setup segmentation ===
        if self.segmentation_f is not None:
            if not segmentation_f.__name__ in self.config:
                raise ValueError(f"Config for {segmentation_f.__name__} is missing from the config file")
                
            seg_directory = os.path.join(self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME)
            self.segmentation_f = segmentation_f(self.config[segmentation_f.__name__], 
                                                 seg_directory,
                                                 debug = self.debug,
                                                 overwrite = self.overwrite,
                                                 intermediate_output = self.intermediate_output)
        else:
            self.segmentation_f = None
            
        # === setup extraction ===
        if extraction_f is not None:
            extraction_directory = os.path.join(self.project_location, self.DEFAULT_EXTRACTION_DIR_NAME)
            
            if not extraction_f.__name__ in self.config:
                raise ValueError(f"Config for {extraction_f.__name__} is missing from the config file")
            
            self.extraction_f = extraction_f(self.config[extraction_f.__name__], 
                                                 extraction_directory,
                                                 debug = self.debug,
                                                 overwrite = self.overwrite,
                                                 intermediate_output = self.intermediate_output)
        else:
            self.extraction_f = None
            
            
        # === setup classification ===
        if classification_f is not None:
            if not classification_f.__name__ in self.config:
                raise ValueError(f"Config for {classification_f.__name__} is missing from the config file")
                
            classification_directory = os.path.join(self.project_location, self.DEFAULT_CLASSIFICATION_DIR_NAME)
            self.classification_f = classification_f(self.config[classification_f.__name__], 
                                                 classification_directory,
                                                 debug = self.debug,
                                                 overwrite = self.overwrite,
                                                 intermediate_output = self.intermediate_output)
        else:
            self.classification_f = None
            
        # === setup selection ===
        if selection_f is not None:
            if not selection_f.__name__ in self.config:
                raise ValueError(f"Config for {selection_f.__name__} is missing from the config file")
                
            selection_directory = os.path.join(self.project_location, self.DEFAULT_SELECTION_DIR_NAME)
            self.selection_f = selection_f(self.config[selection_f.__name__], 
                                                 selection_directory,
                                                 debug = self.debug,
                                                 overwrite = self.overwrite,
                                                 intermediate_output = self.intermediate_output)
        else:
            self.selection_f = None
            
        # parse remapping
        self.remap = None
        if "channel_remap" in self.config:
            char_list = self.config["channel_remap"].split(",")
            self.log("channel remap parameter found:")
            self.log(char_list)
            
            self.remap = [int(el.strip()) for el in char_list]
            
          
    def _load_config_from_file(self, file_path):
        """
        loads config from file and writes it to self.config
        """
        self.log(f"Loading config from {file_path}")
        if not os.path.isfile(file_path):
            raise ValueError("Your config path is invalid")
            
        with open(file_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
                
    def load_input_from_file(self, file_paths, crop=[(0,-1),(0,-1)]):
        """load input image from a number of files.

        Args:
            file_paths (list(str)): List containing paths to each channel like ``["path1/img.tiff", "path2/img.tiff", "path3/img.tiff"]``. Expects a list of file paths with length ``input_channel`` as defined in the config.yml. Input data is NOT copied to the project folder by default. Different segmentation functions especially tiled segmentations might copy the input
            
            crop (list(tuple), optional): When set it can be used to crop the input image. The first element refers to the first dimension of the image and so on. For example use ``[(0,1000),(0,2000)]`` To crop the image to `1000 px height` and `2000 px width` from the top left corner.
        """
        if self.config == None:
            raise ValueError("Dataset has no config file loaded")
            
        # 
        # remap can be used to shuffle the order, for example [1, 0, 2] to invert the first two channels
        #         
        
        if not len(file_paths) == self.config["input_channels"]:
            raise ValueError("Expected {} image paths, only received {}".format(self.config["input_channels"], len(file_paths)))
        
        # append all images channel wise and remap them according to the supplied list
        channels = []
        
        for channel_path in file_paths:
            im = Image.open(channel_path)
            c = np.array(im, dtype="float64")[slice(*crop[0]),slice(*crop[1])]
            
            channels.append(c)
        
        self.input_image = np.stack(channels)
        
        print(self.input_image.shape)
        
        if self.remap != None:
            self.input_image = self.input_image[self.remap]
        
        
    def load_input_from_array(self, array, remap = None):
        """load input image from a number of files.

        Args:
            array (numpy.ndarray): Numpy array of shape ``[channels, height, width]`` .
            
            remap (list(int), optional): Define remapping of channels. For example use ``[1, 0, 2]`` to change the order of the first and the second channel. 

        """
        # input data is not copied to the project folder
        if self.config == None:
            raise ValueError("Dataset has no config file loaded")
            
        if not array.shape[0] == self.config["input_channels"]:
            raise ValueError("Expected {} image paths, only received {}".format(self.config["input_channels"], array.shape[0]))
            
        self.input_image = np.array(array, dtype="float64")
            
        if self.remap != None:
            self.input_image = self.input_image[self.remap]
            
    def segment(self, 
                *args, 
                **kwargs):
        
        """segment project with the defined segmentation under segmentation_f.
        
        Args:
            intermediate_output (bool, optional): Can be set when calling to override the project wide flag.

            debug (bool, optional):Can be set when calling to override the project wide flag.

            overwrite (bool, optional): Can be set when calling to override the project wide flag.
        """
        
        if self.segmentation_f == None:
            raise ValueError("No segmentation method defined")
            
        elif type(self.input_image) == None:
            raise ValueError("No input image defined")
        else:
            self.segmentation_f(self.input_image, *args, **kwargs)
            
    def extract(self, 
                *args, 
                **kwargs):
        """extract single cells with the defined extraction at extraction_f.
        
        Args:
            intermediate_output (bool, optional): Can be set when calling to override the project wide flag.

            debug (bool, optional):Can be set when calling to override the project wide flag.

            overwrite (bool, optional): Can be set when calling to override the project wide flag.
        """
        
        if self.extraction_f == None:
            raise ValueError("No extraction method defined")
            
        input_segmentation = self.segmentation_f.get_output()
        
        input_dir = os.path.join(self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv")
        self.extraction_f(input_segmentation,  input_dir, *args, **kwargs)
        
    def classify(self, 
                *args, 
                **kwargs):
        """classify single cells with the defined classification at classification_f.
        
        Args:
            intermediate_output (bool, optional): Can be set when calling to override the project wide flag.

            debug (bool, optional):Can be set when calling to override the project wide flag.

            overwrite (bool, optional): Can be set when calling to override the project wide flag.
        """
            
        input_extraction = self.extraction_f.get_output_path()
        

        if not os.path.isdir(input_extraction):
            raise ValueError("input was not found at {}".format(input_extraction))
            
        self.classification_f(input_extraction, *args, **kwargs)
        
    def select(self, 
                *args, 
                **kwargs):
        """classify single cells with the defined classification at classification_f.
        
        Args:
            intermediate_output (bool, optional): Can be set when calling to override the project wide flag.

            debug (bool, optional):Can be set when calling to override the project wide flag.

            overwrite (bool, optional): Can be set when calling to override the project wide flag.
        """
                 
        if self.selection_f == None:
            raise ValueError("No classification method defined")
            pass
            
        input_selection = self.segmentation_f.get_output()
            
        self.selection_f(input_selection, *args, **kwargs)
            
    def process(self):
        self.segment()
        self.extract()
        self.classify()

class TimecourseProject(Project):
    DEFAULT_INPUT_IMAGE_NAME = "input_segmentation.h5"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_input_from_array(self, img, label, overwrite = False):
        #check if already exists if so throw error message
        if not os.path.isdir(os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)):
            os.makedirs(os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME))
            
        path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, self.DEFAULT_INPUT_IMAGE_NAME)
        
        if not overwrite:
            if os.path.isfile(path):
                sys.exit("File already exists")
            else:
                overwrite = True
        
        if overwrite:
            #column labels
            column_labels = ["label"]

            #create .h5 dataset to which all results are written
            path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, self.DEFAULT_INPUT_IMAGE_NAME)
            hf = h5py.File(path, 'w')
            dt = h5py.special_dtype(vlen=str)
            hf.create_dataset('label_names', data = column_labels, chunks=None, dtype = dt)
            hf.create_dataset('labels', data = label, chunks=None, dtype = dt)
            hf.create_dataset('input_images', data = img, chunks=(1, 1, img.shape[2], img.shape[2]))

            print(hf.keys())
            hf.close()

    def load_input_from_files(self, input_dir, channels, timepoints, plate_layout, img_size = 1080, overwrite = False):    
        """
        Function to load timecourse experiments recorded with opera phenix into .h5 dataformat for further processing.
        """

        #check if already exists if so throw error message
        if not os.path.isdir(os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME)):
            os.makedirs(os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME))

        path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, self.DEFAULT_INPUT_IMAGE_NAME)
        
        if not overwrite:
            if os.path.isfile(path):
                sys.exit("File already exists")
            else:
                overwrite = True
        
        if overwrite:
            self.img_size = img_size
            
            def _read_write_images(dir, indexes, h5py_path):
                #unpack indexes
                index_start, index_end = indexes
                
                #get information on directory
                well = re.match("^Row._Well[0-9]", dir).group()
                region = re.search("r..._c...$", dir).group()
                
                #list all images within directory
                path = os.path.join(input_dir, dir)
                files = os.listdir(path)

                #filter to only contain the timepoints of interest
                files = np.sort([x for x in files if x.startswith(tuple(timepoints))])

                #filter to only contain the files listes in the plate layout
                
                #read images for that region
                imgs = np.empty((n_timepoints, n_channels, img_size, img_size), dtype='uint16')
                for ix, channel in enumerate(channels):
                    images = [x for x in files if channel in x]
                    for id, im in enumerate(images):
                        imgs[id, ix, :, :] = imread(os.path.join(path, im), 0)

                #create labelling 
                column_values = []
                for column in plate_layout.columns:
                    column_values.append(plate_layout.loc[well, column])

                list_input = [list(range(index_start, index_end)), [dir + "_"+ x for x in timepoints], [dir] * n_timepoints, timepoints, [well]*n_timepoints, [region]*n_timepoints]
                list_input = [np.array(x) for x in list_input]

                for x in column_values:
                    list_input.append(np.array([x]*n_timepoints))

                labelling = np.array(list_input).T

                input_images[index_start:index_end, :, :, :] = imgs
                labels[index_start:index_end] = labelling
                
            #read plate layout
            plate_layout = pd.read_csv(plate_layout, sep = "\s+|;|,", engine = "python")
            plate_layout = plate_layout.set_index("Well")
            
            column_labels = ["index", "ID", "location", "timepoint", "well", "region"] + plate_layout.columns.tolist()
            
            #get information on number of timepoints and number of channels
            n_timepoints = len(timepoints)
            n_channels = len(channels)
            wells = np.unique(plate_layout.index.tolist())
            
            #get all directories contained within the input dir
            directories = os.listdir(input_dir)
            directories.remove('.DS_Store')  #need to remove this because otherwise it gives errors

            #filter directories to only contain those listed in the plate layout
            directories = [_dir for _dir in directories if re.match("^Row._Well[0-9]", _dir).group() in wells ]
            
            #create .h5 dataset to which all results are written
            path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, self.DEFAULT_INPUT_IMAGE_NAME)
            
            with h5py.File(path, 'w') as hf:
                dt = h5py.special_dtype(vlen=str)
                hf.create_dataset('label_names', (len(column_labels)), chunks=None, dtype = dt)
                hf.create_dataset('labels', (len(directories)*n_timepoints, len(column_labels)), chunks=None, dtype = dt)
                hf.create_dataset('input_images', (len(directories)*n_timepoints, n_channels, img_size, img_size), chunks=(1, 1, img_size, img_size))
                
                label_names = hf.get("label_names")
                labels = hf.get("labels")
                input_images = hf.get("input_images")
                
                label_names[:]= column_labels

                #------------------
                #start reading data
                #------------------
                
                indexes = []
                #create indexes
                start_index = 0
                for i, _ in enumerate(directories):
                    stop_index = start_index + n_timepoints
                    indexes.append((start_index, stop_index))
                    start_index = stop_index
                
                #iterate through all directories and add to .h5
                #this is not implemented with multithreaded processing because writing multi-threaded to hdf5 is hard
                #multithreaded reading is easier

                for dir, index in tqdm(zip(directories, indexes), total = len(directories)):
                    _read_write_images(dir, index, h5py_path = path)
    
    def segment(self, 
                overwrite = False,
                *args, 
                **kwargs):
        
        """segment project with the defined segmentation under segmentation_f.
        
        Args:
            intermediate_output (bool, optional): Can be set when calling to override the project wide flag.

            debug (bool, optional):Can be set when calling to override the project wide flag.

            overwrite (bool, optional): Can be set when calling to override the project wide flag.
        """

        #add possibility to delete only segmentation while preserving input_images already written to hdf5

        if overwrite == True:

            #delete segmentation and classes from .hdf5 to be able to create new again
            path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, self.DEFAULT_INPUT_IMAGE_NAME)
            with h5py.File(path, 'a') as hf:
                if "segmentation" in hf.keys():
                    del hf["segmentation"]
                if "classes" in hf.keys():
                    del hf["classes"]

            #delete generated files to make clean
            classes_path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv")
            log_path = os.path.join(self.directory, self.DEFAULT_SEGMENTATION_DIR_NAME, "processing.log")
            if os.path.isfile(classes_path):
                os.remove(classes_path)
            if os.path.isfile(log_path):
                os.remove(log_path)
            
            print("If Segmentation already existed removed.")

        if self.segmentation_f == None:
            raise ValueError("No segmentation method defined")
            
        else:
            self.segmentation_f(*args, **kwargs)

    def extract(self, 
                *args, 
                **kwargs):
        """extract single cells with the defined extraction at extraction_f.
        
        Args:
            intermediate_output (bool, optional): Can be set when calling to override the project wide flag.

            debug (bool, optional):Can be set when calling to override the project wide flag.

            overwrite (bool, optional): Can be set when calling to override the project wide flag.
        """
        
        if self.extraction_f == None:
            raise ValueError("No extraction method defined")
            
        input_segmentation = self.segmentation_f.get_output()
        input_dir = os.path.join(self.project_location, self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv")
        self.extraction_f(input_segmentation,  input_dir, *args, **kwargs)