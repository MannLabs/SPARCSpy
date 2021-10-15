# -*- coding: utf-8 -*-


import warnings
import shutil
import os
import yaml
from PIL import Image
import PIL
import numpy as np

class Project:
    """
    Project base class used to create a new viper project.
    
    Args:
        location_path (str): Path to the folder where to project should be created. The folder is created in case the specified folder does not exist.
        config_path (str, optional): Path pointing to a valid configuration file. The file will be copied to the project directory and renamed to the name specified in ``DEFAULT_CLASSIFICATION_DIR_NAME``. See the section configuration to find out more about the config file. 
        intermediate_output (bool, optional, default ``False``): When set to True intermediate outputs will be saved where applicable.
            
        debug (bool, optional, default ``False``): When set to True debug outputs will be printed where applicable. 
            
        overwrite (bool, optional, default ``True``): When set to False intermediate outputs will be loaded.
            
        segmentation_f (Class, optional, default ``None``): Class containing segmentation workflow.
            
        extraction_f (Class, optional, default ``None``): Class containing extraction workflow.
            
        classification_f (Class, optional, default ``None``): Class containing classification workflow.
            
        selection_f (Class, optional, default ``None``): Class containing selection workflow.
            
    Attributes:
        DEFAULT_CONFIG_NAME (str): Default config name which is used for the config file in the project directory. This name needs to be used when no config is supplied and the config is manually created in the project folder.
        DEFAULT_SEGMENTATION_DIR_NAME (str): Default foldername for the segmentation process.
        DEFAULT_EXTRACTION_DIR_NAME (str): Default foldername for the segmentation process.
        DEFAULT_CLASSIFICATION_DIR_NAME (str): Default foldername for the classification process.
        DEFAULT_SELECTION_DIR_NAME (str, default "selection"): Default foldername for the selection process.
    
    """

    DEFAULT_CONFIG_NAME = "config.yml"
    DEFAULT_SEGMENTATION_DIR_NAME = "segmentation"
    DEFAULT_EXTRACTION_DIR_NAME = "extraction"
    DEFAULT_CLASSIFICATION_DIR_NAME = "classification"
    DEFAULT_SELECTION_DIR_NAME = "Selection"
    
    # Project object is initialized, nothing is written to disk
    def __init__(self, 
                 location_path,
                 config_path =  "",
                 intermediate_output = False,
                 debug = False,
                 overwrite = True,
                 segmentation_f = None,
                 extraction_f = None,
                 classification_f = None,
                 selection_f = None
                 ):
        
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
                
                # check if there is an old config file
                if os.path.isfile(new_config_path):               
                    if overwrite:   
                        # The blueprint config file is copied to the dataset folder and renamed to the default name
                        shutil.copyfile(config_path, new_config_path)
                        
                    else:
                        warnings.warn(f"You specified a new config file but didnt specify overwrite. The existing file will be loaded. Did you mean create(overwrite=True)?")
                else:
                    # The blueprint config file is copied to the dataset folder and renamed to the default name
                    shutil.copyfile(config_path, new_config_path)
                    
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
        
                
    def _load_config_from_file(self, file_path):
        """
        loads config from file and writes it to self.config
        """
        if not os.path.isfile(file_path):
            raise ValueError("Your config path is invalid")
            
        with open(file_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
    def load_input_from_file(self, file_paths, remap = None, crop=[(0,-1),(0,-1)]):
        """load input image from a number of files.

        Args:
            file_paths (list(str)): List containing paths to each channel like ``["path1/img.tiff", "path2/img.tiff", "path3/img.tiff"]``. Expects a list of file paths with length ``input_channel`` as defined in the config.yml. Input data is NOT copied to the project folder by default. Different segmentation functions especially tiled segmentations might copy the input
            
            remap (list(int), optional): Define remapping of channels. For example use ``[1, 0, 2]`` to change the order of the first and the second channel. 

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
        
        if remap != None:
            self.input_image = self.input_image[remap]
        
        
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
            
        if remap != None:
            self.input_image = self.input_image[remap]
            
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
            
        input_segmentation = self.segmentation_f.get_output_file_path()
        if not os.path.isfile(input_segmentation):
            raise ValueError("Segmentation was not found at {}".format(input_segmentation))
        
        input_dir = os.path.join(self.project_location,self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv")
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
                 
        if self.classification_f == None:
            raise ValueError("No classification method defined")
            pass
            
        input_extraction = self.extraction_f.get_output_path()
        

        if not os.path.isdir(input_extraction):
            raise ValueError("input was not found at {}".format(input_extraction))
            
        self.classification_f(input_extraction, *args, **kwargs)
            
    def process(self):
        self.segment()
        self.extract()
<<<<<<< HEAD
        
            
            
            
=======
        self.classify()
>>>>>>> update and documentation of project base class
