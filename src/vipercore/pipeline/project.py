import warnings
import shutil
import os
import yaml
from PIL import Image
import PIL
import numpy as np



class Project:
    # default name for the config file. Name is changed when config file is copied. Name needs to used when config is created manually in dataset folder
    DEFAULT_CONFIG_NAME = "config.yml"
    DEFAULT_SEGMENTATION_DIR_NAME = "segmentation"
    DEFAULT_EXTRACTION_DIR_NAME = "extraction"
    DEFAULT_CLASSIFICATION_DIR_NAME = "classification"
    
    # Project object is initialized, nothing is written to disk
    def __init__(self, 
                 location_path, 
                 config_path =  "",
                 intermediate_output = True,
                 segmentation_f = None,
                 extraction_f = None,
                 classification_f = None,
                 debug = False,
                 overwrite = False):
        
        self.segmentation_f = segmentation_f
        self.extraction_f = extraction_f
        self.classification_f = classification_f
        
        PIL.Image.MAX_IMAGE_PIXELS = 10000000000
        
        self.input_image = None
        self.config = None
        
        # handle location
            
        self.dataset_location = location_path
        
    def create(self, 
               debug = False,
               overwrite = False):
        
        if not os.path.isdir(self.dataset_location):
                os.makedirs(self.dataset_location)
        else:
            warnings.warn("Theres already a directory in the location path")
                
        # handle configuration file
        new_config_path = os.path.join(self.dataset_location,self.DEFAULT_CONFIG_NAME)
        if config_path == "":
            
            # Check if there is already a config file in the dataset folder in case no config file has been specified
            
            if os.path.isfile(new_config_path):
                self.load_config_from_file(new_config_path)
            
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
                    
                self.load_config_from_file(new_config_path)
        
        # === setup segmentation ===
        if self.segmentation_f is not None:
            seg_directory = os.path.join(self.dataset_location, self.DEFAULT_SEGMENTATION_DIR_NAME)
            self.segmentation_f = segmentation_f(self.config[segmentation_f.__name__], 
                                                 seg_directory,
                                                 debug = self.debug,
                                                 overwrite = self.overwrite,
                                                 intermediate_output = self.intermediate_output)
        else:
            self.segmentation_f = None
            
        # === setup extraction ===
        if extraction_f is not None:
            extraction_directory = os.path.join(self.dataset_location, self.DEFAULT_EXTRACTION_DIR_NAME)
            self.extraction_f = extraction_f(self.config["extraction"], 
                                                 extraction_directory,
                                                 debug = self.debug,
                                                 overwrite = self.overwrite,
                                                 intermediate_output = self.intermediate_output)
        else:
            self.extraction_f = None
            
        # === setup classification ===
        if classification_f is not None:
            classification_directory = os.path.join(self.dataset_location, self.DEFAULT_CLASSIFICATION_DIR_NAME)
            self.classification_f = classification_f(self.config["classification"], 
                                                 classification_directory,
                                                 debug = self.debug,
                                                 overwrite = self.overwrite,
                                                 intermediate_output = self.intermediate_output)
        else:
            self.classification_f = None
        
                
    def load_config_from_file(self, file_path):
        
        if not os.path.isfile(file_path):
            raise ValueError("Your config path is invalid")
            
        with open(file_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
    def load_input_from_file(self, file_paths, remap = None, crop=[(0,-1),(0,-1)]):
        if self.config == None:
            raise ValueError("Dataset has no config file loaded")
            
        # input data is NOT copied to the project folder
        # remap can be used to shuffle the order, for example [1, 0, 2] to invert the first two channels
        # Expects a list of file_paths with config.input_channel elements           
        
        if not len(file_paths) == self.config["input_channels"]:
            raise ValueError("Expected {} image paths, only received {}".format(self.config["input_channels"], len(file_paths)))
        
        # append all images channel wise and remap them according to the supplied list
        channels = []
        
        for channel_path in file_paths:
            im = Image.open(channel_path)
            c = np.array(im, dtype="float32")[slice(*crop[0]),slice(*crop[1])]
            
            channels.append(c)
        
        self.input_image = np.stack(channels)
        
        print(self.input_image.shape)
        
        if remap != None:
            self.input_image = self.input_image[remap]
        
        
    def load_input_from_array(self, array, remap = None):
        # input data is not copied to the project folder
        if self.config == None:
            raise ValueError("Dataset has no config file loaded")
            
        if not array.shape[0] == self.config["input_channels"]:
            raise ValueError("Expected {} image paths, only received {}".format(self.config["input_channels"], array.shape[0]))
            
        self.input_image = array
            
        if remap != None:
            self.input_image = self.input_image[remap]
            
    def segment(self, *args, **kwargs):
        if self.segmentation_f == None:
            raise ValueError("No segmentation method defined")
            
        elif type(self.input_image) == None:
            raise ValueError("No input image defined")
        else:
            self.segmentation_f(self.input_image, *args, **kwargs)
            
    def extract(self, *args, **kwargs):
        if self.extraction_f == None:
            raise ValueError("No extraction method defined")
            
        input_segmentation = self.segmentation_f.get_output_file_path()
        if not os.path.isfile(input_segmentation):
            raise ValueError("Segmentation was not found at {}".format(input_segmentation))
        
        input_dir = os.path.join(self.dataset_location,self.DEFAULT_SEGMENTATION_DIR_NAME, "classes.csv")
        self.extraction_f(input_segmentation,  input_dir, *args, **kwargs)
        
    def classify(self, *args, **kwargs):
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
        
            
            
            