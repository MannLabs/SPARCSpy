import os
import shutil

class OperaPhenixDataloader:
    """
    Dataloader for experiments imaged on the opera Phenix
    
    creates the output folder structure annd
    returns a list of dicts with {input: [input paths], output: output directory}
    
    order of the input paths is
    input[0]: dapi
    input[1]: cellmask
    input[2]: golgi
    
    """
    
    golgi_key = "ch4sk1fk1fl1"
    dapi_key = "ch3sk1fk1fl1"
    cellmask_key = "ch2sk1fk1fl1"

    filetype = "tiff"

    separator = "-"
    
    def __init__(self, input_dir="", output_dir="", skip_existing=False, ):
        
        self.queue = []
        
        if input_dir=="":
            raise KeyError("input directory can't be empty string \"\"")
        
        # start with base level and get all directories at the base level
        current_level_directories = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
        current_level_directories = [ el.split("/")[-1] for el in current_level_directories]
        
        # check and create all directories in the output folder
        for rel_in_directory in current_level_directories:
            abs_in_directory = os.path.join(input_dir,rel_in_directory)
            
            abs_out_directory = os.path.join(output_dir,rel_in_directory)
            
            if os.path.exists(abs_out_directory):
                if skip_existing:
                    pass
                else:
                    shutil.rmtree(abs_out_directory)
                
            if not os.path.exists(abs_out_directory):
                os.makedirs(abs_out_directory)
                
                
            self.scan_experiment(abs_in_directory,abs_out_directory)   
            
        #print(self.queue)            

    def __iter__(self):
        return iter(self.queue)

    def __next__(self):
        return next(self.queue)
        
    def __len__(self):
        return len(self.queue)
    
    def scan_experiment(self, input_dir, output_dir):
        
        # check wether Image/ folder exists in experiment
        experiment = input_dir.split("/")[-1]
        input_dir = os.path.join(input_dir, "Images")
        if not os.path.exists(input_dir):
            print("no Images/ folder found for experiment ", experiment)
            return
        
        file_list = os.listdir(input_dir)
        
        # filter all images based on filetype and remove channel info
        filtered_file_list = []
        for file in file_list:
            extension = file.split(".")[-1]
            
            if extension == self.filetype:
                
                keys = file.split(self.separator)
                filtered_file_list.append(keys[0])
        
        filtered_file_list = list(set(filtered_file_list))
        
        # output list of images with combined channel info
        for key in filtered_file_list:
            plate_position = key[:6]
            
            # create well folder if not present
            well_path = os.path.join(output_dir,plate_position)
            if not os.path.exists(well_path):
                os.makedirs(well_path)
            
            dapi_path = input_dir + "/" + key + self.separator + self.dapi_key + "." + self.filetype
            cellmask_path = input_dir + "/" + key + self.separator + self.cellmask_key + "." + self.filetype
            golgi_path = input_dir + "/" + key + self.separator + self.golgi_key + "." + self.filetype
            
            if os.path.exists(dapi_path) and os.path.exists(cellmask_path) and os.path.exists(golgi_path):
                
                obj = {}
                obj["input"] = [dapi_path, cellmask_path, golgi_path]
                obj["output"] = well_path
                
                self.queue.append(obj)