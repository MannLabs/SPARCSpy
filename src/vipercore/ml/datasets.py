from torch.utils.data import Dataset
import torch
import numpy as np
import random
import os
import h5py

class NPYSingleCellDataset(Dataset):
    """Summary line.

    Extended description of function.

    Args:
        arg1 (int): Description of arg1
        arg2 (str): Description of arg2

    Returns:
        bool: Description of return value

    """
    def __init__(self, dir_list, dir_labels, root_dir, max_level=5, transform=None, return_id=False, return_fake_id=False):
        
        self.root_dir = root_dir
        self.dir_labels = dir_labels
        self.dir_list = dir_list
        self.transform = transform
        
        # contains a list of all data
        # each element contains the follwoing information:
        # [label, filename without extension, directory index]
        self.data_locator = []
        
        # contains the directories, where data can be found as specified in directory index
        self.dir_dic = []
        
        # scan all directoreis
        for i, directory in enumerate(dir_list):
            path = os.path.join(self.root_dir,directory)
            current_label = self.dir_labels[i]
            
            # recursively scan for files
            self.scan_directory(path, current_label, max_level)
        
        # print dataset stats at the end
        
        self.return_id = return_id
        self.return_fake_id = return_fake_id
        self.stats()
        
    def scan_directory(self, path, current_label, levels_left):
        if levels_left > 0:
            
            # get files and directories at current level
            input_list = os.listdir(path)
            current_level_directories = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
            current_level_files = [ name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
            
            # append current level files
            self.dir_dic.append(path)
            for i, file in enumerate(current_level_files):
                filetype = file.split(".")[-1]
                filename = file.split(".")[0]
                
                if filetype == "npy":
                    self.data_locator.append([current_label,filename,len(self.dir_dic)-1])
                    
            # recursively scan subdirectories        
            for subdirectory in current_level_directories:
                self.scan_directory(subdirectory, current_label, levels_left-1)
            
        else:
            return
        
    def stats(self):
    
        labels = [el[0] for el in self.data_locator]
        
        print("Total: {}".format(len(labels)))
        
        for l in set(labels):
            print("{}: {}".format(l,labels.count(l)))
        
    def __len__(self):
        return len(self.data_locator)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get the label, filename and directory for the current dataset
        data_info = self.data_locator[idx]
        
        # create file path from this information
        dir_path = self.dir_dic[data_info[-1]]
        file_path = os.path.join(dir_path,data_info[1]+".npy")
        
        # load and shape tensor
        tensor = np.load(file_path)        
        

        t = torch.from_numpy(tensor)
        t = t.float()
        t = t.permute(2, 0, 1)
        
        
        if self.transform:
            t = self.transform(t)
            
        if not list(t.shape) == list(torch.Size([1,128,128])):
            t = torch.zeros((1,128,128))
            
        if self.return_id and self.return_fake_id:
            raise ValueError("either return_id or return_fake_id should be set")
            
        if self.return_id:
            
            ids = int(data_info[1])
            sample = (t, torch.tensor(data_info[0]), torch.tensor(ids))
        elif self.return_fake_id:
            
            sample = (t, torch.tensor(data_info[0]), torch.tensor(0))
        else:
            sample = (t, torch.tensor(data_info[0]))
        
        
        return sample
    
class HDF5SingleCellDataset(Dataset):
    
    HDF_FILETYPES = ["hdf", "hf", "h5"]
    def __init__(self, dir_list, 
                 dir_labels, 
                 root_dir, 
                 max_level=5, 
                 transform=None, 
                 return_id=False, 
                 return_fake_id=False,
                 select_channel=None):
        
        self.root_dir = root_dir
        self.dir_labels = dir_labels
        self.dir_list = dir_list
        self.transform = transform
        
        self.handle_list = []
        self.data_locator = []
        
        self.select_channel = select_channel
        
        # scan all directoreis
        for i, directory in enumerate(dir_list):
            path = os.path.join(self.root_dir, directory)  
            current_label = self.dir_labels[i]

            #check if "directory" is a path to specific hdf5
            filetype = directory.split(".")[-1]
            filename = directory.split(".")[0]
                
            if filetype in self.HDF_FILETYPES:
                self.add_hdf_to_index(current_label, directory)

            else:
                # recursively scan for files
                self.scan_directory(path, current_label, max_level)
        
        # print dataset stats at the end
        
        self.return_id = return_id
        self.return_fake_id = return_fake_id
        self.stats()
 
        
    def add_hdf_to_index(self, current_label, path):       
        try:
            input_hdf = h5py.File(path, 'r')
        
            index_handle = input_hdf.get('single_cell_index')

            handle_id = len(self.handle_list)
            self.handle_list.append(input_hdf.get('single_cell_data'))

            for row in index_handle:
                self.data_locator.append([current_label,handle_id]+list(row))      
        except:
            return
        
    def scan_directory(self, path, current_label, levels_left):
        
        # iterates over all files and folders in a directory
        # hdf5 files are added to the index
        # subfolders are recursively scanned
        
        if levels_left > 0:
            
            # get files and directories at current level
            input_list = os.listdir(path)
            current_level_directories = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

            current_level_files = [ name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
                        
            for i, file in enumerate(current_level_files):
                filetype = file.split(".")[-1]
                filename = file.split(".")[0]
                
                if filetype in self.HDF_FILETYPES:
                    
                    self.add_hdf_to_index(current_label, os.path.join(path, file))
                    
            # recursively scan subdirectories        
            for subdirectory in current_level_directories:
                self.scan_directory(subdirectory, current_label, levels_left-1)
            
        else:
            return
        
    def stats(self):
    
        labels = [el[0] for el in self.data_locator]
        
        print("Total: {}".format(len(labels)))
        
        for l in set(labels):
            print("{}: {}".format(l,labels.count(l)))
        
    def __len__(self):
        return len(self.data_locator)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get the label, filename and directory for the current dataset
        data_info = self.data_locator[idx]
        
        if self.select_channel is not None:
            cell_tensor = self.handle_list[data_info[1]][data_info[2], self.select_channel]
            t = torch.from_numpy(cell_tensor)
            t = torch.unsqueeze(t,0)
            
        else:
            
            cell_tensor = self.handle_list[data_info[1]][data_info[2]]
            t = torch.from_numpy(cell_tensor)
            
        t = t.float()     
        
        if self.transform:
            t = self.transform(t)
        """  
        if not list(t.shape) == list(torch.Size([1,128,128])):
            t = torch.zeros((1,128,128))
        """      
        if self.return_id and self.return_fake_id:
            raise ValueError("either return_id or return_fake_id should be set")
            
        if self.return_id:
            
            ids = int(data_info[3])
            sample = (t, torch.tensor(data_info[0]), torch.tensor(ids))
        elif self.return_fake_id:
            
            sample = (t, torch.tensor(data_info[0]), torch.tensor(0))
        else:
            sample = (t, torch.tensor(data_info[0]))
        
        return sample