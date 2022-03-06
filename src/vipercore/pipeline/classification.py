from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import skfmm
import csv
from functools import partial
from multiprocessing import Pool
import h5py
import torch

from skimage.filters import gaussian, median
from skimage.morphology import binary_erosion, disk, dilation
from skimage.segmentation import watershed
from skimage.color import label2rgb

from scipy.ndimage import binary_fill_holes

from vipercore.processing.utils import plot_image

from vipercore.ml.datasets import NPYSingleCellDataset, HDF5SingleCellDataset
from vipercore.ml.transforms import RandomRotation, GaussianNoise, ChannelReducer, ChannelSelector
from vipercore.ml.plmodels import GeneralModel, MultilabelSupervisedModel

from torchvision import transforms, utils

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

import json
import _pickle as cPickle
import io
from contextlib import redirect_stdout

class MLClusterClassifier:
    
    
    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_DATA_DIR = "data"
    CLEAN_LOG = True
    
    def __init__(self, 
                 config, 
                 path, 
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
        
        
        
        # Create segmentation directory
        self.directory = path
        if not os.path.isdir(self.directory):
            
            os.makedirs(self.directory)
        
            
        # Set up log and clean old log
        if self.CLEAN_LOG:
            log_path = os.path.join(self.directory, self.DEFAULT_LOG_NAME)
            if os.path.isfile(log_path):
                os.remove(log_path) 
                
        # check latest cluster run 
        current_level_directories = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        runs = [int(i) for i in current_level_directories if self.is_Int(i)]
        
        self.current_run = max(runs) +1 if len(runs) > 0 else 0
        self.run_path = os.path.join(self.directory, str(self.current_run))
        
        if not os.path.isdir(self.run_path):
            os.makedirs(self.run_path)
            self.log("Created new directory " + self.run_path)
            
        self.log(f"current run: {self.current_run}")
            
    
    def is_Int(self, s):
        try: 
            int(s)
            return True
        except ValueError:
            return False
                
    def get_timestamp(self):
        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")  
        return "[" + dt_string + "] "
    
    def log(self, message):
        
        log_path = os.path.join(self.run_path, self.DEFAULT_LOG_NAME)
        
        if isinstance(message, str):
            lines = message.split("\n")
            
        if isinstance(message, list):
            lines = message
            
        if isinstance(message, dict):
            
            lines = []
            for key, value in message.items():
                lines.append(f"{key}: {value}")                       
        
        for line in lines:
                with open(log_path, "a") as myfile:
                    myfile.write(self.get_timestamp() + line +" \n")
                
                if self.debug:
                    print(self.get_timestamp() + line)
        
        
    def __call__(self, 
                 extraction_dir, 
                 accessory, 
                 size=0, 
                 project_dataloader=HDF5SingleCellDataset, 
                 accessory_dataloader=HDF5SingleCellDataset):
        
        # is called with the path to the segmented image
        # Size: number of datapoints of the project dataset considered
        # ===== Dataloaders =====
        # should be either GolgiNetDataset for single .npy files or HDF5SingleCellDataset for .h5 datasets
        # project_dataloader: dataloader for the project dataset
        # accessory_dataloader: dataloader for the accesssory datasets
        
        self.log("Started classification")
        self.log(f"starting with run {self.current_run}")
        self.log(self.config)
        
        accessory_sizes, accessory_labels, accessory_paths = accessory
        
        self.log(f"{len(accessory_sizes)} different accessory datasets specified")
        
        
        # Load model and parameters
        network_dir = self.config["network"]
        checkpoint_path = os.path.join(network_dir,"checkpoints")
        checkpoints = current_level_files = [ name for name in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, name))]
        checkpoints.sort()

        if len(checkpoints) < 1:
            raise ValueError(f"No model parameters found at: {self.config['network']}")
        
        #ensure that the most recent version is used if more than one is saved
        if len(checkpoints) > 1:
            #get max epoch number 
            epochs = [int(x.split("epoch=")[1].split("-")[0]) for x in checkpoints]
            max_value = max(epochs)
            max_index = epochs.index(max_value)
            self.log(f"Maximum epoch number found {max_value}")

            #get checkpoint with the max epoch number
            latest_checkpoint_path = os.path.join(checkpoint_path, checkpoints[max_index])
        else:
            latest_checkpoint_path = os.path.join(checkpoint_path, checkpoints[0])
        
        #add log message to ensure that it is always 100% transparent which classifier is being used
        self.log(f"Using the following classifier checkpoint: {latest_checkpoint_path}")
        hparam_path = os.path.join(network_dir,"hparams.yaml")
        
        model = MultilabelSupervisedModel.load_from_checkpoint(latest_checkpoint_path, hparams_file=hparam_path)
        model.eval()
        model.to(self.config['inference_device'])
        
        self.log(f"model parameters loaded from {self.config['network']}")
        
        # generate project dataset dataloader
        # transforms like noise, random rotations, channel selection are still hardcoded
        t = transforms.Compose([ChannelSelector([self.config["channel_classification"]]),
                        RandomRotation()])
                
        self.log(f"loading {extraction_dir}")
        
        # redirect stdout to capture dataset size
        f = io.StringIO()
        with redirect_stdout(f):
            dataset = HDF5SingleCellDataset([extraction_dir],[0],"/",transform=t,return_id=True)
            
            if size == 0:
                size = len(dataset)
            residual = len(dataset) - size
            dataset, _ = torch.utils.data.random_split(dataset, [size, residual])

        
        # Load accessory dataset
        for i in range(len(accessory_sizes)):
            self.log(f"loading {accessory_paths[i]}")
            with redirect_stdout(f):
                local_dataset = HDF5SingleCellDataset([accessory_paths[i]], 
                                           [i+1], 
                                           "/",
                                           transform=t,
                                           return_fake_id=True)
                
            
            
            if len(local_dataset) > accessory_sizes[i]:
                residual = len(local_dataset) - accessory_sizes[i]
                local_dataset, _ = torch.utils.data.random_split(local_dataset, [accessory_sizes[i], residual])
            
            dataset = torch.utils.data.ConcatDataset([dataset,local_dataset])
        
        # log stdout 
        out = f.getvalue()
        self.log(out)
            
        # classify samples
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=self.config["batch_size"], num_workers=self.config["dataloader_worker"], shuffle=True)
        
        self.log(f"log transfrom: {self.config['log_transform']}")
        
        self.inference(dataloader, model.network.encoder_c2)
        self.inference(dataloader, model.network.forward)
        
        
        
        
    def inference(self, 
                  dataloader, 
                  model_fun):
        # 1. performs inference for a dataloader and a given network call
        # 2. performs a dimension reduction on the data
        
        data_iter = iter(dataloader)        
        self.log(f"start processing {len(data_iter)} batches with {model_fun.__name__} based inference")
        with torch.no_grad():

            x, label, class_id = data_iter.next()
            r = model_fun(x.to(self.config['inference_device']))
            result = r.cpu().detach()

            for i in range(len(dataloader)-1):
                if i % 10 == 0:
                    self.log(f"processing batch {i}")
                
                x, l, id = data_iter.next()

                r = model_fun(x.to(self.config['inference_device']))
                result = torch.cat((result, r.cpu().detach()), 0)
                label = torch.cat((label, l), 0)
                class_id = torch.cat((class_id, id), 0)

        result = result.detach().numpy()
        
        #if self.config["exp_transform"]:
        #    result = np.exp(result)
            
        if self.config["log_transform"]:
            sigma = 1e-9
            result = np.log(result+sigma)
        
        label = label.numpy()
        class_id = class_id.numpy()
        
        # save inferred activations / predictions
        result_labels = [f"result_{i}" for i in range(result.shape[1])]
        
        # ===== dimension reduction =====
        
        if self.config["standard_scale"]:
            result = StandardScaler().fit_transform(result)

        self.log(f"start first pca")
        d1, d2 = result.shape
        pca = PCA(n_components=min(d2, self.config["pca_dimensions"]))
        embedding_pca = pca.fit_transform(result)
        
        # save pre dimension reduction pca results
        pca_labels = [f"hd_pca_{i}" for i in range(embedding_pca.shape[1])]
        
        
        print(result.shape)
        print(embedding_pca.shape)
        
        design_labels = result_labels + pca_labels
        design = np.concatenate((result, embedding_pca), axis=1)

        embedding_2_pca = PCA(n_components=2).fit_transform(result)

        #self.log(f"start umap")        
        #reducer = umap.UMAP(n_neighbors=self.config["umap_neighbours"], min_dist=self.config["umap_min_dist"], n_components=2,metric='cosine')
        #embedding_umap = reducer.fit_transform(embedding_pca)
        
        #self.log(f"start tsne")
        #embedding_tsne = TSNE(n_jobs=self.config["threads"]).fit_transform(embedding_pca)
        
        dataframe = pd.DataFrame(data=design, columns=design_labels)
        
        self.log(f"finished processing")

        dataframe["label"] = label
        dataframe["cell_id"] = class_id.astype("int")
        dataframe["pca_0"] = embedding_2_pca[:,0]
        dataframe["pca_1"] = embedding_2_pca[:,1]
        #dataframe["umap_0"] = embedding_umap[:,0]
        #dataframe["umap_1"] = embedding_umap[:,1]
        #dataframe["tsne_0"] = embedding_tsne[:,0]
        #dataframe["tsne_1"] = embedding_tsne[:,1]
        
        path = os.path.join(self.run_path,f"dimension_reduction_{model_fun.__name__}.tsv")
        dataframe.to_csv(path)
