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

from vipercore.pipeline.base import Logable

class Shard(Logable):
    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_OUTPUT_FILE = "segmentation.h5"
    DEFAULT_FILTER_FILE = "classes.csv"
    
    def __init__(self, 
                 shard_id,
                 shard_window,
                 config,
                 folder_path, 
                 input_path,
                 debug=False, 
                 overwrite=False,
                 intermediate_output = False):
        
        super().__init__()
        self.log_path = os.path.join(folder_path, self.DEFAULT_LOG_NAME)
        
        self.config = config
        self.debug = debug
        self.overwrite = overwrite
        self.config = config
        self.intermediate_output = intermediate_output
        
        
        self.shard_window = shard_window
        self.segmentation_directory = folder_path
        self.input_path = input_path
        self.id = shard_id
        
    def load_maps_from_disk(self):       
        # iterating over all maps
        for map_index, map_name in enumerate(self.maps.keys()):

            
            try:
                map_path = os.path.join(self.segmentation_directory, "{}_{}_map.npy".format(map_index, map_name))

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
        self.start_from = np.argmin(is_not_none) if not all(is_not_none) else len(is_not_none)
        
    def save_image(self, array, save_name="", cmap="magma",**kwargs):
        if self.debug:
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
                
    def _normalize(self, im, q1, q2):
        
        if len(im.shape) == 2:
            lower_quantile = np.quantile(np.ravel(im),q1)
            upper_quantile = np.quantile(np.ravel(im),q2)

            IQR = upper_quantile - lower_quantile


            im -= lower_quantile 
            im = im / IQR
            im = np.clip(im, 0, 1)
            return im
        if len(im.shape) == 3:
            for i, channel in enumerate(im):

                lower_quantile = np.quantile(np.ravel(im[i]),q1)
                upper_quantile = np.quantile(np.ravel(im[i]),q2)

                IQR = upper_quantile - lower_quantile


                im[i] -= lower_quantile 
                im[i] = im[i] / IQR
                im[i] = np.clip(im[i], 0, 1)
            return im
    
    def normalize(self, im):
        
        for i, channel in enumerate(im):
            
            lower_quantile = np.quantile(np.ravel(im[i]),self.config["lower_quantile_normalization"])
            upper_quantile = np.quantile(np.ravel(im[i]),self.config["upper_quantile_normalization"])
                                         
            IQR = upper_quantile - lower_quantile

                                         
            im[i] -= lower_quantile 
            im[i] = im[i] / IQR
            im[i] = np.clip(im[i], 0, 1)
        return im
    
    def save_map(self, map_name):
        if self.maps[map_name] is None:
            self.log("Error saving map {}, map is None".format(map_name))
        else:
            map_index = list(self.maps.keys()).index(map_name)
            
            if self.intermediate_output:
                map_path = os.path.join(self.segmentation_directory, "{}_{}_map.npy".format(map_index, map_name))
                np.save(map_path, self.maps[map_name])
                self.log("Saved map {} {} under path {}".format(map_index,map_name, map_path))
            
            # check if map contains more than one channel (3, 1024, 1024) vs (1024, 1024)
            
            if len(self.maps[map_name].shape) > 2:
                
                for i, channel in enumerate(self.maps[map_name]):
                    
                    
                    
                    channel_name = "{}_{}_{}_map.png".format(map_index, map_name,i)
                    channel_path = os.path.join(self.segmentation_directory, channel_name)
                    self.save_image(channel, save_name = channel_path)
            else:
                channel_name = "{}_{}_map.png".format(map_index, map_name)
                channel_path = os.path.join(self.segmentation_directory, channel_name)

                self.save_image(self.maps[map_name], save_name = channel_path)
                
    def visualizue_class(self, class_ids, seg_map, background, *args, **kwargs):
        index = np.argwhere(class_ids==0.)
        class_ids_no_zero = np.delete(class_ids, index)

        outmap_map = np.where(np.isin(seg_map,class_ids_no_zero), 2, seg_map)
        outmap_map = np.where(np.isin(seg_map,class_ids, invert=True), 1,outmap_map)
        
        image = label2rgb(outmap_map,background/np.max(background),alpha=0.4, bg_label=0)
        plot_image(image, save_name="contact_fitler.png", **kwargs)
        
    def segment(self):
        warnings.filterwarnings("ignore")

        if not os.path.isdir(self.segmentation_directory):
            os.makedirs(self.segmentation_directory)
            
        map_path = os.path.join(self.segmentation_directory, self.DEFAULT_OUTPUT_FILE)
        
        
        if os.path.isfile(map_path) and not self.overwrite:
            self.log(f"Exit, segmentation map exists {map_path}")
            return
        
        self.maps = {"normalized": None,
                     "median": None,
                    "nucleus_segmentation": None,
                    "nucleus_mask": None,
                    "wga_mask":None,
                    "wga_potential": None,
                    "travel_time":None,
                    "watershed":None}
        
        
        
        
        if not self.overwrite:
            self.load_maps_from_disk()
        
        self.log(f"Segmentation started shard {self.id}, starting from checkpoint {self.start_from}")
        
        # Normalization
        if self.start_from <= 0:
            self.log("Started with normalized map")
            
            hf = h5py.File(self.input_path, 'r')
            hdf_input = hf.get('channels')
            image = hdf_input[:,self.shard_window[0],self.shard_window[1]]
            hf.close()
            
            self.maps["normalized"] = self.normalize(image)
            self.save_map("normalized")
            
            self.log("Normalized map created")
            
            
            
        # Median calculation
        if self.start_from <= 1:
            self.log("Started with median map")
            
            self.maps["median"] = np.copy(self.maps["normalized"])
                                   
            for i, channel in enumerate(self.maps["median"]):
                self.maps["median"][i] = median(channel, disk(self.config["median_filter_size"]))
            
            self.save_map("median")
            
            self.log("Median map created")
            
         # segment dapi channels based on local tresholding
        if self.debug:
            #plt.style.use("dark_background")
            plt.hist(self.maps["median"][2].flatten(),bins=100,log=False)
            plt.xlabel("intensity")
            plt.ylabel("frequency")
            plt.savefig("nuclei_frequency.png")
            plt.show()
            
       
        if self.start_from <= 2:
            self.log("Started with nucleus segmentation map")
            
            nucleus_map_tr = self._normalize(self.maps["median"][2],0.03,0.97)
            plot_image(nucleus_map_tr)
            self.maps["nucleus_segmentation"] = segment_local_tresh(nucleus_map_tr, 
                                         dilation=self.config["nucleus_segmentation"]["dilation"], 
                                         thr=self.config["nucleus_segmentation"]["threshold"], 
                                         median_block=self.config["nucleus_segmentation"]["median_block"], 
                                         min_distance=self.config["nucleus_segmentation"]["min_distance"], 
                                         peak_footprint=self.config["nucleus_segmentation"]["peak_footprint"], 
                                         speckle_kernel=self.config["nucleus_segmentation"]["speckle_kernel"], 
                                         debug=self.debug)
            
            del nucleus_map_tr
            self.save_map("nucleus_segmentation")

            self.log("Nucleus segmentation map created")
        
        # Calc nucleus map
        
        if self.start_from <= 3:
            self.log("Started with nucleus mask map")
            self.maps["nucleus_mask"] = np.clip(self.maps["nucleus_segmentation"], 0,1)
            
            self.save_map("nucleus_mask")
            self.log("Nucleus mask map created with {} elements".format(np.max(self.maps["nucleus_segmentation"])))
            
        
        # filter nuclei based on size and contact
        center_nuclei, length, coords = mask_centroid(self.maps["nucleus_segmentation"], debug=self.debug)
        
        all_classes = np.unique(self.maps["nucleus_segmentation"])


        # ids of all nucleis which are unconnected and can be used for further analysis
        labels_nuclei_unconnected = contact_filter(self.maps["nucleus_segmentation"], threshold=0.85, reindex=False)
        classes_nuclei_unconnected = np.unique(labels_nuclei_unconnected)

        self.log("Filtered out due to contact limit: {} ".format(len(all_classes)-len(classes_nuclei_unconnected)))

        labels_nuclei_filtered = size_filter(self.maps["nucleus_segmentation"],
                                             limits=[self.config["nucleus_segmentation"]["min_size"],self.config["nucleus_segmentation"]["max_size"]])
        classes_nuclei_filtered = np.unique(labels_nuclei_filtered)
        
        self.log("Filtered out due to size limit: {} ".format(len(all_classes)-len(classes_nuclei_filtered)))


        filtered_classes = set(classes_nuclei_unconnected).intersection(set(classes_nuclei_filtered))
        self.log("Filtered out: {} ".format(len(all_classes)-len(filtered_classes)))
        
        if self.debug:
            um_p_px = 665 / 1024
            um_2_px = um_p_px*um_p_px
            
            self.visualizue_class(classes_nuclei_unconnected, self.maps["nucleus_segmentation"], self.maps["normalized"][2])
            self.visualizue_class(classes_nuclei_filtered, self.maps["nucleus_segmentation"], self.maps["normalized"][2])

  
            plt.hist(length,bins=50)
            plt.xlabel("px area")
            plt.ylabel("number")
            
            plt.savefig('size_dist.eps')
            plt.show()
        # create background map based on WGA
        
        if self.start_from <= 4:
            self.log("Started with WGA mask map")
            
            wga_mask_comp  = self.maps["median"][1] - np.quantile(self.maps["median"][1],0.02)

            nn = np.quantile(self.maps["median"][1],0.98)
            wga_mask_comp = wga_mask_comp / nn
            wga_mask_comp = np.clip(wga_mask_comp, 0, 1)
            
            
            
            wga_mask = wga_mask_comp < self.config["wga_segmentation"]["threshold"]
            wga_mask = wga_mask.astype(float)
   
            wga_mask -= self.maps["nucleus_mask"]
            wga_mask = np.clip(wga_mask,0,1)


            wga_mask = dilation(wga_mask, selem=disk(self.config["wga_segmentation"]["erosion"]))
            self.maps["wga_mask"] = binary_erosion(wga_mask, selem=disk(self.config["wga_segmentation"]["dilation"]))
            
            self.save_map("wga_mask")
            self.log("WGA mask map created")
            
        # create WGA potential map
            
        if self.start_from <= 5:
            self.log("Started with WGA potential map")
            
            wga_mask_comp  = self.maps["median"][1] - np.quantile(self.maps["median"][1],0.02)

            nn = np.quantile(self.maps["median"][1],0.98)
            wga_mask_comp = wga_mask_comp / nn
            wga_mask_comp = np.clip(wga_mask_comp, 0, 1)
            
            # substract golgi and dapi channel from wga
            diff = np.clip(wga_mask_comp-self.maps["median"][0],0,1)
            diff = np.clip(diff-self.maps["nucleus_mask"],0,1)
            diff = 1-diff

            # enhance WGA map to generate speedmap
            # WGA 0.7-0.9
            min_clip = self.config["wga_segmentation"]["min_clip"]
            max_clip = self.config["wga_segmentation"]["max_clip"]
            diff = (np.clip(diff,min_clip,max_clip)-min_clip)/(max_clip-min_clip)

            diff = diff*0.9+0.1
            diff = diff.astype(dtype=float)

            self.maps["wga_potential"] = diff
            
            self.save_map("wga_potential")
            self.log("WGA mask potential created")
        
        # WGA cytosol segmentation by fast marching
        
        if self.start_from <= 6:
            self.log("Started with fast marching")
            
            fmm_marker = np.ones_like(self.maps["median"][0])
            px_center = np.round(center_nuclei).astype(int)
            
            for center in px_center:
                fmm_marker[center[0],center[1]] = 0
                
            fmm_marker  = np.ma.MaskedArray(fmm_marker, self.maps["wga_mask"])
            
            travel_time = skfmm.travel_time(fmm_marker, self.maps["wga_potential"])

            if not isinstance(travel_time, np.ma.core.MaskedArray):
                raise TypeError("travel_time for WGA based segmentation returned no MaskedArray. This is most likely due to missing WGA background determination.")
                
            self.maps["travel_time"] = travel_time.filled(fill_value=np.max(travel_time))
                
            self.save_map("travel_time")
            self.log("Fast marching finished")
                
        if self.start_from <= 7:
            self.log("Started with watershed")   
            
            marker = np.zeros_like(self.maps["median"][0])
            
            px_center = np.round(center_nuclei).astype(int)
            for i, center in enumerate(px_center):
                marker[center[0],center[1]] = i+1


            wga_labels = watershed(self.maps["travel_time"], marker, mask=self.maps["wga_mask"]==0)
            self.maps["watershed"] = np.where(self.maps["wga_mask"]> 0.5,0,wga_labels)
            
            if self.debug:
                image = label2rgb(self.maps["watershed"] ,self.maps["normalized"][0],bg_label=0,alpha=0.2)

                fig = plt.figure(frameon=False)
                fig.set_size_inches(10,10)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image)
                plt.scatter(center_nuclei[:,1],center_nuclei[:,0],color="red")
                plt.savefig(os.path.join(self.segmentation_directory, "watershed.png"))

                plt.show()
            
            self.save_map("watershed")
            self.log("watershed finished")
            
        channels = np.stack([self.maps["normalized"][2],
                           self.maps["normalized"][0],
                           self.maps["normalized"][1]]).astype("float16")
                             
        segmentation = np.stack([self.maps["nucleus_segmentation"],
                           self.maps["watershed"]]).astype("int32")
        
        
                             
                             
        
        map_path = os.path.join(self.segmentation_directory, self.DEFAULT_OUTPUT_FILE)
        hf = h5py.File(map_path, 'w')
        
        hf.create_dataset('labels', data=segmentation, chunks=(1, self.config["chunk_size"], self.config["chunk_size"]))
        hf.create_dataset('channels', data=channels, chunks=(1, self.config["chunk_size"], self.config["chunk_size"]))
        
        hf.close()
        
        
        # print filtered classes
        filtered_path = os.path.join(self.segmentation_directory, self.DEFAULT_FILTER_FILE)
        
        to_write = "\n".join([str(i) for i in list(filtered_classes)])
        with open(filtered_path, 'w') as myfile:
            myfile.write(to_write)
    
        
        self.log("============= Saved segmentation ============= ")
            
        

class ShardedWGASegmentation(Logable):
    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_OUTPUT_FILE = "segmentation.h5"
    DEFAULT_FILTER_FILE = "classes.csv"
    DEFAULT_INPUT_IMAGE_NAME = "input_image.h5"
    DEFAULT_SHARD_FOLDER = "shards"
    CLEAN_LOG = False
    
    def __init__(self, 
                 config, 
                 folder_path, 
                 debug=False, 
                 overwrite=False,
                 intermediate_output = True):
        super().__init__()
        
        
        self.segmentation_directory = folder_path
        self.shard_directory = os.path.join(folder_path, self.DEFAULT_SHARD_FOLDER)
        
        self.log_path = os.path.join(self.segmentation_directory, self.DEFAULT_LOG_NAME)
        self.debug = debug
        self.overwrite = overwrite
        self.config = config
        self.intermediate_output = intermediate_output
        
    def __call__(self, input_image, debug_images=False):
    
        # Create segmentation directory
        if self.overwrite:
            if os.path.isdir(self.shard_directory):
                shutil.rmtree(self.shard_directory)
            
        if not os.path.isdir(self.segmentation_directory):
                os.makedirs(self.segmentation_directory)
        

        if not os.path.isdir(self.shard_directory):
            os.makedirs(self.shard_directory)
            self.log("Created new shard directory " + self.shard_directory)
            
        # Set up log and clean old log
        if self.CLEAN_LOG:
            if os.path.isfile(self.log_path):
                os.remove(self.log_path)
                
        
        self.log("Started shardedd Segmentation")        
                
        self.maps = {"normalized": None,
                     "median": None,
                    "nucleus_segmentation": None,
                    "nucleus_mask": None,
                    "wga_mask":None,
                    "wga_potential": None,
                    "travel_time":None,
                    "watershed":None}
        
        self.start_from = 0
        
        self.save_input_image(input_image)
        
        
        # calculate sharding plan
        sharding_plan = []
        self.config["shard_size"] = int(self.config["shard_size"])
        self.image_size = input_image.shape[1:]

        if self.config["shard_size"] >= np.prod(self.image_size):
            self.log("target size is equal or larger to input image. Sharding will not be used.")

            sharding_plan.append((slice(0,self.image_size[0]),slice(0,self.image_size[1])))
        else:
            self.log("target size is smaller than input image. Sharding will be used.")
            sharding_plan = self.calculate_sharding_plan(self.image_size)
            
        shard_list = self.initialize_shard_list(sharding_plan)
        self.log(f"sharding plan with {len(sharding_plan)} elements generated, sharding with {self.config['threads']} threads begins")
        
        
        with Pool(processes=self.config["threads"]) as pool:
            x = pool.map(Shard.segment, shard_list)  
        self.log("Finished parallel segmentation")
        
        self.resolve_sharding(sharding_plan)
        
        self.log("============= Finished Segmentation ============= ")
        
        
    def resolve_sharding(self, sharding_plan):
        """
        The function iterates over a sharding plan and generates a new stitched hdf5 based segmentation.
        """
        
        self.log("resolve sharding plan")
        
        output = os.path.join(self.segmentation_directory,self.DEFAULT_OUTPUT_FILE)
        
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
        filtered_path = os.path.join(self.segmentation_directory, self.DEFAULT_FILTER_FILE)
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
        
        
    def initialize_shard_list(self, sharding_plan):
        _shard_list = []
        
        input_path = os.path.join(self.segmentation_directory, self.DEFAULT_INPUT_IMAGE_NAME)
        
        for i, window in enumerate(sharding_plan):
            local_shard_directory = os.path.join(self.shard_directory,str(i))
            
            _shard_list.append(Shard(i,
                 window,
                 self.config,
                 local_shard_directory,
                 input_path,
                 debug=self.debug))
            
        return _shard_list    
            
        
        
        
    def save_input_image(self, input_image):
        path = os.path.join(self.segmentation_directory, self.DEFAULT_INPUT_IMAGE_NAME)
        hf = h5py.File(path, 'w')
        hf.create_dataset('channels', data=input_image, chunks=(1, self.config["chunk_size"], self.config["chunk_size"]))
        hf.close()
        
        self.log(f"saved input_image: {path}")
        
    def calculate_sharding_plan(self, image_size):
        _sharding_plan = []
        side_size = np.floor(np.sqrt(self.config["shard_size"]))
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
    
    def get_output_file_path(self):
        return os.path.join(self.segmentation_directory, self.DEFAULT_OUTPUT_FILE)
        
class WGASegmentation:
    
    
    DEFAULT_LOG_NAME = "processing.log" 
    DEFAULT_OUTPUT_FILE = "segmentation.h5"
    DEFAULT_FILTER_FILE = "classes.csv"
    CLEAN_LOG = False
    
    def __init__(self, 
                 config, 
                 folder_path, 
                 debug=False, 
                 overwrite=False,
                 intermediate_output = True):
        
        """class can be initiated to create a WGA segmentation workfow

        :param config: Configuration for the segmentation passed over from the :class:`pipeline.Dataset`
        :type config: dict

        :param string: Directiory for the segmentation log and results. is created if not existing yet
        :type config: string
        
        :param debug: Flag used to output debug information and map images
        :type debug: bool, default False
        
        :param overwrite: Flag used to recalculate all maps and ignore precalculated maps
        :type overwrite: bool, default False
        """
        
        
        self.debug = debug
        self.overwrite = overwrite
        self.config = config
        self.intermediate_output = intermediate_output
        
        # Create segmentation directory
        self.segmentation_directory = folder_path
        if not os.path.isdir(self.segmentation_directory):
                os.makedirs(self.segmentation_directory)
                
        # Set up log and clean old log
        if self.CLEAN_LOG:
            log_path = os.path.join(self.segmentation_directory, self.DEFAULT_LOG_NAME)
            if os.path.isfile(log_path):
                os.remove(log_path)
        
        self.maps = {"normalized": None,
                     "median": None,
                    "nucleus_segmentation": None,
                    "nucleus_mask": None,
                    "wga_mask":None,
                    "wga_potential": None,
                    "travel_time":None,
                    "watershed":None}
        
        self.start_from = 0
        
        
            
    def get_output_file_path(self):
        return os.path.join(self.segmentation_directory, self.DEFAULT_OUTPUT_FILE)
        
    
    def get_timestamp(self):
        # datetime object containing current date and time
        now = datetime.now()

        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")  
        return "[" + dt_string + "] "
    
    def log(self, message):
        
        log_path = os.path.join(self.segmentation_directory, self.DEFAULT_LOG_NAME)
        
        with open(log_path, "a") as myfile:
            myfile.write(self.get_timestamp() + message +" \n")
            
        if self.debug:
            print(self.get_timestamp() + message)
    
    def load_maps_from_disk(self):       
        
        # iterating over all maps
        for map_index, map_name in enumerate(self.maps.keys()):

            
            try:
                map_path = os.path.join(self.segmentation_directory, "{}_{}_map.npy".format(map_index, map_name))

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
        self.start_from = np.argmin(is_not_none) if not all(is_not_none) else len(is_not_none)
        
    
    def save_image(self, array, save_name="", cmap="magma",**kwargs):
        if self.debug:
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
   
    
    def normalize(self, im):
        
        for i, channel in enumerate(im):
            
            lower_quantile = np.quantile(np.ravel(im[i]),self.config["lower_quantile_normalization"])
            upper_quantile = np.quantile(np.ravel(im[i]),self.config["upper_quantile_normalization"])
                                         
            IQR = upper_quantile - lower_quantile

                                         
            im[i] -= lower_quantile 
            im[i] = im[i] / IQR
            im[i] = np.clip(im[i], 0, 1)
        return im
    
    def save_map(self, map_name):
        if self.maps[map_name] is None:
            self.log("Error saving map {}, map is None".format(map_name))
        else:
            map_index = list(self.maps.keys()).index(map_name)
            
            if self.intermediate_output:
                map_path = os.path.join(self.segmentation_directory, "{}_{}_map.npy".format(map_index, map_name))
                np.save(map_path, self.maps[map_name])
                self.log("Saved map {} {} under path {}".format(map_index,map_name, map_path))
            
            # check if map contains more than one channel (3, 1024, 1024) vs (1024, 1024)
            
            if len(self.maps[map_name].shape) > 2:
                
                for i, channel in enumerate(self.maps[map_name]):
                    
                    
                    
                    channel_name = "{}_{}_{}_map.png".format(map_index, map_name,i)
                    channel_path = os.path.join(self.segmentation_directory, channel_name)
                    self.save_image(channel, save_name = channel_path)
            else:
                channel_name = "{}_{}_map.png".format(map_index, map_name)
                channel_path = os.path.join(self.segmentation_directory, channel_name)

                self.save_image(self.maps[map_name], save_name = channel_path)

    def __call__(self, image):
        """This function segments a wga image

        Parameters
        ----------
        image : numpy.array
            numpy array of shape (channels, size, width), channels needs to be = 3 for WGA. [golgi,wga,dapi]

        debug : bool, default = False
            Needed for parameter tuning with new data. Results in the display of all intermediate maps.
        """
        self.log("Segmentation started")
        
        if not self.overwrite:
            self.load_maps_from_disk()
        
        # Normalization
        if self.start_from <= 0:
            self.log("Started with normalized map")
            
            self.maps["normalized"] = self.normalize(image)
            self.save_map("normalized")
            
            self.log("Normalized map created")
                    
        
        # Median calculation
        if self.start_from <= 1:
            self.log("Started with median map")
            
            self.maps["median"] = np.copy(self.maps["normalized"])
                                   
            for i, channel in enumerate(self.maps["median"]):
                self.maps["median"][i] = median(channel, disk(self.config["median_filter_size"]))
            
            self.save_map("median")
            
            self.log("Median map created")
            
         # segment dapi channels based on local tresholding
        if self.debug:
            #plt.style.use("dark_background")
            plt.hist(self.maps["median"][2].flatten(),bins=100,log=False)
            plt.xlabel("intensity")
            plt.ylabel("frequency")
            plt.savefig("nuclei_frequency.png")
            plt.show()
            
       
        if self.start_from <= 2:
            self.log("Started with nucleus segmentation map")
            
            self.maps["nucleus_segmentation"] = segment_local_tresh(self.maps["median"][2], 
                                         dilation=self.config["nucleus_segmentation"]["dilation"], 
                                         thr=self.config["nucleus_segmentation"]["threshold"], 
                                         median_block=self.config["nucleus_segmentation"]["median_block"], 
                                         min_distance=self.config["nucleus_segmentation"]["min_distance"], 
                                         peak_footprint=self.config["nucleus_segmentation"]["peak_footprint"], 
                                         speckle_kernel=self.config["nucleus_segmentation"]["speckle_kernel"], 
                                         debug=self.debug)
            

            self.save_map("nucleus_segmentation")

            self.log("Nucleus segmentation map created")
        
        # Calc nucleus map
        
        if self.start_from <= 3:
            self.log("Started with nucleus mask map")
            self.maps["nucleus_mask"] = np.clip(self.maps["nucleus_segmentation"], 0,1)
            
            self.save_map("nucleus_mask")
            self.log("Nucleus mask map created with {} elements".format(np.max(self.maps["nucleus_segmentation"])))
            
        
        # filter nuclei based on size and contact
        center_nuclei, length, coords = mask_centroid(self.maps["nucleus_segmentation"], debug=self.debug)
        all_classes = np.unique(self.maps["nucleus_segmentation"])


        # ids of all nucleis which are unconnected and can be used for further analysis
        labels_nuclei_unconnected = contact_filter(self.maps["nucleus_segmentation"], threshold=0.7, reindex=False)
        classes_nuclei_unconnected = np.unique(labels_nuclei_unconnected)

        self.log("Filtered out due to contact limit: {} ".format(len(all_classes)-len(classes_nuclei_unconnected)))

        labels_nuclei_filtered = size_filter(self.maps["nucleus_segmentation"],limits=[150,700])
        classes_nuclei_filtered = np.unique(labels_nuclei_filtered)
        
        self.log("Filtered out due to size limit: {} ".format(len(all_classes)-len(classes_nuclei_filtered)))


        filtered_classes = set(classes_nuclei_unconnected).intersection(set(classes_nuclei_filtered))
        self.log("Filtered out: {} ".format(len(all_classes)-len(filtered_classes)))
            
            
        # create background map based on WGA
        
        if self.start_from <= 4:
            self.log("Started with WGA mask map")
            
            wga_mask_comp  = self.maps["median"][1] - np.quantile(self.maps["median"][1],0.02)

            nn = np.quantile(self.maps["median"][1],0.98)
            wga_mask_comp = wga_mask_comp / nn
            wga_mask_comp = np.clip(wga_mask_comp, 0, 1)
            
            
            
            wga_mask = wga_mask_comp < self.config["wga_segmentation"]["threshold"]
            wga_mask = wga_mask.astype(float)
   
            wga_mask -= self.maps["nucleus_mask"]
            wga_mask = np.clip(wga_mask,0,1)


            wga_mask = dilation(wga_mask, selem=disk(self.config["wga_segmentation"]["erosion"]))
            self.maps["wga_mask"] = binary_erosion(wga_mask, selem=disk(self.config["wga_segmentation"]["dilation"]))
            
            self.save_map("wga_mask")
            self.log("WGA mask map created")
            
        # create WGA potential map
            
        if self.start_from <= 5:
            self.log("Started with WGA potential map")
            
            wga_mask_comp  = self.maps["median"][1] - np.quantile(self.maps["median"][1],0.02)

            nn = np.quantile(self.maps["median"][1],0.98)
            wga_mask_comp = wga_mask_comp / nn
            wga_mask_comp = np.clip(wga_mask_comp, 0, 1)
            
            # substract golgi and dapi channel from wga
            diff = np.clip(wga_mask_comp-self.maps["median"][0],0,1)
            diff = np.clip(diff-self.maps["nucleus_mask"],0,1)
            diff = 1-diff

            # enhance WGA map to generate speedmap
            # WGA 0.7-0.9
            min_clip = self.config["wga_segmentation"]["min_clip"]
            max_clip = self.config["wga_segmentation"]["max_clip"]
            diff = (np.clip(diff,min_clip,max_clip)-min_clip)/(max_clip-min_clip)

            diff = diff*0.9+0.1
            diff = diff.astype(dtype=float)

            self.maps["wga_potential"] = diff
            
            self.save_map("wga_potential")
            self.log("WGA mask potential created")
        
        # WGA cytosol segmentation by fast marching
        
        if self.start_from <= 6:
            self.log("Started with fast marching")
            
            fmm_marker = np.ones_like(self.maps["median"][0])
            px_center = np.round(center_nuclei).astype(int)
            
            for center in px_center:
                fmm_marker[center[0],center[1]] = 0
                
            fmm_marker  = np.ma.MaskedArray(fmm_marker, self.maps["wga_mask"])
            
            travel_time = skfmm.travel_time(fmm_marker, self.maps["wga_potential"])

            if not isinstance(travel_time, np.ma.core.MaskedArray):
                raise TypeError("travel_time for WGA based segmentation returned no MaskedArray. This is most likely due to missing WGA background determination.")
                
            self.maps["travel_time"] = travel_time.filled(fill_value=np.max(travel_time))
                
            self.save_map("travel_time")
            self.log("Fast marching finished")
                
        if self.start_from <= 7:
            self.log("Started with watershed")   
            
            marker = np.zeros_like(self.maps["median"][0])
            
            px_center = np.round(center_nuclei).astype(int)
            for i, center in enumerate(px_center):
                marker[center[0],center[1]] = i+1


            wga_labels = watershed(self.maps["travel_time"], marker, mask=self.maps["wga_mask"]==0)
            self.maps["watershed"] = np.where(self.maps["wga_mask"]> 0.5,0,wga_labels)
            
            if self.debug:
                image = label2rgb(self.maps["watershed"] ,self.maps["normalized"][0],bg_label=0,alpha=0.2)

                fig = plt.figure(frameon=False)
                fig.set_size_inches(10,10)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(image)
                plt.scatter(center_nuclei[:,1],center_nuclei[:,0],color="red")
                plt.savefig(os.path.join(self.segmentation_directory, "watershed.png"))

                plt.show()
            
            self.save_map("watershed")
            self.log("watershed finished")
            
        channels = np.stack([self.maps["normalized"][2],
                           self.maps["normalized"][0],
                           self.maps["normalized"][1]]).astype("float16")
                             
        segmentation = np.stack([self.maps["nucleus_segmentation"],
                           self.maps["watershed"]]).astype("int32")
        
        
                             
                             
        
        map_path = os.path.join(self.segmentation_directory, self.DEFAULT_OUTPUT_FILE)
        hf = h5py.File(map_path, 'w')
        
        hf.create_dataset('labels', data=segmentation, chunks=(1, self.config["chunk_size"], self.config["chunk_size"]))
        hf.create_dataset('channels', data=channels, chunks=(1, self.config["chunk_size"], self.config["chunk_size"]))
        
        hf.close()
        
        
        # print filtered classes
        filtered_path = os.path.join(self.segmentation_directory, self.DEFAULT_FILTER_FILE)
        
        to_write = "\n".join([str(i) for i in list(filtered_classes)])
        with open(filtered_path, 'w') as myfile:
            myfile.write(to_write)
    
        
        self.log("============= Saved segmentation ============= ")