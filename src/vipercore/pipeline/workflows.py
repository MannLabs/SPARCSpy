from vipercore.pipeline.segmentation import Segmentation, ShardedSegmentation
from vipercore.processing.preprocessing import percentile_normalization
from vipercore.processing.utils import plot_image, visualize_class
from vipercore.processing.segmentation import segment_local_tresh, segment_global_tresh, mask_centroid, contact_filter, size_filter, shift_labels, _class_size, global_otsu

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

class WGASegmentation(Segmentation):
    
    def process(self, input_image):
        
        self.maps = {"normalized": None,
                     "median": None,
                    "nucleus_segmentation": None,
                    "nucleus_mask": None,
                    "wga_mask":None,
                    "wga_potential": None,
                    "travel_time":None,
                    "watershed":None}
        
        start_from = self.load_maps_from_disk()
        

        if self.identifier is not None:
            self.log(f"Segmentation started shard {self.identifier}, starting from checkpoint {start_from}")
            
        else:
            self.log(f"Segmentation started, starting from checkpoint {start_from}")
        
        # Normalization
        if start_from <= 0:
            self.log("Started with normalized map")
            
            self.maps["normalized"] = percentile_normalization(input_image, 
                                                               self.config["lower_quantile_normalization"], 
                                                               self.config["upper_quantile_normalization"])
            self.save_map("normalized")
            
            self.log("Normalized map created")
            
            
            
        # Median calculation
        if start_from <= 1:
            self.log("Started with median map")
            
            self.maps["median"] = np.copy(self.maps["normalized"])
                                   
            for i, channel in enumerate(self.maps["median"]):
                self.maps["median"][i] = median(channel, disk(self.config["median_filter_size"]))
            
            self.save_map("median")
            
            self.log("Median map created")
            
        # segment dapi channels based on local tresholding
        if self.debug:
            plt.hist(self.maps["median"][0].flatten(),bins=100,log=False)
            plt.xlabel("intensity")
            plt.ylabel("frequency")
            plt.yscale('log')

            plt.title("DAPI intensity distribution")
            plt.savefig("dapi_intensity_dist.png")
            plt.show()

            plt.hist(self.maps["median"][1].flatten(),bins=100,log=False)
            plt.xlabel("intensity")
            plt.ylabel("frequency")
            plt.yscale('log')

            plt.title("WGA intensity distribution")
            plt.savefig("wga_intensity_dist.png")
            plt.show()
            
       
        if start_from <= 2:
            self.log("Started with nucleus segmentation map")
            
            nucleus_map_tr = percentile_normalization(self.maps["median"][0],
                                                      self.config["nucleus_segmentation"]["lower_quantile_normalization"],
                                                      self.config["nucleus_segmentation"]["upper_quantile_normalization"])

            # Use manual threshold if defined in ["wga_segmentation"]["threshold"]
            # If not, use global otsu
            if 'threshold' in self.config["nucleus_segmentation"] and 'median_block' in self.config["nucleus_segmentation"]:
                self.maps["nucleus_segmentation"] = segment_local_tresh(nucleus_map_tr, 
                                         dilation=self.config["nucleus_segmentation"]["dilation"], 
                                         thr=self.config["nucleus_segmentation"]["threshold"], 
                                         median_block=self.config["nucleus_segmentation"]["median_block"], 
                                         min_distance=self.config["nucleus_segmentation"]["min_distance"], 
                                         peak_footprint=self.config["nucleus_segmentation"]["peak_footprint"], 
                                         speckle_kernel=self.config["nucleus_segmentation"]["speckle_kernel"], 
                                         median_step=self.config["nucleus_segmentation"]["median_step"],
                                         debug=self.debug)
            else:
                self.log('No treshold or median_block for nucleus segmentation defined, global otsu will be used.')
                self.maps["nucleus_segmentation"] = segment_global_tresh(nucleus_map_tr, 
                                         dilation=self.config["nucleus_segmentation"]["dilation"], 
                                         min_distance=self.config["nucleus_segmentation"]["min_distance"], 
                                         peak_footprint=self.config["nucleus_segmentation"]["peak_footprint"], 
                                         speckle_kernel=self.config["nucleus_segmentation"]["speckle_kernel"], 
                                         debug=self.debug)            
            
            del nucleus_map_tr
            self.save_map("nucleus_segmentation")

            self.log("Nucleus segmentation map created")
        
        # Calc nucleus map
        
        if start_from <= 3:
            self.log("Started with nucleus mask map")
            self.maps["nucleus_mask"] = np.clip(self.maps["nucleus_segmentation"], 0,1)
            
            self.save_map("nucleus_mask")
            self.log("Nucleus mask map created with {} elements".format(np.max(self.maps["nucleus_segmentation"])))
            
        
        # filter nuclei based on size and contact
        center_nuclei, length = _class_size(self.maps["nucleus_segmentation"], debug=self.debug)
        
        all_classes = np.unique(self.maps["nucleus_segmentation"])

        
        


        # ids of all nucleis which are unconnected and can be used for further analysis
        labels_nuclei_unconnected = contact_filter(self.maps["nucleus_segmentation"], 
                                    threshold=self.config["nucleus_segmentation"]["contact_filter"], 
                                    reindex=False)
        classes_nuclei_unconnected = np.unique(labels_nuclei_unconnected)

        self.log("Filtered out due to contact limit: {} ".format(len(all_classes)-len(classes_nuclei_unconnected)))

        labels_nuclei_filtered = size_filter(self.maps["nucleus_segmentation"],
                                             limits=[self.config["nucleus_segmentation"]["min_size"],
                                                     self.config["nucleus_segmentation"]["max_size"]])
        
        
        classes_nuclei_filtered = np.unique(labels_nuclei_filtered)
        
        self.log("Filtered out due to size limit: {} ".format(len(all_classes)-len(classes_nuclei_filtered)))


        filtered_classes = set(classes_nuclei_unconnected).intersection(set(classes_nuclei_filtered))
        self.log("Filtered out: {} ".format(len(all_classes)-len(filtered_classes)))
        
        if self.debug:
            um_p_px = 665 / 1024
            um_2_px = um_p_px*um_p_px
            
            visualize_class(classes_nuclei_unconnected, self.maps["nucleus_segmentation"], self.maps["normalized"][0])
            visualize_class(classes_nuclei_filtered, self.maps["nucleus_segmentation"], self.maps["normalized"][0])

  
            plt.hist(length,bins=50)
            plt.xlabel("px area")
            plt.ylabel("number")
            plt.title('Nucleus size distribution')
            plt.savefig('nucleus_size_dist.png')
            plt.show()
        
        # create background map based on WGA
        
        if start_from <= 4:
            self.log("Started with WGA mask map")

            # Perform percentile normalization
            wga_mask_comp = percentile_normalization(self.maps["median"][1],
                                                      self.config["wga_segmentation"]["lower_quantile_normalization"],
                                                      self.config["wga_segmentation"]["upper_quantile_normalization"])
            
            # Use manual threshold if defined in ["wga_segmentation"]["threshold"]
            # If not, use global otsu
            if 'threshold' in self.config["wga_segmentation"]:
                wga_mask = wga_mask_comp < self.config["wga_segmentation"]["threshold"]
            else:
                self.log('No treshold for cytosol segmentation defined, global otsu will be used.')
                wga_mask = wga_mask_comp < global_otsu(wga_mask_comp)


            wga_mask = wga_mask.astype(float)
            wga_mask -= self.maps["nucleus_mask"]
            wga_mask = np.clip(wga_mask,0,1)

            # Apply dilation and erosion
            wga_mask = dilation(wga_mask, footprint=disk(self.config["wga_segmentation"]["erosion"]))
            self.maps["wga_mask"] = binary_erosion(wga_mask, footprint=disk(self.config["wga_segmentation"]["dilation"]))
            
            self.save_map("wga_mask")
            self.log("WGA mask map created")
            
        # create WGA potential map
            
        if start_from <= 5:
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
            
            #self.save_map("wga_potential")
            self.log("WGA mask potential created")
        
        # WGA cytosol segmentation by fast marching
        
        if start_from <= 6:
            self.log("Started with fast marching")
            
            fmm_marker = np.ones_like(self.maps["median"][0])
            px_center = np.round(center_nuclei).astype(int)
            
            for center in px_center[1:]:
                fmm_marker[center[0],center[1]] = 0
                
            fmm_marker  = np.ma.MaskedArray(fmm_marker, self.maps["wga_mask"])
            
            travel_time = skfmm.travel_time(fmm_marker, self.maps["wga_potential"])

            if not isinstance(travel_time, np.ma.core.MaskedArray):
                raise TypeError("travel_time for WGA based segmentation returned no MaskedArray. This is most likely due to missing WGA background determination.")
                
            self.maps["travel_time"] = travel_time.filled(fill_value=np.max(travel_time))
                
            self.save_map("travel_time")
            self.log("Fast marching finished")
                
        if start_from <= 7:
            self.log("Started with watershed")   
            
            marker = np.zeros_like(self.maps["median"][1])
            
            px_center = np.round(center_nuclei).astype(int)
            for i, center in enumerate(px_center[1:]):
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
                plt.savefig(os.path.join(self.directory, "watershed.png"))

                plt.show()

            # filter cells based on cytosol size
            center_cell, length, coords = mask_centroid(self.maps["watershed"], debug=self.debug)
        
            
            
            all_classes_wga = np.unique(self.maps["watershed"])

            labels_wga_filtered = size_filter(self.maps["watershed"],
                                                 limits=[self.config["wga_segmentation"]["min_size"],
                                                         self.config["wga_segmentation"]["max_size"]])
            
            classes_wga_filtered = np.unique(labels_wga_filtered)
            
            self.log("Cells filtered out due to cytosol size limit: {} ".format(len(all_classes_wga)-len(classes_wga_filtered)))

            filtered_classes_wga = set(classes_wga_filtered)
            filtered_classes = set(filtered_classes).intersection(filtered_classes_wga)
            self.log("Filtered out: {} ".format(len(all_classes)-len(filtered_classes)))
            self.log("Remaining: {} ".format(len(filtered_classes)))
            
            
            if self.debug:
                um_p_px = 665 / 1024
                um_2_px = um_p_px*um_p_px
                
                visualize_class(classes_wga_filtered, self.maps["watershed"], self.maps["normalized"][1])
                visualize_class(classes_wga_filtered, self.maps["watershed"], self.maps["normalized"][0])

                
                plt.hist(length, bins=50)
                plt.xlabel("px area")
                plt.ylabel("number")
                plt.title('Cytosol size distribution')
                plt.savefig('cytosol_size_dist.png')
                plt.show()
            
            
            self.save_map("watershed")
            self.log("watershed finished")
            
        # The required maps are the nucelus channel and a membrane marker channel like WGA
        required_maps = [self.maps["normalized"][0],
                         self.maps["normalized"][1]]
        
        # Feature maps are all further channel which contain phenotypes needed for the classification
        feature_maps = [element for element in self.maps["normalized"][2:]]
            
        channels = np.stack(required_maps+feature_maps).astype("float16")
                             
        segmentation = np.stack([self.maps["nucleus_segmentation"],
                                 self.maps["watershed"]]).astype("int32")
        
        self.save_segmentation(channels, segmentation, filtered_classes)

class ShardedWGASegmentation(ShardedSegmentation):
    method = WGASegmentation    
