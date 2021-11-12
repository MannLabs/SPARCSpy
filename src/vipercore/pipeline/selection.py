from vipercore.pipeline.base import ProcessingStep
import os
import csv
from pathlib import Path
import numpy as np
import h5py
from vipercore.processing.segmentation import selected_coords_fast
from functools import partial
from tqdm import tqdm
import multiprocessing

from scipy import ndimage
from scipy.signal import convolve2d

import skimage as sk
from skimage.morphology import dilation as sk_dilation
from skimage.morphology import binary_erosion, disk

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from lmd.lmd import LMD_object, LMD_shape

class LMDSelection(ProcessingStep):
    """Select single cells from a segmented hdf5 file and generate cutting data for the LMD Microscope
    
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def process(self, hdf_location, cell_sets, calibration_marker):
        """Process function for starting the processing
        
        Args:
            hdf_location (str): path of the segmentation hdf5 file. If this class is used as part of a project processing workflow, this argument will be provided.
            
            cell_sets (list(dict)): List of dictionaries containing the sets of cells which should be sorted into a single well.
            
            calibration_marker (np.array): Array of size '(3,2)' containing the calibration marker coordinates in the '(row, column)' format.
            
        Important:
        
            If this class is used as part of a project processing workflow, the first argument will be provided by the ``Project`` class based on the previous segmentation. Therefore, only the second and third argument need to be provided. Under the hood, the Project class retrives the most recent segmentation and passes it forward together with the supplied parameters. The Implementation is similar to:
            
            .. code-block:: python
                
                def select(self, *args, **kwargs):
                
                    input_segmentation = self.segmentation_f.get_output()
                    
                    self.selection_f.process(input_segmentation, *args, **kwargs)      
                    
                    
        Example:
            
            The following parameters are required in the config file:
            
            .. code-block:: python
            
                # Calibration marker should be defined as (row, column).
                marker_0 = np.array([-10,-10])
                marker_1 = np.array([-10,1100])
                marker_2 = np.array([1100,505])

                # A numpy Array of shape (3, 2) should be passed.
                calibration_marker = np.array([marker_0, marker_1, marker_2])
                
                
                # Sets of cells can be defined by providing a name and a list of classes in a dictionary.
                cells_to_select = [{"name": "dataset1", "classes": [1,2,3]}]
                
                # Alternatively, a path to a csv file can be provided. 
                # If a relative path is provided, it is accessed relativ to the projects base directory.
                cells_to_select += [{"name": "dataset2", "classes": "segmentation/class_subset.csv"}]
                
                # If desired, wells can be passed with the individual sets.
                cells_to_select += [{"name": "dataset3", "classes": [4,5,6], "well":"A1"}]
                
                project.select(cells_to_select, calibration_marker)
                    
        Note:
            
            The following parameters are required in the config file:
            
            .. code-block:: yaml
            
                LMDSelection:
                    threads: 10

                    # defines the channel used for generating cutting masks
                    # segmentation.hdf5 => labels => segmentation_channel
                    # When using WGA segmentation:
                    #    0 corresponds to nuclear masks
                    #    1 corresponds to cytosolic masks.
                    segmentation_channel: 0

                    # dilation of the cutting mask in pixel 
                    shape_dilation: 10

                    # number of datapoints which are averaged for smoothing
                    # the number of datapoints over an distance of n pixel is 2*n
                    smoothing_filter_size: 10

                    # fold reduction of datapoints for compression
                    poly_compression_factor: 30

        """
        
        self.log("Selection process started")
        self.hdf_location = hdf_location
        self.calibration_marker = calibration_marker
        
        sets = []
        
        # iterate over all defined sets, perform sanity checks and load external data
        for i, cell_set in enumerate(cell_sets):
            self.log(f"sanity check for cell set {i}")
            self.check_cell_set_sanity(cell_set)
            cell_set["classes_loaded"] = self.load_classes(cell_set)
            sets.append(cell_set)
            self.log(f"cell set {i} passed sanity check")
            
        for i, cell_set in enumerate(cell_sets):
            self.generate_cutting_data(cell_set)
        
    def generate_cutting_data(self, cell_set):
        self.log(f"generate cutting data for set {cell_set['name']}")
        self.log(f"load hdf5 segmentation")
        
        hf = h5py.File(self.hdf_location, 'r')
        hdf_channels = hf.get('channels')
        hdf_labels = hf.get('labels')
        
        self.log("Finished loading channel data " + str(hdf_channels.shape))
        self.log("Finished loading label data " + str(hdf_labels.shape))
        
        self.log("Convert mask format into coordinate format")
        
        center, length, coords = selected_coords_fast(hdf_labels[self.config['segmentation_channel']], cell_set["classes_loaded"])
        
        self.log("Conversion finished, sanity check")
        
        if len(center) == len(cell_set["classes_loaded"]):
            self.log("Check passed")
        else:
            self.log("Check failed, returned lengths do not match cell set.\n Some classes were not found in the segmentation and were therefore removed.\n Please make sure all classes specified are present in your segmentation.")
            elements_removed =  len(cell_set["classes_loaded"]) - len(center)
            self.log(f"{elements_removed} classes were not found and therefore removed.")
        
        # Sanity check for the returned coordinates
        if len(center) == len(length) == len(length):
            self.log("Check passed")
        else:
            self.log("Check failed, returned lengths do not match. Please check if all classes specified are present in your segmentation")
        
        zero_elements = 0
        for el in coords:
            if len(el) == 0:
                zero_elements += 1
                
        if zero_elements == 0:
            self.log("Check passed")
        else:
            self.log("Check failed, returned coordinates contain empty elements. Please check if all classes specified are present in your segmentation")           
        
        self.log("Initializing shapes for polzgon creation")
        shapes = []
        for i in range(len(center)):
            
            if i % 1000 == 0:
                self.log(f"Initializing shape {i}")
                
            s = Shape(center[i],length[i],coords[i])
            shapes.append(s)

        self.log("Calculating polygons")
        with multiprocessing.Pool(processes=self.config['threads']) as pool:                                          
            shapes = list(tqdm(pool.imap(partial(Shape.create_poly, 
                                                 smoothing_filter_size = self.config['smoothing_filter_size'], 
                                                 poly_compression_factor = self.config['poly_compression_factor'],
                                                 dilation = self.config['shape_dilation'],
                                                ),
                                                 shapes), total=len(center)))
        
        self.log("Polygon calculation finished")
        
        # Plot coordinates if in debug mode
        if self.debug:
            
            center = np.array(center)
            figure(figsize=(8, 8), dpi=120)
            ax = plt.gca()

            ax.imshow(hdf_labels[0], cmap="magma")
            ax.scatter(center[:,1],center[:,0], s=1)

            for shape in shapes:
                shape.plot(ax, color="red",linewidth=1)

            ax.scatter(self.calibration_marker[:,1], self.calibration_marker[:,0], color="lime")

            plt.show()
        
        self.log("Generate XML from polygons")
        
        orientation_transform = np.array([[-1,0],[0,1]])
    
        #generate array of marker cross positions
        ds = LMD_object()
        ds.calibration_points = self.calibration_marker @ orientation_transform

        for shape in shapes:
            s = shape.get_poly() @ orientation_transform
            
            if "well" in cell_set:
                ds.new_shape(s, well="well")
            else:
                ds.new_shape(s)
        
        if self.debug:
            ds.plot(calibration =True)
        
        savename = cell_set['name'].replace(" ","_") + ".xml"
        savepath = os.path.join(self.directory, savename)
        ds.save(savepath)
        
        self.log(f"Saved output at {savepath}")

        
    def check_cell_set_sanity(self, cell_set):
        """Check if cell_set dictionary contains the right fields

        """
        if "name" in cell_set:
            if not isinstance(cell_set["name"], str):
                self.log("No name of type str specified for cell set")
                raise TypeError()
        else:
            self.log("No name of type str specified for cell set")
            raise KeyError()
        
        if "classes" in cell_set:
            if not isinstance(cell_set["classes"], (list, str)):
                self.log("No list of classes specified for cell set")
                raise TypeError()
        else:
            self.log("No classes specified for cell set")
            raise KeyError()\
            
        if "well" in cell_set:
            if not isinstance(cell_set["well"], str):
                self.log("No well of type str specified for cell set")
                raise TypeError()
                
    def load_classes(self, cell_set):
        if isinstance(cell_set["classes"], list):
            return cell_set["classes"]
        
        if isinstance(cell_set["classes"], str):
            
            # If the path is relative, it is interpreted relative to the project directory
            if os.path.isabs(cell_set["classes"]):
                path = cell_set["classes"]
            else:
                path = os.path.join(Path(self.directory).parents[0],cell_set["classes"])        
            
    
            if os.path.isfile(path):
                
                cr = csv.reader(open(path,'r'))
                filtered_classes = np.array([int(el[0]) for el in list(cr)], dtype = "int64")
                self.log("Loaded {} classes from csv".format(len(filtered_classes)))
                return filtered_classes
            else:
     
                self.log("path containing classes was not readable: " + path)
                raise ValueError()
            
class Shape:
    """
    Helper class which is created for every segment. Can be used to convert a list of pixels into a polygon.
    Reasonable results should only be expected for fully connected sets of coordinates. 
    The resulting polygon has a baseline resolution of twice the pixel density.
    
    """
    
    def __init__(self, center,length,coords):
        self.center = center
        self.length = length
        self.coords = np.array(coords)
        
        #print(self.center,self.length)
        
        #print(self.coords.shape)
        # may be needed for further debugging
        """
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(bounds)
        ax.plot(edges[:,1]*2,edges[:,0]*2)"""
        
    def create_poly(self, 
                    smoothing_filter_size = 12,
                    poly_compression_factor = 8,
                   dilation = 0):
        
        """
        Converts a list of pixels into a polygon.
        Parameters
        ----------
        smoothing_filter_size : int, default = 12
            The smoothing filter is the circular convolution with a vector of length smoothing_filter_size and all elements 1 / smoothing_filter_size.
            
        poly_compression_factor : int, default = 8    
            When compression is seeked, only every n-th element is kept for n = poly_compression_factor.
        """
        
        
        safety_offset = 3
        dilation_offset = dilation 
        
        
        # top left offsett used for creating the offset map
        self.offset = np.min(self.coords,axis=0)-safety_offset-dilation_offset
        
        
        self.offset_coords = self.coords-self.offset
        
        self.offset_map = np.zeros(np.max(self.offset_coords,axis=0)+2*safety_offset+dilation_offset)
        
        
        y = tuple(self.offset_coords.T[0])
        x = tuple(self.offset_coords.T[1])

        self.offset_map[(y,x)] = 1
        self.offset_map = self.offset_map.astype(int)
        
        plt.imshow(self.offset_map)
        plt.show()
        
        self.offset_map = sk_dilation(self.offset_map , selem=disk(dilation))
        
        plt.imshow(self.offset_map)
        plt.show()
        
        # find polygon bounds from mask
        bounds = sk.segmentation.find_boundaries(self.offset_map, connectivity=1, mode="subpixel", background=0)
        
        
        
        edges = np.array(np.where(bounds == 1))/2
        edges = edges.T
        edges = self.sort_edges(edges)
        
        # smoothing resulting shape
        smk = np.ones((smoothing_filter_size,1))/smoothing_filter_size
        edges = convolve2d(edges,smk,mode="full",boundary="wrap")
        
        
        # compression of the resulting polygon      
        newlen = np.round(len(edges)/poly_compression_factor).astype(int)
        
        mine = 0
        maxe= len(edges)-1
        
        indices=np.linspace(mine,maxe,newlen).astype(int)

        self.poly = edges[indices]
        
        # Useful for debuging
        """
        print(self.poly.shape)
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(bounds)
        ax.plot(edges[:,1]*2,edges[:,0]*2)
        ax.plot(self.poly[:,1]*2,self.poly[:,0]*2)
        """
        
        return self
    
    def get_poly(self):
        return self.poly+self.offset
    
        
    def sort_edges(self, edges):
        """
        greedy sorts the vertices of a graph.
        
        """

        it = len(edges)
        new = []
        new.append(edges[0])

        edges = np.delete(edges,0,0)

        for i in range(1,it):

            old = np.array(new[i-1])


            dist = np.linalg.norm(edges-old,axis=1)

            min_index = np.argmin(dist)
            new.append(edges[min_index])
            edges = np.delete(edges,min_index,0)
        
        return(np.array(new))
    
    def plot(self, axis,  flip=True, **kwargs):
        """
        Args
            flip (bool, True): Shapes are still in the (row, col) format and need to bee flipped if plotted with a (x, y) coordinate system.
        """
        if flip:
            axis.plot(self.poly[:,1]+self.offset[1],self.poly[:,0]+self.offset[0], **kwargs)
        else:
            axis.plot(self.poly[:,0]+self.offset[0],self.poly[:,1]+self.offset[1], **kwargs)