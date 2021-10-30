import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import multiprocessing
from numba import njit
import numba as nb

from skimage.color import label2rgb
from skimage.filters import gaussian
from skimage import filters
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import dilation as sk_dilation

from skimage.morphology import binary_erosion, disk

from vipercore.processing.utils import plot_image

import skimage as sk
import numpy as np
import skfmm

def segment_global_thresh(image, min_distance=8, min_size=20, dilation=0, threshold=3,peak_threshold=3, return_markers=False):
    """This function takes a preprocessed image with low background noise and extracts and segments the foreground.
    Extraction is performed based on global thresholding, therefore a preprocessed image with homogenous low noise background is needed.
    
    :param image: 2D numpy array of type float containing the image. 
    :type image: class:`numpy.array`
    
    :param min_distance: Minimum distance between the centers of two cells. This value is applied before mask dilation. defaults to 10
    :type min_distance: int, optional
    
    :param min_size: Minimum number of pixels occupied by a single cell. This value is applied before mask dilation. defaults to 20
    :type min_size: int, optional
    
    :param dilation: Dilation of the segmented masks in pixel. If no dilation is desired, set to zero. defaults to 0
    :type dilation: int, optional
    
    :param threshold: Threshold for areas to be considered as cells. Areas are considered as if they are larger than threshold * standard_dev. defaults to 3
    :type threshold: int, optional
    """
    
    # calculate global standard deviation
    std = np.std(image.flatten())
    
    image = image - np.median(image.flatten())
    
    # filter image for smoother shapes
    image = gaussian(image, sigma=1, preserve_range=True)
    peak_mask = np.where(image > peak_threshold * std, 1,0)
    

    distance = ndimage.distance_transform_edt(peak_mask)
    
    # find peaks based on distance transform
    peak_idx  = peak_local_max(distance, min_distance=min_distance, footprint=np.ones((3, 3)))
    local_maxi = np.zeros_like(image, dtype=bool)
    local_maxi[tuple(peak_idx.T)] = True
    markers = ndimage.label(local_maxi)[0]
    
    kernel = disk(dilation)
    
    mask = np.where(image > threshold * std, 1,0)

    dilated_mask = sk_dilation(mask,selem=kernel)
    dilated_distance = ndimage.distance_transform_edt(dilated_mask)
    

    labels = watershed(-dilated_distance,markers, mask=dilated_mask)
    
    
    if return_markers:
        return labels.astype(int), peak_idx
    else:
        return labels.astype(int)
    
def segment_local_tresh(image, 
                        dilation=4, 
                        thr=0.01, 
                        median_block=51, 
                        min_distance=10, 
                        peak_footprint=7, 
                        speckle_kernel=4, 
                        debug=False):
    """This function takes a unprocessed image with low background noise and extracts and segments approximately round foreground objects based on intensity.
    Extraction is performed based on local median thresholding.
    
    Parameters
    ----------
    image : numpy.array
        2D numpy array of type float containing the image. 
    
    thr : float, default = 0.01
        Treshold above median for segmentation.
        
    median_block : int, default = 51
        size of the receptive field for median calculation. Needs to be an odd number.
        
    min_distance : int, default = 10
        Minimum distance in px between the centers of two segemnts. This value is applied before mask dilation.
        
    peak_footprint : int, default = 7
        average width of peaks in px for the center detection.
        
    speckle_kernel : int, default = 4
        width of the kernal used for denoising. First a binary erosion with disk(speckle_kernel) is performed,
        then a dilation by disk(speckle_kernel -1)
    
    debug : bool, default = False
        Needed for parameter tuning with new data. Results in the display of all intermediate maps.
    """
    
    if speckle_kernel < 1:
        raise ValueError("speckle_kernel needs to be at least 1")

    local_thresh = filters.threshold_local(image, block_size=median_block,method="median", offset=-thr)
        
    image_mask = image > local_thresh
    
    if debug:
        plot_image(image_mask, cmap="Greys_r")
        
    # removing speckles by binary erosion and dilation 
    image_mask_clean = binary_erosion(image_mask, selem=disk(speckle_kernel))
    image_mask_clean = sk_dilation(image_mask_clean, selem=disk(speckle_kernel-1))
    
    #if debug:
    #    plot_image(image_mask_clean)
        
    
    
    # find peaks based on distance transform
    distance = ndimage.distance_transform_edt(image_mask_clean)
    
    peak_idx  = peak_local_max(distance, min_distance=min_distance, footprint=disk(peak_footprint))
    local_maxi = np.zeros_like(image_mask_clean, dtype=bool)
    local_maxi[tuple(peak_idx.T)] = True
    markers = ndimage.label(local_maxi)[0]
    
    if debug:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image_mask_clean,  cmap="Greys_r")
        plt.scatter(peak_idx[:,1],peak_idx[:,0],color="red")
        plt.show()
    
    
    # segmentation by fast marching and watershed
    

    dilated_mask = sk_dilation(image_mask_clean,selem=disk(dilation))


    fmm_marker = np.ones_like(dilated_mask)
    for center in peak_idx:
        fmm_marker[center[0],center[1]] = 0


    m = np.ma.masked_array(fmm_marker, np.logical_not(dilated_mask))
    distance_2 = skfmm.distance(m)
    
    if debug:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(distance_2,  cmap="viridis")
        plt.scatter(peak_idx[:,1],peak_idx[:,0],color="red")
        
        

    marker = np.zeros_like(image_mask_clean).astype(int)
    for i, center in enumerate(peak_idx):
        marker[center[0],center[1]] = i+1

    labels = watershed(distance_2, marker, mask=dilated_mask)


    if debug: 
        image = label2rgb(labels,image/np.max(image),alpha=0.2, bg_label=0)
        plot_image(image)
        
    return labels

def shift_labels(input_map, shift, return_shifted_labels=False):
    """Input is a segmentation in form of a 2d or 3d numpy array of type int representing a labeled map. All labels but the background are incremented and all classes in contact with the edges of the canvas are returned.
    """
        
    imap = input_map[:].copy()
    #if not issubclass(imap.dtype.type, np.integer):
    #    raise ValueError("operation is only permitted for integer arrays")

    shifted_map = np.where(imap == 0, 0, imap+shift)

    edge_label = []

    if len(imap.shape) == 2:

        edge_label += _return_edge_labels(imap)
    else:
        for dimension in imap:
            edge_label += _return_edge_labels(dimension)
            
    if return_shifted_labels:
        edge_label = [label + shift for label in edge_label]
        
    return shifted_map, list(set(edge_label))
        
@njit
def _return_edge_labels(input_map):
    top_row = input_map[0]
    bottom_row = input_map[-1]
    first_column = input_map[:,0]
    last_column = input_map[:,-1]
    
    full_union = set(top_row.flatten()).union(set(bottom_row.flatten())).union(set(first_column.flatten())).union(set(last_column.flatten()))
    full_union.discard(0)
    
    return list(full_union)



@njit(parallel=True)
def contact_filter_lambda(label, background=0):
    
    num_labels = np.max(label)
    
    background_matrix = np.ones((np.max(label)+1,2),dtype='int')

    for y in range(1,len(label)-1):
        for x in range(1,len(label[0])-1):
            
            current_label = label[y,x]
            
            background_matrix[current_label,0] += int(label[y-1,x] == 0)
            background_matrix[current_label,0] += int(label[y,x-1] == 0)
            background_matrix[current_label,0] += int(label[y+1,x] == 0)
            background_matrix[current_label,0] += int(label[y,x+1] == 0)
            
            background_matrix[current_label,1] += int(label[y-1,x] != current_label)
            background_matrix[current_label,1] += int(label[y,x-1] != current_label)
            background_matrix[current_label,1] += int(label[y+1,x] != current_label)
            background_matrix[current_label,1] += int(label[y,x+1] != current_label)
            
    
    pop = background_matrix[:,0]/background_matrix[:,1]
    
    return pop

@njit(parallel=True)
def remove_classes(label_in, to_remove, background=0, reindex=False):
    label = label_in.copy()
    # generate library which contains the new class for label x at library[x]
    remove_set = set(to_remove)

    library = np.zeros(np.max(label)+1, dtype="int")
    
    
    carry = 0
    for current_class in range(len(library)):
        if current_class in remove_set:
            library[current_class] = background
            
            if reindex:
                carry -= 1
            
        else:
            library[current_class] = current_class +carry
            
    # rewrite array based on library
    for y in range(len(label)):
        for x in range(len(label[0])):
            current_label = label[y,x]
            if current_label != background:
                label[y,x] = library[current_label]
                    
    
                    
    return label



def contact_filter(inarr, threshold=1, reindex=False, background=0):
    
    label = inarr.copy()

    #laulate contact matrix for labels
    background_contact = contact_filter_lambda(label, background=0)

    # extract all classes with less background contact than the threshold, but not the background  class
    to_remove = np.argwhere(background_contact<threshold).flatten()

    to_remove = np.delete(to_remove, np.where(to_remove == background))

    # remove these classes
    label = remove_classes(label,to_remove,reindex=reindex)
    
    return label

def size_filter(label, limits=[0,100000], background=0, reindex=False):
    center,points_class,coords = mask_centroid(label)
    
    below = np.argwhere(points_class < limits[0]).flatten()
    above = np.argwhere(points_class > limits[1]).flatten()
    
    to_remove = list(below) + list(above)
    
    if len(to_remove) > 0:
        label = remove_classes(label,to_remove,reindex=reindex)
    
    return label


@njit
def numba_mask_centroid(mask, debug=False, skip_background=True):
    
    
    num_classes = np.max(mask)+1
    class_range = [0,np.max(mask)]
        
    if class_range[1] > np.max(mask):
        raise ValueError("upper class range limit exceeds total classes")
        return

    points_class = np.zeros((num_classes,), dtype="uint32")
    
    center = np.zeros((num_classes,2,))
    
    if skip_background:
        points_class[0] = 1
        for y in range(len(mask)):
            for x in range(len(mask[0])):

                class_id = mask[y,x]
                if class_id > 0:

                    points_class[class_id] +=1
                    center[class_id] += np.array([x,y])
                    
    else:
        for y in range(len(mask)):
            for x in range(len(mask[0])):

                class_id = mask[y,x]
                points_class[class_id] +=1
                center[class_id] += np.array([x,y])
                    
        
    x = center[:,0]/points_class
    y = center[:,1]/points_class
    
    center = np.stack((y,x)).T
    return center, points_class

@njit
def selected_coords(segmentation, classes, debug=False):
    num_classes = len(classes)
    
    #setup emtpy lists
    coords = [[(np.array([0.,0.], dtype="int64"))]]
    
    
    
    for i in range(num_classes):
        coords.append([(np.array([0.,0.], dtype="int64"))])

    
    points_class = np.zeros((num_classes))
    center = np.zeros((num_classes,2,))
    
    y_size, x_size = segmentation.shape
    
    for y in range(y_size):
        for x in range(x_size):
            
            class_id = segmentation[y,x]
            
            if class_id in classes:
                
                return_id = np.argwhere(classes==class_id)[0][0]
                
                
                coords[return_id].append(np.array([y,x], dtype="int64")) # coords[translated_id].append(np.array([x,y]))
                points_class[return_id] +=1
                center[return_id] += np.array([x,y])
                
                
    x = center[:,0]/points_class
    y = center[:,1]/points_class
    center = np.stack((y,x)).T
    
    return center, points_class, coords

def mask_centroid(mask, class_range=None, debug=False):
    
    if class_range == None:
        num_classes = np.max(mask)
        class_range = [0,np.max(mask)]
    else:
        num_classes = class_range[1]-class_range[0]
        print(num_classes)
        
    if class_range[1] > np.max(mask):
        raise ValueError("upper class range limit exceeds total classes")
        return

    points_class = np.zeros((num_classes,))
    center = np.zeros((num_classes,2,))
    
    cl = np.empty(num_classes)
    
    
    coords = [[] for _ in range(num_classes)]

    
    for y in tqdm(range(len(mask)), disable = not debug):
   
        for x in range(len(mask[0])):
            class_id = mask[y,x]-1
            
            translated_id = class_id - class_range[0]
            if class_id >= class_range[0] and class_id < class_range[1]:

                coords[translated_id].append([x,y]) # coords[translated_id].append(np.array([x,y]))
                points_class[translated_id] +=1
                center[translated_id] += np.array([x,y])
            
        
    x = center[:,0]/points_class
    y = center[:,1]/points_class
    
    center = np.stack((y,x)).T
    return center,points_class,coords
  

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
    
    def get_center(self):
        return np.array([self.center[1],self.center[0]])
    
        
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
    
    def plot(self, axis, **kwargs):
        axis.plot(self.poly[:,0]+self.offset[0],self.poly[:,1]+self.offset[1], **kwargs)