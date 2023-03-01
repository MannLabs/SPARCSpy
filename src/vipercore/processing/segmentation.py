import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import multiprocessing
from numba import njit
from numba import prange
import numba as nb

from skimage.color import label2rgb
from skimage.filters import gaussian
from skimage import filters
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import dilation as sk_dilation
from skimage.morphology import binary_erosion, disk

from hilbertcurve.hilbertcurve import HilbertCurve

from vipercore.processing.utils import plot_image

import skimage as sk
from skimage.transform import resize
import numpy as np
import skfmm

def segment_global_tresh(image, 
                        dilation=4, 
                        min_distance=10, 
                        peak_footprint=7, 
                        speckle_kernel=4, 
                        debug=False):
    
    image_mask = image > global_otsu(image)
    
    if debug:
        plot_image(image_mask, cmap="Greys_r")
        
    # removing speckles by binary erosion and dilation 
    image_mask_clean = binary_erosion(image_mask, footprint=disk(speckle_kernel))
    image_mask_clean = sk_dilation(image_mask_clean, footprint=disk(speckle_kernel-1))
    
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
    

    dilated_mask = sk_dilation(image_mask_clean, footprint=disk(dilation))


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
    
def segment_local_tresh(image, 
                        dilation=4, 
                        thr=0.01, 
                        median_block=51, 
                        min_distance=10, 
                        peak_footprint=7, 
                        speckle_kernel=4, 
                        median_step = 1,
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

    downsampled_image = image[::median_step, ::median_step]

    local_thresh = filters.threshold_local(downsampled_image, 
                                            block_size=median_block,
                                            method="median", 
                                            offset=-thr)

    local_thresh = resize(local_thresh, image.shape)
        
    image_mask = image > local_thresh
    
    if debug:
        plot_image(image_mask, cmap="Greys_r")
        
    # removing speckles by binary erosion and dilation 
    image_mask_clean = binary_erosion(image_mask, footprint=disk(speckle_kernel))
    image_mask_clean = sk_dilation(image_mask_clean, footprint=disk(speckle_kernel-1))
    
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
    

    dilated_mask = sk_dilation(image_mask_clean,footprint=disk(dilation))


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

def global_otsu(image):
    counts, bin_edges = np.histogram(np.ravel(image), bins=512)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges)/2

    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]
    return threshold

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

def remove_classes(label_in, to_remove, background=0, reindex=False):
    return _remove_classes(label_in, to_remove, background=background, reindex=reindex)


@njit(parallel = True)
def _remove_classes(label_in, to_remove, background=0, reindex=False):
    
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
            library[current_class] = current_class + carry
            
    # rewrite array based on library
    for y in prange(len(label)):
        for x in prange(len(label[0])):
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
    
    # numba typed list fails to determine type for empty list
    # return without removing classes if no classes should be removed
    if len(to_remove) > 0:
        # remove these classes
        label = remove_classes(label,nb.typed.List(to_remove),reindex=reindex)
    else:
         pass
    
    return label

@njit
def _numba_subtract(array1, min_number):
    for i in range(array1.shape[0]):     # parallel --> for i in nb.prange(c.shape[0]):
        for j in range(array1.shape[1]):
            if array1[i, j] != 0:
                array1[i, j] = array1[i, j] - min_number

    return array1

@njit
def numba_mask_centroid(mask, debug=False, skip_background=True):
    """
    returns
    center
        numpy array containing the y, x coordinates of each element
    points_class
        the number of pixels associated with each class
    class_id
        the id number of each class
    """
    
    # need to perform this adjustment here so that we can also work with segmentations that do not start with a seg index of 1!
    # this is relevant when working with segmentations that have been reindexed over different tiles

    cell_ids = list(np.unique(mask).flatten())
    
    if 0 in cell_ids: cell_ids.remove(0)
    
    cell_ids = np.array(cell_ids)
    min_cell_id = np.min(cell_ids) #need to convert to array since numba min functions requires array as input not list
                                       #-1 important since otherwise the cell with the lowest id becomes 0 and is ignored (since 0 = background)
    
    if min_cell_id != 1:
        mask = _numba_subtract(mask, min_cell_id - 1 ) 

    num_classes = np.max(mask)
    class_range = [0, np.max(mask)]
        
    if class_range[1] > np.max(mask):
        raise ValueError("upper class range limit exceeds total classes")
        return
    
    #add check to make sure that not run when there is only background
    if class_range[1] == 0:
        print("no cells in image.")
        #raise ValueError("no cells in image.")
        return None, None, None

    points_class = np.zeros((num_classes,), dtype = nb.uint32)
    center = np.zeros((num_classes, 2, ))
    ids = np.zeros((num_classes,))

    if skip_background:
        for y in range(len(mask)):
            for x in range(len(mask[0])):

                class_id = mask[y,x]
                if class_id > 0:
                    class_id -= 1
                    points_class[class_id] +=1
                    center[class_id] += np.array([x,y])
                    ids[class_id] = class_id + 1
                    
    else:
        for y in range(len(mask)):
            for x in range(len(mask[0])):
                class_id = mask[y,x]
                points_class[class_id] += 1
                center[class_id] += np.array([x,y])
                ids[class_id] = class_id
                    
    x = center[:,0]/points_class
    y = center[:,1]/points_class
    
    center = np.stack((y,x)).T
    
    if min_cell_id != 1:
        ids += (min_cell_id - 1 ) 

    return center, points_class, ids.astype("int32")

@njit
def _selected_coords_fast(mask, classes, debug=False, background=0):
    
    num_classes = np.max(mask)+1
    
    coords = []
    
    for i in prange(num_classes):
        coords.append([np.array([0.,0.], dtype="uint32")])
    
    rows, cols = mask.shape
    
    for row in range(rows):
        for col in range(cols):
            return_id = mask[row, col]
            if return_id != background:
                coords[return_id].append(np.array([row, col], dtype="uint32")) # coords[translated_id].append(np.array([x,y]))
    
    for i, el in enumerate(coords):
        #print(i, el)
        if i not in classes:
            #print(i)
            coords[i] = [np.array([0.,0.], dtype="uint32")]
                
        #return
    return coords
             

def selected_coords_fast(inarr, classes, debug=False):
    # return with empty lists if no classes are provided
    if len(classes) == 0:
        return [],[],[]
    
    # calculate all coords in list
    # due to typing issues in numba, every list and sublist contains np.array([0.,0.], dtype="int32") as first element
    coords = _selected_coords_fast(inarr.astype("uint32"), nb.typed.List(classes))
    
    #print("start removal of zero vectors")
    # removal of np.array([0.,0.], dtype="int32")
    coords = [np.array(el[1:]) for el in coords[1:]]
    
    #print("start removal of out of class cells")
    # remove empty elements, not in class list
    coords_filtered = [np.array(el) for i, el in enumerate(coords) if i+1 in classes]
    
    #print("start center calculation")
    center = [np.mean(el, axis=0) for el in coords_filtered]
    
    #print("start length calculation")
    length = [len(el) for el in coords_filtered]
    
    return center, length, coords_filtered

def size_filter(label, limits=[0,100000], background=0, reindex=False):
    center,points_class = _class_size(label)
    
    below = np.argwhere(points_class < limits[0]).flatten()
    above = np.argwhere(points_class > limits[1]).flatten()
    
    to_remove = list(below) + list(above)
    
    if len(to_remove) > 0:
        label = remove_classes(label,nb.typed.List(to_remove),reindex=reindex)
    
    return label

@njit
def _class_size(mask, debug=False, background=0):
    
    num_classes = np.max(mask)+1
    
    mean_sum = np.zeros((num_classes, 2))
    length = np.zeros((num_classes, 1))
    
    
    
    rows, cols = mask.shape
    
    for row in range(rows):
        if row % 10000 == 0:
            print(row)
        for col in range(cols):
            return_id = mask[row, col]
            if return_id != background:
                mean_sum[return_id] += np.array([row, col], dtype="uint32") 
                length[return_id][0] += 1
                
                
    mean_arr = np.divide(mean_sum, length)

    return mean_arr, length.flatten()

@njit
def _selected_coords(segmentation, classes, debug=False):
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

# calculate center, coordinates and number of pixel for a selected set of classes
def selected_coords(segmentation, classes, debug=False):
    center, points_class, coords = _selected_coords(segmentation, classes, debug=False)
    
    # ccoords array contains [0, 0] as first element.
    # hack needed to tell numba the datatype
    # folowing lines are needed for removal
    out_l = []
    for elem in coords:
        if len(elem) > 1:
            out_l.append(np.array(elem[1:]))
    
    coords = out_l
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
                
    # empty classes 
    points_class[points_class==0]=1       
        
    x = center[:,0]/points_class
    y = center[:,1]/points_class
    
    center = np.stack((y,x)).T
    return center,points_class,coords
  
# return the first element not present in a list
def _get_closest(used, choices, world_size):
    for element in choices:
        if element not in used:
            # knn matrix contains -1 if the number of elements is smaller than k
            if element == -1:
                return None
            else:
                return element
        
    return None
    # all choices have been taken, return closest free index due to local optimality
    
def _tps_greedy_solve(data, k=100):
    samples = len(data)
    
    print(f"{samples} nodes left")
    #recursive abort
    if samples == 1:
        return data
    
    import umap
    knn_index, knn_dist, _ = umap.umap_.nearest_neighbors(data, n_neighbors=k, 
                                       metric='euclidean', metric_kwds={},
                                       angular=True, random_state=np.random.RandomState(42))

    knn_index = knn_index[:,1:]
    knn_dist = knn_dist[:,1:]

    # follow greedy knn as long as a nearest neighbour is found in the current tree            
    nodes = []
    current_node = 0
    while current_node is not None:
        nodes.append(current_node)
        #print(current_node, knn_index[current_node], next_node)
        next_node = _get_closest(nodes, knn_index[current_node], samples)
        
        current_node = next_node

    # as soon as no nearest neigbour can be found, create a new list of all elements still remeining
    # nodes: [0, 2, 5], nodes_left: [1, 3, 4, 6, 7, 8, 9]
    # add the last node assigned as starting point to the new list
    # nodes: [0, 2], nodes_left: [5, 1, 3, 4, 6, 7, 8, 9]

  
    nodes_left = list(set(range(samples))-set(nodes))
    

    # add last node from nodes to nodes_left

    nodes_left = [nodes.pop(-1)] + nodes_left
    

    node_data_left = data[nodes_left]
    
    # join lists
    
    return np.concatenate([data[nodes], _tps_greedy_solve(node_data_left, k=k)])

# calculate the index array for a sorted 2d list based on an unsorted list
@njit()
def _get_nodes(data, sorted_data):
    indexed_data = [(i,el) for i, el in enumerate(data)]

    epsilon = 1e-10
    nodes = []

    print("start sorting")
    for element in sorted_data:

        for j, tup in enumerate(indexed_data):
            i, el = tup

            if np.array_equal(el, element):
                nodes.append(i)
                indexed_data.pop(j)
    return nodes
    
def tsp_greedy_solve(node_list, k=100, return_sorted=False):
    """Find an approximation of the closest path through a list of coordinates
    
    Args:
        node_list (np.array): Array of shape `(N, 2)` containing a list of coordinates
        
        k (int, default: 100): Number of Nearest Neighbours calculated for each Node.
        
        return_sorted: If set to False a list of indices is returned. If set to True the sorted coordinates are returned.
    
    """
    
    sorted_nodes = _tps_greedy_solve(node_list)
    
    if return_sorted:
        return sorted_nodes
        
    else:
        nodes_order = _get_nodes(node_list, sorted_nodes)
        return nodes_order
    
@njit()
def assign_vertices(hilbert_points, data_rounded):

    data_rounded = data_rounded.astype(np.int64)
    hilbert_points = hilbert_points.astype(np.int64)


    output_order = np.zeros(len(data_rounded)).astype(np.int64)
    current_index = 0

    for hilbert_point in hilbert_points:

        for i, data_point in enumerate(data_rounded):
            if np.array_equal(hilbert_point, data_point):
                output_order[current_index] = i
                current_index += 1

    return output_order

def tsp_hilbert_solve(data , p=3):

    p=p; n=2
    max_n = 2**(p*n)
    hilbert_curve = HilbertCurve(p, n)
    distances = list(range(max_n))
    hilbert_points = hilbert_curve.points_from_distances(distances)
    hilbert_points = np.array(hilbert_points)




    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)


    hilbert_min = np.min(hilbert_points, axis=0)
    hilbert_max = np.max(hilbert_points, axis=0)



    data_scaled = data - data_min
    data_scaled = data_scaled / (data_max-data_min) * (hilbert_max - hilbert_min)




    data_rounded = np.round(data_scaled).astype(int)

    order = assign_vertices(hilbert_points, data_rounded)
    
    return order
    
def calc_len(data):
    """calculate the length of a path based on a list of coordinates
    
    Args:
        data (np.array): Array of shape `(N, 2)` containing a list of coordinates
       
    """

    index = np.arange(len(data)).astype(int)
    
    not_shifted = data[index[:-1]]
    shifted = data[index[1:]]
    
    diff = not_shifted-shifted
    sq = np.square(diff)
    dist = np.sum(np.sqrt(np.sum(sq, axis = 1)))
    
    return dist