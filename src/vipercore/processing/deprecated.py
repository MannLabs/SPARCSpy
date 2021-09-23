from numba import jit, njit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_kernel( dilation ):

    ex_dim = np.ceil(dilation).astype(int)
    s = np.zeros((2*ex_dim+1,2*ex_dim+1))

    center = (np.array(s.shape)-1)/2
    for index in np.ndindex(s.shape):
        dist = np.linalg.norm(center-np.array(index))
        s[index] = 1 if dist <= dilation else 0

    return s

@njit(parallel=True)
def contact_filter_lambda_old(label, background=0):
    
    to_remove = []
    for y in range(1,len(label)-2):
        for x in range(1,len(label[0])-2):
            
            current_label = label[y,x]
            
            if current_label != background:
            
                contact = []
                contact.append(label[y-1,x])
                contact.append(label[y,x-1])
                contact.append(label[y+1,x])
                contact.append(label[y,x+1])

                contact = np.array(contact)

                in_contact = np.logical_and((contact != current_label),(contact != background))
                if np.any(in_contact):
                    to_remove.append(current_label)
                    
    
                    
    to_remove = list(set(to_remove))
    
    for y in range(len(label)):
        for x in range(len(label[0])):
            current_label = label[y,x]
            if current_label != background:
                if current_label in to_remove:
                    label[y,x] = background
    return label

    
def contact_filter_old(inarr, background=0, reindex=True):
    
    label = inarr.copy()
    print("contact filter started")
    labels = contact_filter_lambda_old(label, background=0)
    print("contact filter reindex")
    
    if reindex:
        labels = np.clip(labels,0,1)
        labels = sk_label(labels,connectivity=1)
    print("contact filter finished")
    return labels

@njit
def mask_to_centroid(mask):
    num_classes = np.max(mask)
    
    points_class = np.zeros((num_classes,))
    center = np.zeros((num_classes,2,))
    
    for y in range(len(mask)):
        
        for x in range(len(mask[0])):
            class_id = mask[y,x]
            if class_id > 0:
                points_class[class_id-1] +=1
                center[class_id-1] += np.array([x,y])
            
        
    x = center[:,0]/points_class
    y = center[:,1]/points_class
    
    center = np.stack((y,x)).T
    return center

import sys
import os
sys.path.append("/home/default/projects/opticalScreening/ImageProcessing/optical")

from readlif.reader import LifFile
from readlif.utilities import get_xml

import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
from multiprocessing import Pool, TimeoutError

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import multiprocessing
from numba import njit
from skimage.color import label2rgb
from skimage.filters import gaussian, median
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from skimage.filters import gaussian, median

from skimage.morphology import binary_erosion, disk, dilation

from scipy.ndimage import binary_fill_holes
import skimage as sk

from PIL import Image

import skfmm

from image_processing.Preprocessing import rolling_window_mean, origins_from_distance, plot_image
from image_processing.Segmentation import segment_local_tresh, mask_centroid, contact_filter, size_filter

def normalize(im):
    
    im = im-np.quantile(im,0.001)
    
    nn = np.quantile(im,0.999)
    im = im / nn
    im = np.clip(im, 0, 1)
    return im

def segment_wga(image, debug=False):
    """This function segments a wga image
    
    Parameters
    ----------
    image : numpy.array
        numpy array of shape (channels, size, width), channels needs to be = 3 for WGA. [golgi,wga,dapi]
    
    debug : bool, default = False
        Needed for parameter tuning with new data. Results in the display of all intermediate maps.
    """
    

    # load channels
    golgi = normalize(image[0])
    
    wga_raw = normalize(image[1])
    if debug:
        plot_image(wga_raw, save_name = "wga")
        
    wga = median(wga_raw,disk(4))
    
    if debug:
        plot_image(wga, save_name = "wga_median")
    
    dapi_raw = normalize(image[2])
    
    if debug:
        plot_image(dapi_raw, save_name = "dapi")
        
    # segment dapi channels based on local tresholding

    dapi = median(dapi_raw, disk(4))
    
    if debug:
        plt.style.use("dark_background")
        plt.hist(wga.flatten(),bins=100,log=False)
        plt.xlabel("intensity")
        plt.ylabel("frequency")
        plt.savefig("nuclei_frequency.png")
        plt.show()
    
    if debug:
        plot_image(dapi, save_name = "dapi_median")

    dapi_labels = segment_local_tresh(dapi, 
                                 dilation=0, 
                                 thr=0.08, 
                                 median_block=81, 
                                 min_distance=12, 
                                 peak_footprint=12, 
                                 speckle_kernel=5, 
                                 debug=debug)

    dapi_mask = np.clip(dapi_labels, 0,1)
    
    # filter nuclei based on size and contact
    center_nuclei, length, coords = mask_centroid(dapi_labels)
    all_classes = np.unique(dapi_labels)
    print(len(all_classes))
    

    # ids of all nucleis which are unconnected and can be used for further analysis
    labels_nuclei_unconnected = contact_filter(dapi_labels, threshold=0.7, reindex=False)
    classes_nuclei_unconnected = np.unique(labels_nuclei_unconnected)
    
    if debug:
        print("filtered due to contact limit",len(all_classes)-len(classes_nuclei_unconnected))

    labels_nuclei_filtered = size_filter(dapi_labels,limits=[150,700])
    classes_nuclei_filtered = np.unique(labels_nuclei_filtered)
    if debug:
        print("filtered due to size limits",len(all_classes)-len(classes_nuclei_filtered))


    filtered_classes = set(classes_nuclei_unconnected).intersection(set(classes_nuclei_filtered))
    if debug:
        print("filtered",len(all_classes)-len(filtered_classes))
    
    
    # create background map based on WGA
    if debug:
        #plot_image(wga)
        
        plt.hist(wga.flatten(),bins=100,log=True)
        plt.xlabel("intensity")
        plt.ylabel("frequency")
        plt.savefig("nuclei_frequency.png")
        plt.show()
        
    #wga_median = np.median(wga.flatten())
    wga_mask = wga < 0.1
    wga_mask = wga_mask.astype(float)
    if debug:
        plot_image(wga_mask, cmap="Greys",save_name="wga_mask")
    wga_mask -= dapi_mask
    wga_mask = np.clip(wga_mask,0,1)


    wga_mask = binary_erosion(wga_mask, selem=disk(4))
    wga_mask = dilation(wga_mask, selem=disk(4))
    
    if debug:
        plot_image(wga_mask, cmap="Greys",save_name="wga_mask_smooth")

    # substract golgi and dapi channel from wga
    diff = np.clip(median(wga,disk(3))-median(golgi,disk(3)),0,1)
    diff = np.clip(diff-dapi_mask,0,1)
    diff = 1-diff

    # enhance WGA map to generate speedmap
    # WGA 0.7-0.9
    min_clip = 0.5
    max_clip = 0.9
    diff = (np.clip(diff,min_clip,max_clip)-min_clip)/(max_clip-min_clip)

    diff = diff*0.9+0.1
    diff = diff.astype(dtype=float)

    speedmap = diff
    if debug:

        
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(speedmap, cmap="magma")

        
        
        plt.scatter(center_nuclei[:,1],center_nuclei[:,0],color="red")
        plt.savefig("wga_mask_fmm.png")
        
    
    # fast marching for segmentation
    
    fmm_marker = np.ones_like(dapi)
    px_center = np.round(center_nuclei).astype(int)

    for center in px_center:
        fmm_marker[center[0],center[1]] = 0


    fmm_marker  = np.ma.MaskedArray(fmm_marker, wga_mask)

    travel_time = skfmm.travel_time(fmm_marker,speedmap)
    
    if not isinstance(travel_time, np.ma.core.MaskedArray):
        raise TypeError("travel_time for WGA based segmentation returned no MaskedArray. This is most likely due to missing WGA background determination.")

    
    travel_time = travel_time.filled(fill_value=np.max(travel_time))
    
    if debug:
        
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(travel_time , cmap="viridis")
        
        plt.scatter(center_nuclei[:,1],center_nuclei[:,0],color="red")
        plt.savefig("wga_mask_dist.png")
        
        
    # watershed segmentation
    
    marker = np.zeros_like(dapi)
    for i, center in enumerate(px_center):
        marker[center[0],center[1]] = i+1


    wga_labels = watershed(travel_time,marker)
    wga_labels = np.where(wga_mask> 0.5,0,wga_labels)
 

    if debug:
        image = label2rgb(wga_labels ,golgi,bg_label=0,alpha=0.2)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)
        plt.scatter(center_nuclei[:,1],center_nuclei[:,0],color="red")
        plt.savefig("full_segment.png")
        plt.show()
        
    
    layers = np.stack([dapi_raw,golgi,wga_raw,dapi_labels,wga_labels]).astype("float32")
        
    return list(filtered_classes), center_nuclei, layers

def extract_classes(center_list, layers, arg, debug=True, output_f="output/"):

    if arg == 0:
        return
    
    index = arg-1
    px_center = center_list[index]
    
    
    dapi = layers[0]
    golgi = layers[1]
    wga = layers[2]
    labels_nuclei = layers[3]
    labels_cell = layers[4]
    
    image_width = len(labels_nuclei)
    image_height = len(labels_nuclei)
    
    width = 64
    
    if width < px_center[0] and px_center[0] < image_width-width and width < px_center[1] and px_center[1] < image_height-width:
        w = [px_center[0]-width,px_center[0]+width,px_center[1]-width,px_center[1]+width]

        # channel 0: nucleus mask
        nuclei_mask = np.where(labels_nuclei == index+1, 1,0)
        
        nuclei_mask_extended = gaussian(nuclei_mask,preserve_range=True,sigma=5)
        nuclei_mask = gaussian(nuclei_mask,preserve_range=True,sigma=1)
        
        nuclei_mask_extended = nuclei_mask_extended[w[0]:w[1],w[2]:w[3]]
        nuclei_mask = nuclei_mask[w[0]:w[1],w[2]:w[3]]
   
        
        if debug:
            plot_image(nuclei_mask, size=(5,5), save_name=os.path.join(output_f,"{}_0".format(index)), cmap="Greys") 
        
        # channel 1: cell mask

        cell_mask = np.where(labels_cell == index+1,1,0).astype(int)
        cell_mask = binary_fill_holes(cell_mask)
                
                
        cell_mask_extended = dilation(cell_mask,selem=disk(6))

        cell_mask = cell_mask[w[0]:w[1],w[2]:w[3]]

        cell_mask_extended = cell_mask_extended[w[0]:w[1],w[2]:w[3]]

        cell_mask =  gaussian(cell_mask,preserve_range=True,sigma=1)   
        cell_mask_extended = gaussian(cell_mask_extended,preserve_range=True,sigma=5)
                
   
        
        if debug:
            plot_image(cell_mask, size=(5,5), save_name=os.path.join(output_f,"{}_1".format(index)), cmap="Greys") 
        
        # channel 2: nucleus
        channel_nucleus = dapi[w[0]:w[1],w[2]:w[3]]
        channel_nucleus = normalize(channel_nucleus)
        channel_nucleus = channel_nucleus *nuclei_mask_extended
                
        channel_nucleus = MinMax(channel_nucleus)
        
        if debug:
            plot_image(channel_nucleus, size=(5,5), save_name=os.path.join(output_f,"{}_2".format(index)))
        

        # channel 3: golgi
        channel_golgi = golgi[w[0]:w[1],w[2]:w[3]]
        channel_golgi = normalize(channel_golgi)
        channel_golgi = channel_golgi*cell_mask_extended
                
        channel_golgi = MinMax(channel_golgi)
        
        if debug:
            plot_image(channel_golgi, size=(5,5), save_name=os.path.join(output_f,"{}_3".format(index)))

        # channel 4: cellmask
        channel_wga = wga[w[0]:w[1],w[2]:w[3]]
        channel_wga = normalize(channel_wga)
        channel_wga = channel_wga*cell_mask_extended
                
        if debug:
            plot_image(channel_wga, size=(5,5), save_name=os.path.join(output_f,"{}_4".format(index)))
        
        design = np.stack([nuclei_mask,cell_mask,channel_nucleus,channel_golgi,channel_wga], axis=-1).astype("float32")
        np.save(os.path.join(output_f,"{}.npy".format(index)),design)

def extract_class_old(arg):
    global dapi
    global golgi
    global cellmask
    global labels
    global output_f
    global labels_nuclei
    
    global DEBUG_sub
    DEBUG = DEBUG_sub
    
    image_width = len(labels)
    image_height = len(labels)

    width = 96
    
    i, coords = arg
    
    idl = i + 1
    
    if idl in ok:
            # check if the current cell is far enough away from the image border
            if width < coords[0] and coords[0] < image_width-width and width < coords[1] and coords[1] < image_height-width:
                
                w = [coords[0]-width,coords[0]+width,coords[1]-width,coords[1]+width]

                # channel 0: nucleus mask
                nuclei_mask = np.where(labels_nuclei == idl, 1,0)             
                nuclei_mask = gaussian(nuclei_mask,preserve_range=True,sigma=1)
                nuclei_mask = nuclei_mask[w[0]:w[1],w[2]:w[3]]
                
                nuclei_mask = MinMax(nuclei_mask)

                if DEBUG:
                    plot_image(nuclei_mask, size=(5,5), save_name=output_f+"{}_0".format(i), cmap="Greys")


                # channel 1: cell mask

                cell_mask = np.where(labels == idl,1,0).astype(int)
                cell_mask = binary_fill_holes(cell_mask)
                
                
                cell_mask_extended = sk_dilation(cell_mask,selem=disk(10))

                cell_mask = cell_mask[w[0]:w[1],w[2]:w[3]]
                cell_mask_extended = cell_mask_extended[w[0]:w[1],w[2]:w[3]]

                cell_mask =  gaussian(cell_mask,preserve_range=True,sigma=1)   
                cell_mask_extended = gaussian(cell_mask_extended,preserve_range=True,sigma=5)
                
                cell_mask = MinMax(cell_mask)

                if DEBUG:
                    plot_image(cell_mask, size=(5,5), save_name=output_f+"{}_1".format(i), cmap="Greys") 

                # channel 2: nucleus
                channel_nucleus = dapi[w[0]:w[1],w[2]:w[3]]
                channel_nucleus = channel_nucleus *cell_mask
                
                channel_nucleus = MinMax(channel_nucleus)

                if DEBUG:
                    plot_image(channel_nucleus, size=(5,5), save_name=output_f+"{}_2".format(i))

                # channel 3: golgi
                channel_golgi = golgi[w[0]:w[1],w[2]:w[3]]
                channel_golgi = channel_golgi*cell_mask_extended
                
                channel_golgi = MinMax(channel_golgi)

                if DEBUG:
                    plot_image(channel_golgi, size=(5,5), save_name=output_f+"{}_3".format(i))

                # channel 4: cellmask
                channel_cellmask = cellmask[w[0]:w[1],w[2]:w[3]]
                
                channel_cellmask = channel_cellmask*cell_mask_extended
                
                channel_cellmask = MinMax(channel_cellmask)
                
                
                
                if DEBUG:
                    plot_image(channel_cellmask, size=(5,5), save_name=output_f+"{}_4".format(i))


                design = np.stack([nuclei_mask,cell_mask,channel_nucleus,channel_golgi,channel_cellmask], axis=-1).astype("float32")
                np.save(output_f +"{}.npy".format(idl),design)
 
        
def MinMax(inarr):
    if np.max(inarr) - np.min(inarr) > 0:
        return (inarr - np.min(inarr)) / (np.max(inarr) - np.min(inarr))
    else:
        return inarr
    

def process_golgi_map(input_map, output_folder, DEBUG=False):
    global DEBUG_sub
    DEBUG_sub = DEBUG
    
    global output_f
    output_f = output_folder
    global dapi
    dapi = median(input_map[:,:,1], disk(2))
    dapi = dapi-np.median(dapi.flatten())
    dapi = np.clip(dapi,0,np.max(dapi))
    
    if DEBUG:
        plot_image(dapi)

    dapi_med = median(dapi, disk(5))
    
    global golgi
    golgi = input_map[:,:,0]
    golgi = golgi-np.median(golgi.flatten())
    golgi = np.clip(golgi,0,np.max(golgi))
    
    global cellmask
    cellmask = input_map[:,:,2]
    cellmask = cellmask-np.median(cellmask.flatten())
    cellmask = np.clip(cellmask,0,np.max(cellmask))

    # === nuclear segmentation ===
    
    global labels_nuclei
    labels_nuclei = segment_thr(dapi_med, dilation=0, min_distance=25, threshold=1.0,peak_threshold=1.0)

    #labels_nuclei_size_filter = size_filter(labels_nuclei,50,200, reindex=False)
    
    if DEBUG:
        image = label2rgb(labels_nuclei, bg_label=0)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)
        plt.show()


    # ids of all nucleis which are unconnected and can be used for further analysis
    labels_nuclei_unconnected = contact_filter(labels_nuclei, threshold=0.95, reindex=False)
    list_nuclei_unconnected = np.unique(labels_nuclei_unconnected)

    # centers of all nucleis
    center_nuclei = mask_to_centroid(labels_nuclei)

    if DEBUG:
        plot_image(dapi)
        plt.scatter(center_nuclei[:,1],center_nuclei[:,0],color="red")

    # === cell mask segmentation
    
    if DEBUG:
        plot_image(cellmask)

    std = np.std(cellmask.flatten())
    med = np.median(cellmask.flatten())

    thr = med + 0.5*std

    mask = np.where(cellmask > thr,1,0) + labels_nuclei
    mask = np.clip(mask,0,1)
    mask = sk_dilation(mask,selem=disk(2))
    mask = binary_erosion(mask)

    if DEBUG:
        plot_image(mask)

    # === fast marching separation

    fmm_marker = np.ones_like(dapi)
    px_center = np.round(center_nuclei).astype(int)

    for center in px_center:
        fmm_marker[center[0],center[1]] = 0

    m = np.ma.masked_array(fmm_marker, np.logical_not(mask))
    distance = skfmm.distance(m)
    
    if DEBUG:
        plot_image(distance)


    marker = np.zeros_like(dapi)
    for i, center in enumerate(px_center):
        marker[center[0],center[1]] = i+1

    global labels
    labels = watershed(distance,marker, mask=mask)
    
    labels_cells_unconnected = contact_filter(labels, threshold=0.50, reindex=False)
    list_cells_unconnected = np.unique(labels_cells_unconnected)
    
    
    global ok
    ok = list(set(list_cells_unconnected) & set(list_nuclei_unconnected))

    print("cells total: ", np.max(labels), "cells filtered: ", len(ok))

    if DEBUG:
        image = label2rgb(labels_cells_unconnected, bg_label=0)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)
        plt.show()

        image = label2rgb(labels, bg_label=0)

        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image)
        plt.show()
    
    # output segments
    
    
    
    with Pool(processes=40) as pool:
        x = pool.map(extract_class,enumerate(px_center))
    
    return len(ok)
    """ 
    for i, coords in tqdm(enumerate(px_center[0:],start=0),len(px_center[0:])):
        idl = i+1
        if idl in ok:


            # check if the current cell is far enough away from the image border
            if width < coords[0] and coords[0] < image_width-width and width < coords[1] and coords[1] < image_height-width:
                w = [coords[0]-width,coords[0]+width,coords[1]-width,coords[1]+width]

                # channel 0: nucleus mask
                nuclei_mask = np.where(labels_nuclei == idl, 1,0)             
                nuclei_mask = gaussian(nuclei_mask,preserve_range=True,sigma=1)
                nuclei_mask = nuclei_mask[w[0]:w[1],w[2]:w[3]]

                if DEBUG:
                    plot_image(nuclei_mask, size=(5,5), save_name=output_folder+"{}_0".format(i), cmap="Greys")


                # channel 1: cell mask

                cell_mask = np.where(labels == idl,1,0).astype(int)
                cell_mask = binary_fill_holes(cell_mask)
                cell_mask_extended = sk_dilation(cell_mask,selem=disk(10))

                cell_mask = cell_mask[w[0]:w[1],w[2]:w[3]]
                cell_mask_extended = cell_mask_extended[w[0]:w[1],w[2]:w[3]]

                cell_mask =  gaussian(cell_mask,preserve_range=True,sigma=1)   
                cell_mask_extended = gaussian(cell_mask_extended,preserve_range=True,sigma=5)

                if DEBUG:
                    plot_image(cell_mask, size=(5,5), save_name=output_folder+"{}_1".format(i), cmap="Greys") 

                # channel 2: nucleus
                channel_nucleus = dapi[w[0]:w[1],w[2]:w[3]]
                channel_nucleus = channel_nucleus *cell_mask/np.max(channel_nucleus)

                if DEBUG:
                    plot_image(channel_nucleus, size=(5,5), save_name=output_folder+"{}_2".format(i))

                # channel 3: golgi
                channel_golgi = golgi[w[0]:w[1],w[2]:w[3]]
                channel_golgi = channel_golgi*cell_mask_extended/np.max(channel_golgi)

                if DEBUG:
                    plot_image(channel_golgi, size=(5,5), save_name=output_folder+"{}_3".format(i))

                # channel 4: cellmask
                channel_cellmask = cellmask[w[0]:w[1],w[2]:w[3]]
                channel_cellmask = channel_cellmask*cell_mask_extended/np.max(cell_mask)

                if DEBUG:
                    plot_image(channel_cellmask, size=(5,5), save_name=output_folder+"{}_4".format(i))


                design = np.stack([nuclei_mask,cell_mask,channel_nucleus,channel_golgi,channel_cellmask], axis=-1)
                np.save(output_folder +"{}.npy".format(idl),design)
    """   
    