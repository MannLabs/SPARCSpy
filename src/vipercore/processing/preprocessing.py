from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

@jit(nopython=True, parallel = True) # Set "nopython" mode for best performance, equivalent to @njit
def rolling_window_mean(array,size,scaling = False):
    overlap=0
    lengthy, lengthx = array.shape
    delta = size-overlap
    
    ysteps = lengthy // delta
    xsteps = lengthx // delta
    
    x = 0
    y = 0
    
    for i in range(ysteps):
        for j in range(xsteps):
            y = i*delta
            x = j*delta
            
            yd = min(y+size,lengthy)
            xd = min(x+size,lengthx)
            
            
            chunk = array[y:yd,x:xd]
            std = np.std(chunk.flatten())
            max = np.max(chunk.flatten())
            if scaling:
                    chunk = chunk / std
                    
            
            mean = np.median(chunk.flatten())
            
            if max > 0:
                chunk = (chunk - mean)
                chunk = np.where(chunk < 0,0,chunk)
       
            
            array[y:yd,x:xd] = chunk
    if scaling:
        array = array/np.max(array.flatten())
    return array

def origins_from_distance(array):
    array = gaussian(array, sigma=1)
    std = np.std(array.flatten())
    
    thr_mask = np.where(array > 3 * std, 1,0)
    distance = ndimage.distance_transform_edt(thr_mask)
    
    peak_list  = peak_local_max(distance, min_distance=5, threshold_abs=std,footprint=np.ones((3, 3)))

    peak_map = np.zeros_like(base, dtype=bool)
    peak_map[tuple(peak_list.T)] = True
    
    return peak_list, peak_map

