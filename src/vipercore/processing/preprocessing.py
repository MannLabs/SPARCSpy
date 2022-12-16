from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def percentile_normalization(im, lower_percentile, upper_percentile):
    
    """Normalize an input image channel wise based on defined percentiles. 
    The percentiles will be calculated and the image will be normalized to ``[0, 1]`` based on the lower and upper percentile.
    
    Args
        channels (np.array): Numpy array of shape ``(height, width)`` or ``(channels, height, width)``.
        
        lower_percentile (float, between [0, 1]): lower percentile used for normalization. All lower values will be clipped to 0.
        
        upper_percentile (float, between [0, 1]): upper percentile used for normalization. All higher values will be clipped to 1.
    """
    
    # chek if data is passed as (height, width) or (channels, height, width)
    
    if len(im.shape) == 2:
        im = _percentile_norm(im, lower_percentile, upper_percentile)
        
    elif len(im.shape) == 3:
        for i, channel in enumerate(im):
            im[i] = _percentile_norm(im[i], lower_percentile, upper_percentile)
            
    else:
        raise ValueError("Input dimensions should be (height, width) or (channels, height, width).")

        
    return im
    
def _percentile_norm(im, lower_percentile, upper_percentile):
    
    lower_value = np.quantile(np.ravel(im),lower_percentile)
    upper_value = np.quantile(np.ravel(im),upper_percentile)
                                         
    IPR = upper_value - lower_value


    out_im = im - lower_value 
    #add check to make sure IPR is not 0
    if IPR != 0:
        out_im = out_im / IPR
    out_im = np.clip(out_im, 0, 1)
            
    return out_im
    
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

