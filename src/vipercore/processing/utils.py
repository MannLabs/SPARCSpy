import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
import os

def plot_image(array, size = (10,10), save_name="", cmap="magma", **kwargs):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(array, cmap=cmap, **kwargs)
    
    if save_name != "":
        plt.savefig(save_name + ".png")
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()
        
def visualize_class( class_ids, seg_map, background ,*args, **kwargs):
    index = np.argwhere(class_ids==0.)
    class_ids_no_zero = np.delete(class_ids, index)

    outmap_map = np.where(np.isin(seg_map,class_ids_no_zero), 2, seg_map)
    outmap_map = np.where(np.isin(seg_map,class_ids, invert=True), 1,outmap_map)

    image = label2rgb(outmap_map,background/np.max(background),alpha=0.4, bg_label=0)
    plot_image(image, **kwargs)
    
def download_testimage(folder):
    
    images = [("testimage_dapi.tiff","https://zenodo.org/record/5701474/files/testimage_dapi.tiff?download=1"),
             ("testimage_wga.tiff","https://zenodo.org/record/5701474/files/testimage_wga.tiff?download=1")]
    
    import urllib.request
    
    returns = []
    for name, url in images:
        path = os.path.join(folder, name)
    
        f = open(path,'wb')
        f.write(urllib.request.urlopen(url).read())
        f.close()
        print(f"Successfully downloaded {name} from {url}")
        returns.append(path)
    return returns