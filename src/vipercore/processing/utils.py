import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb

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