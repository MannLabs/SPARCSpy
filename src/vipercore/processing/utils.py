import matplotlib.pyplot as plt

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