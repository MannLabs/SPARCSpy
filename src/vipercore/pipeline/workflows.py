class NWGASegmentation(Segmentation):
    
    def process(self, input_image):
       
        plot_image(input_image[0])
        
        segmentation = segment_local_tresh(input_image[0], 
                                         dilation=self.config["nucleus_segmentation"]["dilation"], 
                                         thr=self.config["nucleus_segmentation"]["threshold"], 
                                         median_block=self.config["nucleus_segmentation"]["median_block"], 
                                         min_distance=self.config["nucleus_segmentation"]["min_distance"], 
                                         peak_footprint=self.config["nucleus_segmentation"]["peak_footprint"], 
                                         speckle_kernel=self.config["nucleus_segmentation"]["speckle_kernel"], 
                                         debug=self.debug)
        
        
        
        all_classes = np.unique(segmentation)
        
        self.save_segmentation(input_image, segmentation, all_classes)

class NShardedWGASegmentation(ShardedSegmentation):
    method = NWGASegmentation    