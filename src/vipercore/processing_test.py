from vipercore.processing.preprocessing import percentile_normalization

import numpy as np
import pytest

from vipercore.processing.preprocessing import percentile_normalization
from vipercore.processing.segmentation import selected_coords, selected_coords_fast, remove_classes

import numpy as np
import pytest

def test_percentile_normalization_C_H_W():
    
    test_array = np.random.randint(2, size=(3,100,100))
    test_array[:,10:11,10:11]=-1
    test_array[:,12:13,12:13]=3
    
    normalized = percentile_normalization(test_array, 0.05,0.95)
    assert np.max(normalized) == pytest.approx(1)
    assert np.min(normalized) == pytest.approx(0)
    

def test_percentile_normalization_H_W():
    
    test_array = np.random.randint(2, size=(100,100))
    test_array[10:11,10:11]=-1
    test_array[12:13,12:13]=3
    
    normalized = percentile_normalization(test_array, 0.05,0.95)
    assert np.max(normalized) == pytest.approx(1)
    assert np.min(normalized) == pytest.approx(0)
    
# test processing.segmentation.selected_coords
def test_selected_coords():
    
    image_size = 20
    
    test_array = np.zeros((image_size,image_size))
    
    class_1 = np.array([[1,2],[2,3],[3,4],[4,5]])
    class_2 = np.array([[2,2],[3,3],[4,4],[5,6]])
    class_3 = np.array([[13,2],[14,3],[15,4],[16,6]])

    
    test_array[class_1[:,0],class_1[:,1]] = 1
    test_array[class_2[:,0],class_2[:,1]] = 2
    test_array[class_3[:,0],class_3[:,1]] = 3
    
    center, points_class, coords = selected_coords(test_array, np.array([1,2]))
    
    # check center obejct
    assert len(center) == 2
    
    np.testing.assert_array_equal(center[0], np.mean(class_1, axis=0))
    np.testing.assert_array_equal(center[1], np.mean(class_2, axis=0))
    
    # check points per class
    
    assert points_class[0] == len(class_1)
    assert points_class[0] == len(class_1)
    
    # check coordinates
    pred_class_1 = np.array(coords[0])
    np.testing.assert_array_equal(class_1[class_1[:,0].argsort()], pred_class_1[pred_class_1[:,0].argsort()])
    
    pred_class_2 = np.array(coords[1])
    np.testing.assert_array_equal(class_2[class_2[:,0].argsort()], pred_class_2[pred_class_2[:,0].argsort()])
    
    # test processing.segmentation.selected_coords_fast
    
def test_selected_coords_fast():
    
    image_size = 20
    
    test_array = np.zeros((image_size,image_size))
    
    class_1 = np.array([[1,2],[2,3],[3,4],[4,5]])
    class_2 = np.array([[2,2],[3,3],[4,4],[5,6]])
    class_3 = np.array([[13,2],[14,3],[15,4],[16,6]])

    
    test_array[class_1[:,0],class_1[:,1]] = 1
    test_array[class_2[:,0],class_2[:,1]] = 2
    test_array[class_3[:,0],class_3[:,1]] = 3
    
    center, points_class, coords = selected_coords_fast(test_array, np.array([1,2]))
    
    # check center obejct
    assert len(center) == 2
    
    np.testing.assert_array_equal(center[0], np.mean(class_1, axis=0))
    np.testing.assert_array_equal(center[1], np.mean(class_2, axis=0))
    
    # check points per class
    
    assert points_class[0] == len(class_1)
    assert points_class[0] == len(class_1)
    
    # check coordinates
    pred_class_1 = np.array(coords[0])
    np.testing.assert_array_equal(class_1[class_1[:,0].argsort()], pred_class_1[pred_class_1[:,0].argsort()])
    
    pred_class_2 = np.array(coords[1])
    np.testing.assert_array_equal(class_2[class_2[:,0].argsort()], pred_class_2[pred_class_2[:,0].argsort()])
    
def test_remove_classes():
    
    imgsize = 10
    maxclass = 20
    inarr = np.random.choice(maxclass, size=(imgsize*imgsize), replace=True).astype('int')
    inarr[0]=0
    
    to_delete_idx = np.random.choice(imgsize*imgsize, size=(10), replace=False)
    to_delete_classes = set(inarr[to_delete_idx])
    
    # reshape inarr to 2D 
    inarr = np.reshape(inarr, (imgsize,imgsize))
    
    # dont remove background label
    if 0 in to_delete_classes:
        to_delete_classes.remove(0)
        
    # create remain classes
    to_remain_classes = set(np.unique(inarr))-to_delete_classes
    
    outarr = remove_classes(inarr, to_delete_classes, reindex=False)
    assert to_remain_classes == set(np.unique(outarr))
    
    outarr_reindex = remove_classes(inarr, to_delete_classes, reindex=True)
    assert set(np.unique(outarr_reindex)) == set(np.arange(len(to_remain_classes)))
    
def test_test():
    assert 1 == 1