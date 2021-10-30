from vipercore.processing.preprocessing import percentile_normalization

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