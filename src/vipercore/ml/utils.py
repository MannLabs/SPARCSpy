from scipy.sparse import csr_matrix
import numpy as np
import torch

def combine_datasets_balanced(list_of_datasets, class_labels, train_per_class, val_per_class, test_per_class,):
    
    elements = [len(el) for el in list_of_datasets]
    rows = np.arange(len(list_of_datasets))
    
    # create dataset fraction array of len(list_of_datasets)
    mat = csr_matrix((elements, (rows, class_labels))).toarray()
    cells_per_class = np.sum(mat,axis=0)
    normalized = mat / cells_per_class
    dataset_fraction = np.sum(normalized,axis=1)
    
    train_dataset = []
    test_dataset = []
    val_dataset = []
    
    for dataset, label, fraction in zip(list_of_datasets, class_labels, dataset_fraction):
        print(dataset, label, fraction)
        train_size = np.round(train_per_class*fraction).astype(int)
        test_size = np.round(test_per_class*fraction).astype(int)
        val_size = np.round(val_per_class*fraction).astype(int)
        
        residual_size = len(dataset) - train_size - test_size - val_size
        
        if(residual_size < 0):
            raise ValueError(f"Dataset with length {len(dataset)} is to small to be split into test set of size {test_size} and train set of size {train_size}. Use a smaller test and trainset.")
        
        train, test, val, _ = torch.utils.data.random_split(dataset, [train_size, test_size, val_size, residual_size])
        train_dataset.append(train)
        test_dataset.append(test)
        val_dataset.append(val)
    
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)
    val_dataset = torch.utils.data.ConcatDataset(val_dataset)
    
    return train_dataset, val_dataset, test_dataset