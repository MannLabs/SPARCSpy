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