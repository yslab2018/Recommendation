import chainer
import numpy as np


def convert(batch, device):
    x = []
    y = []

    #==========
    # Use CPU
    #==========
    if (device is None) or (device < 0):
        
        for data in batch:
            # data      = [0,1,2,3,4]
            # data[:-1] = [0,1,2,3] -> input sequence  
            # data[-1]  = [4]       -> gold data  

            x.append(np.array(data[:-1], np.int32)) 
            y.append(data[-1])

        # convert numpy vec
        y = np.array(y, np.int32) 
            
    
    #==========
    # Use GPU
    #==========
    else:
        for data in batch:
            # data      = [0,1,2,3,4]
            # data[:-1] = [0,1,2,3] -> input sequence  
            # data[-1]  = [4]       -> gold data

            x.append(np.array(data[:-1], np.int32)) 
            y.append(data[-1])
            
        # convert numpy vec
        y = np.array(y, np.int32) 
        
    return x, y
