import torch
import torch.nn as nn

import pickle

##(K^2) * C_in * H_out * W_out * C_out
#macs = 5*5*32*32*24*24
#print("MACs: ", macs)

'''
macs -      conv        lin         act
0               0          0           0
1         5529600                  73728
2        14745600                  18432
3                   3686400            0
4                     20800        18532

'''
'''
data
0 -------   84736
conv 1 -  2359296
conv 2 -   589824
mat.. -      3200
mat 2 -      0
'''
vgg_info = { # action No. : [layer, type num{1: conv, 2: fc, 3: act}, total, mac{1: conv, 2: fc, 3: act}, mid_data_size, partition point]
                0: [5, 13,  5,        20275200,      3707200,    110692,       84736,       0],
                1: [4, 13,  4,        14745600,      3707200,     36964,     2359296,       1],
                2: [3, 23,  3,               0,      3707200,     18532,      589824,       2],
                3: [2, 23,  2,               0,        20800,       100,        3200,       3],
                4: [1, 23,  1,               0,            0,         0,        3200,       4],
}


"""
torch.Size([1, 3, 96, 96])
partition point  0 tensor([[-1.0845,  0.0648,  2.2654, -1.1534]])
torch.Size([1, 32, 48, 48])
partition point  1 tensor([[-1.0845,  0.0648,  2.2654, -1.1534]])
torch.Size([1, 32, 24, 24])
partition point  2 tensor([[-1.0845,  0.0648,  2.2654, -1.1534]])
torch.Size([1, 100])
partition point  3 tensor([[-1.0845,  0.0648,  2.2654, -1.1534]])
torch.Size([1, 100])
partition point  4 tensor([[-1.0845,  0.0648,  2.2654, -1.1534]])
"""
if 0:
    import torch

    # Define the input tensor
    input_size = (100, 100)

    # Create the input tensor
    input_tensor = torch.randn(input_size)

    # Apply the ReLU operation
    output_tensor = torch.relu(input_tensor)

    # Count the FLOPs
    flops = input_tensor.numel()
    print("FLOPs:", flops)

data = [0.0,0.003791332,0.004397392,0.00520196,0.005397034,0.006002092]
if 1:
    with open('models\\vcFrontEndDelay.pkl', 'wb' ) as f:
            pickle.dump(data, f)#CPU_Unpickler(f).load()# pickle.load(f)
if 1:
    with open('models\\vcFrontEndDelay.pkl', 'rb') as f:
            print(pickle.load(f))