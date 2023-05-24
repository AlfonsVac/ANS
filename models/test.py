import torch
import torch.nn as nn

import pickle

# Define the convolutional layer
conv_layer = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

##(K^2) * C_in * H_out * W_out * C_out
macs = 5*5*32*32*24*24
print("MACs: ", macs)

'''
macs -      conv        lin         act
conv 1 -- 5529600                  73728
conv 2 - 14745600                  18432
mat.. -             3686400        18432
mat 2 -               20000          100
mat 3 -                 800            0
'''
'''
data
0 -------   84736
conv 1 -  2359296
conv 2 -   589824
mat.. -      3200
mat 2 -      3200
mat 3 -         0
'''
vgg_info = { # action No. : [layer, type num{1: conv, 2: fc, 3: act}, total, mac{1: conv, 2: fc, 3: act}, mid_data_size, partition point]
                0: [5, 13,  5,        20275200,      3707200,    110692,       84736,       0],
                1: [4, 13,  4,        14745600,      3707200,     36964,     2359296,       1],
                2: [3, 23,  3,               0,      3707200,     18532,      589824,       2],
                3: [2, 23,  2,               0,        20800,       100,        3200,       3],
                4: [1, 23,  1,               0,          800,         0,        3200,       4],
                5: [0,  1,  0,               0,            0,         0,           0,       5]
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

data = [0.0, 0.010004281997680664, 0.051000356674194336, 0.07601261138916016, 0.09553170204162598, 0.1305232048034668, 0.13075470924377441, 0.15352702140808105, 0.163041353225708, 0.1870577335357666, 0.19806647300720215, 0.18957853317260742, 0.25607991218566895, 0.27457690238952637, 0.2698848247528076, 0.27856874465942383, 0.27063560485839844, 0.313493013381958, 0.28806400299072266, 0.3045930862426758, 0.31359004974365234, 0.32933664321899414, 0.29956984519958496]
if 1:
    with open('models\\vgg16FrontEndDelay.pkl', 'wb' ) as f:
            pickle.dump(data, f)#CPU_Unpickler(f).load()# pickle.load(f)
if 0:
    with open('models\\vgg16FrontEndDelay.pkl', 'rb') as f:
            print(pickle.load(f))