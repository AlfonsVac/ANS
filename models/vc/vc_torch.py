import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class vc(nn.Module):
    def __init__(self):
        super(vc, self).__init__()

        self.init_weights()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=1, padding='same', bias=False) #3, 32, 5, 1, 0, bias=False)
        self.conv1.weight = torch.nn.Parameter(self.w1)
        self.max_pool2d1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=1, padding='same', bias=False) #3, 32, 5, 1, 0, bias=False)
        self.conv2.weight = torch.nn.Parameter(self.w2)
        self.max_pool2d2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, server=True, partition=0):
        if server == True:
            if partition == 0:
                #first part
                x = self.max_pool2d1(F.relu(self.conv1(x)))
            if partition <= 1:
                #second part
                x = self.max_pool2d2(F.relu(self.conv2(x)))
            if partition <= 2:
                #reshape + third part
                x = torch.reshape(torch.t(torch.reshape(torch.reshape(x.permute(0,2,3,1),[24,24,32]),[24*24,32])), [1, 24*24*32])
                x = torch.matmul(x, self.W_d1)
            if partition <= 3:
                #fourth part
                x = F.relu(torch.matmul(F.relu(x), self.W_d2))
                x = torch.matmul(x, self.W_out)
            #if partition <= 4: nothing
        else:
            if partition == 0:
                # do nothing
                return x
            if partition >= 1:
                #first part
                x = self.max_pool2d1(F.relu(self.conv1(x)))
            if partition >= 2:
                #second part
                x = self.max_pool2d2(F.relu(self.conv2(x)))
            if partition >= 3:
                #reshape + third part
                x = torch.reshape(torch.t(torch.reshape(torch.reshape(x.permute(0,2,3,1),[24,24,32]),[24*24,32])), [1, 24*24*32])
                x = torch.matmul(x, self.W_d1)
            if partition >= 4:
                #fourth part
                x = F.relu(torch.matmul(F.relu(x), self.W_d2))
                x = torch.matmul(x, self.W_out)
        return x
    
    def init_weights(self):
        # Load weights
        self.w1 = torch.tensor(self.readRaw4D( 'models\\vc\\parameter\\conv1_update.bin', [5,5,3,32]), dtype=torch.float)
        #print(w1.shape)
        self.w2 = torch.tensor(self.readRaw4D('models\\vc\\parameter\\conv2_update.bin', [5,5,32,32]), dtype=torch.float)
        #print(w2.shape)
        self.W_d1 = torch.from_numpy(self.readRaw2D('models\\vc\\parameter\\ip3.bin', [24*24*32, 100]))
        self.W_d2 = torch.from_numpy(self.readRaw2D('models\\vc\\parameter\\ip4.bin', [100, 100]))
        self.W_out = torch.from_numpy(self.readRaw2D('models\\vc\\parameter\\ip_last.bin', [100, 4]))

    def readRaw4D(self, filename, size):
        wcomp_raw = np.fromfile(filename, dtype='float32')
        #print(wcomp_raw.shape)
        flipped = np.fliplr(np.flip(np.reshape(wcomp_raw, size, order='F'),0))#
        transposed = np.transpose(flipped, axes=[3,2, 1,0]).astype('float32')
        #print('transposed shape: ', transposed.shape)
        return transposed
        transposed = np.transpose(flipped, axes=[1,0,2,3]).astype('float32')
        return np.moveaxis(transposed, [-1, -2], [0, 1])

    def readRaw3D(self, filename, X, Y, Z):
        D_vector1 = np.fromfile(filename, dtype='float32')
        D_matrix2 = np.reshape(D_vector1, [X, Y, Z], order='F')
        D_rotated = np.rot90(D_matrix2, -1)
        D_mirrored = np.fliplr(D_rotated)
        return D_mirrored

    def readRaw2D(self, filename, size):
        wcomp_raw = np.fromfile(filename, dtype='float32')
        return np.reshape(wcomp_raw, size, order='F')

    def getTestData(self):
        X_test=[]

        X_plaf = self.readRaw3D('models\\vc\\input\\c.bin', 96, 96, 3)
        X_test.append(np.transpose(X_plaf, axes=[2, 0, 1]))
        return np.asarray(X_test)
"""
# Load test data
dev_im = 'getTestData'
#print(dev_im.shape)
dev_im_tensor = torch.from_numpy(dev_im)
print(dev_im_tensor.shape)

start = time.time()
output = vc(dev_im_tensor)
end = time.time()

# Printing results
print('y_out for white truck image:')
print(output.detach().numpy())

print('y_softmax:')
print(np.round(F.softmax(output, -1).detach().numpy()))

print('computation time:')
print(end - start)

"""

if __name__ == '__main__':
    print('test partition points in vc!!!')

    import json
    import torchvision.transforms as transforms
    from PIL import Image

    model = vc()
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.to(torch.device("cpu"))

    min_img_size = 96
    transform_pipeline = transforms.Compose([transforms.Resize((min_img_size, min_img_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    if 1:
        img = Image.open('models\\Golden_Retriever_Hund_Dog.jpg')
        img = transform_pipeline(img)
        img = img.unsqueeze(0)
    else:
        img = torch.from_numpy(model.getTestData())
    #print(img.shape)
    times = []
    for partition in range(6):
        with torch.no_grad():
            start = time.time()
            intermediate = model(img, server=False, partition=partition)
            print(intermediate.shape)
            end = time.time()
            prediction = model(intermediate, server=True, partition=partition)
            times.append(end - start)
            #print('partition point ', partition, '  result:', np.round(torch.softmax(prediction, 1).clone().detach().numpy()))
    print(times)