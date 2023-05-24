# Vehicle classifier for Tensorflow 2
# Authors: Mir Khan, mir.khan@tuni.fi
#          Jani Boutellier, jani.boutellier@uwasa.fi

import numpy as np
import time
import tensorflow as tf


@tf.function
def conv2d(x, W):
  return tf.nn.conv2d(x,
                      W,
                      strides=[1, 1],
                      padding='SAME')
@tf.function
def max_pool_2x2(x):
  return tf.nn.max_pool(x,
                        ksize=2,
                        strides=2,
                        padding='SAME')

def readRaw4D(filename, size):
    wcomp_raw = np.fromfile(filename, dtype='float32')
    flipped = np.fliplr(np.flip(np.reshape(wcomp_raw, size, order='F'),0))
    return np.transpose(flipped, axes=[1,0,2,3]).astype('float32')

def readRaw3D(fileName, X, Y, Z): # 96, 96, 3
    D_vector1 = np.fromfile(fileName, dtype='float32')
    D_matrix2 = np.reshape(D_vector1, [X, Y, Z], order='F')
    D_rotated = np.rot90(D_matrix2,-1)
    D_mirrored = np.fliplr(D_rotated)
    return D_mirrored

def readRaw2D(filename, size):
    wcomp_raw = np.fromfile(filename, dtype='float32')
    return np.reshape(wcomp_raw, size, order='F')

def getTestData():
    X_test=[]

    X_plaf = readRaw3D('input/c.bin', 96, 96, 3)

    X_test.append(X_plaf)
    return np.asarray(X_test)

def testgraph(X, w1, w2, W_d1, W_d2, W_out):
	W_conv1 = tf.Variable(w1, dtype=tf.float32)
	h_conv1 = tf.nn.relu(conv2d(X, W_conv1))
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = tf.Variable(w2,dtype=tf.float32)
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)) 
	h_pool2 = max_pool_2x2(h_conv2)

	h_pool2_flat=tf.reshape(tf.transpose(tf.reshape(tf.reshape(h_pool2,[24,24,32]),[24*24,32])), [1, 24*24*32])
	h_d1 = tf.nn.relu(tf.matmul(h_pool2_flat, tf.Variable(W_d1,dtype=tf.float32)))

	h_d2 = tf.nn.relu(tf.matmul(h_d1, tf.Variable(W_d2,dtype=tf.float32)))


	y_out = tf.matmul(h_d2, tf.Variable(W_out,dtype=tf.float32))
	return y_out


#____________________________________
#------------- main() ------------- #
#____________________________________

print('Constructing network\n')
dev_im = getTestData()
w1 = readRaw4D('parameter/conv1_update.bin', [5,5,3,32])
w2 = readRaw4D('parameter/conv2_update.bin', [5,5,32,32])
W_d1 = readRaw2D('parameter/ip3.bin',[24*24*32,100])
W_d2 = readRaw2D('parameter/ip4.bin',[100,100])
W_out = readRaw2D('parameter/ip_last.bin',[100,4])

start = time.time()
res = testgraph(dev_im, w1, w2, W_d1, W_d2, W_out)
end = time.time()
print('shape first', np.moveaxis(res, -1, 1).shape)
#np.savetxt('data_tf.csv', np.moveaxis(res, -1, 1)[0][0], delimiter=';')
# printing the execution time by subtracting 
# the time before the function from
# the time after the function

print('y_out for white truck image')   
print(np.round(res, 4))

print('y_softmax')   
print(np.round(tf.nn.softmax(res), 4))

print('computation time')   
print(end-start)

"""
conv2d_test = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5),strides=(1,1), padding='SAME')
conv2d_test.build(input_shape = (96, 96, 3))
print(w1.shape)
#conv2d_test.set_weights(np.asarray(w1))
print(conv2d_test.get_weights()[0].shape)
"""
