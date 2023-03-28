import numpy as np
import os
from os.path import dirname
import tensorflow as tf

import layers

def defaultCheckpoint():
    return os.path.join(dirname(dirname(dirname(__file__))), 
                              'checkpoints', 
                              'vd16_pitts30k_conv5_3_vlad_preL2_intra_white')

def vgg16NetvladPca(image_batch):
    ''' Assumes rank 4 input, first 3 dims fixed or dynamic, last dim 1 or 3. 
    '''

    # MAKING SURE BATCH SIZE IS EQUAL TO 4 IMAGES.
    assert len(image_batch.shape) == 4
    
    with tf.variable_scope('vgg16_netvlad_pca'):
        # Gray to color if necessary.
        if image_batch.shape[3] == 1:
            x = tf.nn.conv2d(image_batch, np.ones((1, 1, 1, 3)), 
                                  np.ones(4).tolist(), 'VALID')
        else :
            assert image_batch.shape[3] == 3
            x = image_batch
        
        # Subtract trained average image.
        average_rgb = tf.get_variable(
                'average_rgb', 3, dtype=image_batch.dtype)
        x = x - average_rgb
        
        # VGG16
        def vggConv(inputs, numbers, out_dim, with_relu):
            if with_relu:
                activation = tf.nn.relu
            else:
                activation = None
            return tf.layers.conv2d(inputs, out_dim, [3, 3], 1, padding='same',
                                    activation=activation, 
                                    name='conv%s' % numbers)

        # APPLIED MAX POOLING 2,2
        def vggPool(inputs):
            return tf.layers.max_pooling2d(inputs, 2, 2)
        
        x = vggConv(x, '1_1', 64, True)
        x = vggConv(x, '1_2', 64, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '2_1', 128, True)
        x = vggConv(x, '2_2', 128, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '3_1', 256, True)
        x = vggConv(x, '3_2', 256, True)
        x = vggConv(x, '3_3', 256, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '4_1', 512, True)
        x = vggConv(x, '4_2', 512, True)
        x = vggConv(x, '4_3', 512, False)
        x = vggPool(x)
        x = tf.nn.relu(x)
        
        x = vggConv(x, '5_1', 512, True)
        x = vggConv(x, '5_2', 512, True)
        x = vggConv(x, '5_3', 512, False)


        # L2 NORM NORMALIZATION RETURNS TENSOR WITH DIM X.
        # NetVLAD
        x = tf.nn.l2_normalize(x, dim=-1)
        x = layers.netVLAD(x, 64)
        
        # PCA
        x = tf.layers.conv2d(tf.expand_dims(tf.expand_dims(x, 1), 1), 
                             4096, 1, 1, name='WPCA')
        x = tf.nn.l2_normalize(tf.layers.flatten(x), dim=-1)
        
    return x


# Modified from original: Removed PCA
def vgg16Netvlad(image_batch):
    ''' Assumes rank 4 input, first 3 dims fixed or dynamic, last dim 1 or 3.
    '''
    assert len(image_batch.shape) == 4

    with tf.variable_scope('vgg16_netvlad_pca'):
        # Gray to color if necessary.
        if image_batch.shape[3] == 1:
            x = tf.nn.conv2d(image_batch, np.ones((1, 1, 1, 3)),
                             np.ones(4).tolist(), 'VALID')
        else:
            assert image_batch.shape[3] == 3
            x = image_batch

        # Subtract trained average image.
        average_rgb = tf.get_variable(
            'average_rgb', 3, dtype=image_batch.dtype)
        x = x - average_rgb

        # VGG16
        def vggConv(inputs, numbers, out_dim, with_relu):
            if with_relu:
                activation = tf.nn.relu
            else:
                activation = None
            return tf.layers.conv2d(inputs, out_dim, [3, 3], 1, padding='same',
                                    activation=activation,
                                    name='conv%s' % numbers)

        def vggPool(inputs):
            return tf.layers.max_pooling2d(inputs, 2, 2)

        x = vggConv(x, '1_1', 64, True)
        x = vggConv(x, '1_2', 64, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '2_1', 128, True)
        x = vggConv(x, '2_2', 128, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '3_1', 256, True)
        x = vggConv(x, '3_2', 256, True)
        x = vggConv(x, '3_3', 256, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '4_1', 512, True)
        x = vggConv(x, '4_2', 512, True)
        x = vggConv(x, '4_3', 512, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x = vggConv(x, '5_1', 512, True)
        x = vggConv(x, '5_2', 512, True)
        x = vggConv(x, '5_3', 512, False)

        # NetVLAD
        x = tf.nn.l2_normalize(x, dim=-1)
        x = layers.netVLAD(x, 64)

    return x

def vgg16Netvlad_all_layers(image_batch):
    ''' Assumes rank 4 input, first 3 dims fixed or dynamic, last dim 1 or 3.
    '''
    assert len(image_batch.shape) == 4

    with tf.variable_scope('vgg16_netvlad_pca'):
        # Gray to color if necessary.
        if image_batch.shape[3] == 1:
            x = tf.nn.conv2d(image_batch, np.ones((1, 1, 1, 3)),
                             np.ones(4).tolist(), 'VALID')
        else:
            assert image_batch.shape[3] == 3
            x = image_batch

        # Subtract trained average image.
        average_rgb = tf.get_variable(
            'average_rgb', 3, dtype=image_batch.dtype)
        x = x - average_rgb

        # VGG16
        def vggConv(inputs, numbers, out_dim, with_relu):
            if with_relu:
                activation = tf.nn.relu
            else:
                activation = None
            return tf.layers.conv2d(inputs, out_dim, [3, 3], 1, padding='same',
                                    activation=activation,
                                    name='conv%s' % numbers)

        def vggPool(inputs):
            return tf.layers.max_pooling2d(inputs, 2, 2)

        x = vggConv(x, '1_1', 64, True)
        x = vggConv(x, '1_2', 64, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x1 = vggConv(x, '1_2_skip_vgg', 512, False)
        x1 = tf.nn.l2_normalize(x1, dim=-1)
        x1 = layers.netVLAD_all_layers(x1, 64, '1_2_skip_vgg', 'cluster_centers_1_2_skip_vgg')

        x = vggConv(x, '2_1', 128, True)
        x = vggConv(x, '2_2', 128, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x2 = vggConv(x, '2_2_skip_vgg', 512, False)
        x2 = tf.nn.l2_normalize(x2, dim=-1)
        x2 = layers.netVLAD_all_layers(x2, 64, '2_2_skip_vgg', 'cluster_centers_2_2_skip_vgg')

        x = vggConv(x, '3_1', 256, True)
        x = vggConv(x, '3_2', 256, True)
        x = vggConv(x, '3_3', 256, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x3 = vggConv(x, '3_3_skip_vgg', 512, False)
        x3 = tf.nn.l2_normalize(x3, dim=-1)
        x3 = layers.netVLAD_all_layers(x3, 64, '3_3_skip_vgg', 'cluster_centers_3_3_skip_vgg')

        x = vggConv(x, '4_1', 512, True)
        x = vggConv(x, '4_2', 512, True)
        x = vggConv(x, '4_3', 512, False)
        x = vggPool(x)
        x = tf.nn.relu(x)

        x4 = vggConv(x, '4_3_skip_vgg', 512, False)
        x4 = tf.nn.l2_normalize(x4, dim=-1)
        x4 = layers.netVLAD_all_layers(x4, 64, '4_3_skip_vgg', 'cluster_centers_4_3_skip_vgg')

        x = vggConv(x, '5_1', 512, True)
        x = vggConv(x, '5_2', 512, True)
        x = vggConv(x, '5_3', 512, False)

        # NetVLAD
        x = tf.nn.l2_normalize(x, dim=-1)
        x = layers.netVLAD_all_layers(x, 64, 'assignment', 'cluster_centers')

        x = x + x1 + x2 + x3 + x4
        x = layers.matconvnetNormalize(x, 1e-12)

    return x