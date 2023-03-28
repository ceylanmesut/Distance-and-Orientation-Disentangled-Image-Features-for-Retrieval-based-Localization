# MIT License
#
# Copyright (c) 2018 Robotics and Perception Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
# The function in this file was modified from its original version as follows: Removed PCA

import numpy as np
import tensorflow as tf

import layers
# import netvlad_tf.layers as layers


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
