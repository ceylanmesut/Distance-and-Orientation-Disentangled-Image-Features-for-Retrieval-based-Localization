import argparse
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle

from src.nets1 import vgg16NetvladPca, vgg16Netvlad
from util.cold_helper import get_file_list
from util.loading_touples import load_img
from util import utils

parser = argparse.ArgumentParser()
# Paths
parser.add_argument('--checkpoint', default='checkpoints/vd16_pitts30k_conv5_3_vlad_preL2_intra_white/vd16_pitts30k_conv5_3_vlad_preL2_intra_white')
parser.add_argument('--query_folder', default='cold_pickles')
parser.add_argument('--use_wpca', type=bool, default=False)

FLAGS = parser.parse_args()

CHECKPOINT = FLAGS.checkpoint
QUERY_FOLDER = FLAGS.query_folder
USE_WPCA = FLAGS.use_wpca
step = 50

with tf.Session() as sess:
    net_in = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
    if USE_WPCA:
        net_out = vgg16NetvladPca(net_in)
    else:
        net_out = vgg16Netvlad(net_in)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT)

    set_names = get_file_list(QUERY_FOLDER, '*.pickle')

    for set_name in set_names:
        print(set_name)

        out_dir = os.path.splitext(set_name)[0] + '_' + os.path.basename(CHECKPOINT) + '_latent_vectors'

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        LOG_OUT = open(os.path.join(out_dir, 'latent_vectors_log.txt'), 'w')
        LOG_OUT.write(str(FLAGS) + '\n')

        with open(set_name, 'rb') as handle:
            queries = pickle.load(handle)

        features = []
        actual_angles = []

        for i in range(0,len(queries),step):
            print('Processing image {} of {}.'.format(i, len(queries)))
            image = load_img(queries[i]['query'])
            image = np.expand_dims(image, axis=0)
            feature = sess.run(net_out, feed_dict={net_in: image})
            features.append(feature)
            actual_angles.append(queries[i]['a'])

        with open(os.path.join(out_dir, 'latent_vectors_{}.pickle'.format(step)), 'wb') as handle:
            pickle.dump([queries, features], handle)

        # Get pairwise difference:
        num_dims = features[0].shape[1]
        pairwise_dists = []
        pairwise_f_angles = []
        pairwise_actual_angles = []
        for i in range(len(features)):
            for j in range(i):
                dist = np.sum((features[i][:,:num_dims//2] - features[j][:,:num_dims//2]) ** 2)
                pairwise_dists.append(dist)

                f_angle = np.sum((features[i][:,num_dims//2:] - features[j][:,num_dims//2:]) ** 2)
                pairwise_f_angles.append(f_angle)

                angle_diff = utils.angle_correction(actual_angles[i], actual_angles[j])
                pairwise_actual_angles.append(angle_diff**2) # Adding squared angle difference

        pairwise_dists = np.asarray(pairwise_dists)
        pairwise_f_angles = np.asarray(pairwise_f_angles)
        pairwise_actual_angles = np.asarray(pairwise_actual_angles)

        mean_dist = np.mean(pairwise_dists)
        max_dist = np.max(pairwise_dists)
        min_dist = np.min(pairwise_dists)

        mean_f_angle = np.mean(pairwise_f_angles)
        max_f_angle = np.max(pairwise_f_angles)
        min_f_angle = np.min(pairwise_f_angles)

        mean_actual_angle = np.mean(pairwise_actual_angles)
        max_actual_angle = np.max(pairwise_actual_angles)
        min_actual_angle = np.min(pairwise_actual_angles)

        LOG_OUT.write('Mean feature distance: {}\n'.format(mean_dist))
        LOG_OUT.write('Max feature distance: {}\n'.format(max_dist))
        LOG_OUT.write('Min feature distance: {}\n'.format(min_dist))

        LOG_OUT.write('Mean feature angle: {}\n'.format(mean_f_angle))
        LOG_OUT.write('Max feature angle: {}\n'.format(max_f_angle))
        LOG_OUT.write('Min feature angle: {}\n'.format(min_f_angle))

        LOG_OUT.write('Mean feature angle: {}\n'.format(mean_actual_angle))
        LOG_OUT.write('Max feature angle: {}\n'.format(max_actual_angle))
        LOG_OUT.write('Min feature angle: {}\n'.format(min_actual_angle))

        with open(os.path.join(out_dir, 'max_dist_{}.pickle'.format(step)), 'wb') as handle:
            pickle.dump([max_dist], handle)
        with open(os.path.join(out_dir, 'max_f_angle_{}.pickle'.format(step)), 'wb') as handle:
            pickle.dump([max_f_angle], handle)
        with open(os.path.join(out_dir, 'max_actual_angle_{}.pickle'.format(step)), 'wb') as handle:
            pickle.dump([max_actual_angle], handle)

        utils.export_plots(pairwise_dists, out_dir, 'pairwise_feature_distances')
        utils.export_plots(pairwise_actual_angles, out_dir, 'pairwise_actual_angles')
        utils.export_plots(pairwise_f_angles, out_dir, 'pairwise_feature_angles')
