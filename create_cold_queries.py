import argparse
import os
import pickle
import random
from tqdm import tqdm

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from util.cold_helper import get_recursive_file_list, parse_file_list

parser = argparse.ArgumentParser()


parser.add_argument('--train_set', default='datasets/cold/train')
parser.add_argument('--test_set', default='datasets/cold/test/part_b_path_3')
parser.add_argument('--train_ref_filter', default='*seq[12]*[124]/std_cam/*.jpeg')
parser.add_argument('--train_query_filter', default='*seq[12]*[3]/std_cam/*.jpeg')
parser.add_argument('--test_ref_filter', default='*seq3*[12]/std_cam/*.jpeg')
parser.add_argument('--test_query_filter', default='*seq3*[3]/std_cam/*.jpeg')
parser.add_argument('--data_name', default='cold')

parser.add_argument('--out_dir', default='cold_pickles')
parser.add_argument('--max_set_size', type=int, default=100000)

parser.add_argument('--max_pos_radius', type=float, default=1)
parser.add_argument('--min_neg_radius', type=float, default=4)
parser.add_argument('--angle_threshold', type=float, default=0.50) # 3.14 = 180 degrees
FLAGS = parser.parse_args()

DATA_NAME = FLAGS.data_name
TRAIN_SET = FLAGS.train_set+'/'+DATA_NAME
TEST_SET = FLAGS.test_set
TRAIN_REF_FILTER = FLAGS.train_ref_filter
TEST_REF_FILTER = FLAGS.test_ref_filter
TRAIN_QUERY_FILTER = FLAGS.train_query_filter
TEST_QUERY_FILTER = FLAGS.test_query_filter
OUT_DIR = FLAGS.out_dir
MAX_POS_RADIUS = FLAGS.max_pos_radius
MIN_NEG_RADIUS = FLAGS.min_neg_radius
MAX_SET_SIZE = FLAGS.max_set_size
ANGLE_THRESHOLD = FLAGS.angle_threshold

def angle_correction(angle1, angle2):
    angle = abs(angle1 - angle2)
    if angle>3.14:
        angle = abs(6.28 - angle)
    return angle

def get_valid_files(path, pattern):
    all_files = get_recursive_file_list(path, pattern)
    all_files, TXYA = parse_file_list(all_files)
    return all_files, TXYA


def parse_cold_folder(path, pattern):
    all_files = get_recursive_file_list(path, pattern)
    all_files, TXYA, w_names = parse_file_list(all_files)
    all_parsed = pd.DataFrame(columns=['file', 't', 'x', 'y', 'a', 'w'])

    if len(all_files) > MAX_SET_SIZE:
        idxs_to_keep = np.linspace(0, len(all_files), num=MAX_SET_SIZE, endpoint=False, dtype=int)
    else:
        idxs_to_keep = np.arange(len(all_files))

    for i in tqdm(range(len(idxs_to_keep))):
        idx = idxs_to_keep[i]
        all_parsed = all_parsed.append(
            pd.DataFrame([[all_files[idx], TXYA[idx, 0], TXYA[idx, 1], TXYA[idx, 2], TXYA[idx, 3], w_names[idx]]],
                         columns=['file', 't', 'x', 'y', 'a', 'w'], index=[i]))
    return all_parsed


def construct_query_dict(df_centroids, filename, angle_threshold, train_data = True):
    if train_data:
        tree = KDTree(df_centroids[['x', 'y']])
        ind_nn = tree.query_radius(df_centroids[['x', 'y']], r=MAX_POS_RADIUS, return_distance=False)
        ind_r = tree.query_radius(df_centroids[['x', 'y']], r=MIN_NEG_RADIUS)
    queries = {}
    for i in tqdm(range(len(df_centroids))):
        query = df_centroids.iloc[i]["file"]
        t = df_centroids.iloc[i]["t"]
        x = df_centroids.iloc[i]["x"]
        y = df_centroids.iloc[i]["y"]
        a = df_centroids.iloc[i]["a"]
        w = df_centroids.iloc[i]["w"]
        if train_data:
            positives = np.setdiff1d(ind_nn[i], [i]).tolist()
            positives_final = []
            for idx in positives:
                pos_a = df_centroids.iloc[idx]["a"]
                if angle_correction(pos_a, a) < angle_threshold:
                    positives_final.append(idx)
    
            negatives = np.setdiff1d(df_centroids.index.values.tolist(), ind_r[i]).tolist()
            random.shuffle(negatives)
            if len(negatives)>10:
                negatives = negatives[:10]
            queries[i] = {"query": query, "positives": positives_final, "negatives": negatives, 't': t, 'x': x, 'y': y, 'a': a, 'w': w}
        else:
            queries[i] = {"query": query, 't': t, 'x': x, 'y': y, 'a': a, 'w': w}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)



def construct_queries():
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    log = open(os.path.join(OUT_DIR, 'query_flags.txt'), 'a')
    log.write(str(FLAGS) + '\n')

    train_ref_df = parse_cold_folder(TRAIN_SET, TRAIN_REF_FILTER)
    test_ref_df = parse_cold_folder(TEST_SET, TEST_REF_FILTER)
    train_query_df = parse_cold_folder(TRAIN_SET, TRAIN_QUERY_FILTER)
    test_query_df = parse_cold_folder(TEST_SET, TEST_QUERY_FILTER)

    log.write('{} train reference images\n'.format(len(train_ref_df)))
    log.write('{} test reference images\n'.format(len(test_ref_df)))
    log.write('{} train query images\n'.format(len(train_query_df)))
    log.write('{} test query images\n'.format(len(test_query_df)))

    construct_query_dict(train_ref_df, os.path.join(OUT_DIR, 'train_ref_'+DATA_NAME+'.pickle'), ANGLE_THRESHOLD)
    construct_query_dict(test_ref_df, os.path.join(OUT_DIR, "test_ref_s.pickle"), ANGLE_THRESHOLD, train_data=False)
    construct_query_dict(train_query_df, os.path.join(OUT_DIR, 'train_query_'+DATA_NAME+'.pickle'), ANGLE_THRESHOLD)
    construct_query_dict(test_query_df, os.path.join(OUT_DIR, "test_query_s.pickle"), ANGLE_THRESHOLD, train_data=False)



if __name__ == "__main__":
    construct_queries()
