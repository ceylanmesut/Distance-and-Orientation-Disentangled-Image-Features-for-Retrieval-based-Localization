import random

import numpy as np
import os
import pickle
from img_helper import load_imgs, load_img

import utils

def get_xy(queries):
    XY = np.empty([len(queries), 2], dtype=float)
    for i in range(len(queries)):
        XY[i, :] = [queries[i]['x'], queries[i]['y']]
    return XY


def get_t(queries):
    T = np.empty([len(queries)], dtype=float)
    for i in range(len(queries)):
        T[i] = queries[i]['t']
    return T

def load_queries(query_folder, filename):
    with open(os.path.join(query_folder, filename), 'rb') as handle:
        queries = pickle.load(handle)
    
    return queries


def get_query_tuple(dict_value, num_pos, num_neg, QUERY_DICT, easy_threshold, hard_neg=[], other_neg=False, 
                    add_pos_dists=False, add_pos_angle=False, use_hard_positives=False):
    # get query tuple for dictionary entry
    # return list [query,positives,negatives]

    query = load_img(dict_value["query"])  # Nx3

    random.shuffle(dict_value["positives"])
    pos_files = []
    pos_dists = []
    pos_angles = []
    
    if use_hard_positives:
        easy_counter = 0
        hard_counter = 0
        counter = 0
        i = 0
        while i<len(dict_value['positives']):
            if counter >= num_pos:
                break
            pos_dist = (dict_value['x'] - QUERY_DICT[dict_value["positives"][i]]["x"]) ** 2 + (dict_value['y'] - QUERY_DICT[dict_value["positives"][i]]["y"]) ** 2
            pos_angle = utils.angle_correction(dict_value['a'], QUERY_DICT[dict_value['positives'][i]]['a'])
            
            if easy_counter < num_pos//2 and pos_angle < easy_threshold:
                pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
                if add_pos_dists:
                    pos_dists.append(pos_dist)
                if add_pos_angle:
                    pos_angles.append(pos_angle)
                easy_counter += 1
            
            if hard_counter < num_pos - num_pos//2 and pos_angle >= easy_threshold:
                pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
                if add_pos_dists:
                    pos_dists.append(pos_dist)
                if add_pos_angle:
                    pos_angles.append(pos_angle)
                hard_counter += 1
            
            counter = easy_counter + hard_counter
            i += 1
        
        if len(pos_files) < num_pos:
            for i in range(len(dict_value["positives"])):
                pos_file = QUERY_DICT[dict_value["positives"][i]]["query"]
                if pos_file not in pos_files:
                    pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
                    if add_pos_dists:
                        pos_dists.append(pos_dist)
                    if add_pos_angle:
                        pos_angles.append(pos_angle)
                    
                    if len(pos_files) == num_pos:
                        break
    
    
    else:
        for i in range(num_pos):
            pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
            if add_pos_dists:
                pos_dists.append((dict_value['x'] - QUERY_DICT[dict_value["positives"][i]]["x"]) ** 2 +
                                 (dict_value['y'] - QUERY_DICT[dict_value["positives"][i]]["y"]) ** 2)
            if add_pos_angle:
                angle_diff = utils.angle_correction(dict_value['a'], QUERY_DICT[dict_value['positives'][i]]['a'])
                angle_diff = angle_diff**2 # Adding squared angle difference
                pos_angles.append(angle_diff)

    positives = load_imgs(pos_files)

    neg_files = []
    neg_indices = []
    if isinstance(hard_neg, int):
        hard_neg = [hard_neg]  # Getting int where list is needed

    if len(hard_neg) == 0:
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
            neg_indices.append(dict_value["negatives"][i])
    else:
        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j = 0
        while len(neg_files) < num_neg:

            if not dict_value["negatives"][j] in hard_neg:
                neg_files.append(QUERY_DICT[dict_value["negatives"][j]]["query"])
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    negatives = load_imgs(neg_files)

    if not other_neg:
        output = [query, positives, negatives]
    # For Quadruplet Loss
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs = list(set(QUERY_DICT.keys()) - set(neighbors))
        random.shuffle(possible_negs)

        if len(possible_negs) == 0:
            output = [query, positives, negatives, np.array([])]
        else:
            neg2 = load_img(QUERY_DICT[possible_negs[0]]["query"])
            output = [query, positives, negatives, neg2]

    if add_pos_dists:
        output = output + [pos_dists]
    if add_pos_angle:
        output = output + [pos_angles]
    return output
