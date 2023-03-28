import os
import re
import tensorflow as tf
import numpy as np
import sklearn
import matplotlib as mpl
import json

mpl.use('Agg')
import matplotlib.pyplot as plt

from img_helper import load_img, merge_images, put_text, save_img
from cold_helper import get_t_x_y_a_from_path


def create_save_dir(args, timestamp):
    experiment_name_format = '{}_tf_td{}_{}_b{}_e{}_d{}_w{}_a{}_{}'
    experiment_name = experiment_name_format.format(timestamp, args.data_name, args.loss, args.queries_per_batch, 
                                                    args.max_epoch, args.dropout_rate, args.window_size, 
                                                    args.angle_threshold, args.message)
    save_dir = os.path.join(args.out_dir, experiment_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    return save_dir


def save_config(args, use_dists, use_angle, log_path, fpath, save_dir):
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config['use_dists'] = use_dists
    config['use_angle'] = use_angle
    config['log_path'] = log_path
    config['save_dir'] = save_dir
    json.dump(config, open(fpath, 'w'), indent=4, sort_keys=True)


def load_checkpoint(sess, train_folder):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(train_folder, latest_filename='epoch-checkpoint')
    
    saver.restore(sess, ckpt.model_checkpoint_path)

def export_plots(data, out_dir, plot_name):
    plt.clf()
    plt.hist(data)
    plt.savefig(os.path.join(out_dir, plot_name+'_{}.pdf'.format(50)))

def get_bn_decay(batch, config):
    bn_momentum = tf.train.exponential_decay(
        config['bn_init_decay'],
        batch * config['queries_per_batch'],
        config['bn_decay_step'],
        config['bn_decay_rate'],
        staircase=True)
    bn_decay = tf.minimum(config['bn_decay_clip'], 1 - bn_momentum)
    return bn_decay


def log_string(out_str, fpath):
    LOG_FOUT = open(fpath, 'w')
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    LOG_FOUT.close()
    print(out_str)


def get_learning_rate(epoch, config):
    learning_rate = config['base_lr'] * (config['lr_down_factor'] ** (epoch // config['lr_down_frequency']))
    learning_rate = tf.maximum(learning_rate, config['minimal_lr'])
    return learning_rate

def angle_correction(angle1, angle2):
    angle = abs(angle1 - angle2)
    if angle>3.14:
        angle = abs(6.28 - angle)
    return angle

def get_gpu_config():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.gpu_options.polling_inactive_delay_msecs = 10
    config.allow_soft_placement = True
    config.log_device_placement = False

    return config

def check_faulty_tuple(train_ref_queries, config):
    faulty_tuple_pos = False
    faulty_tuple_neg = False
    if len(train_ref_queries["positives"]) < config['positives_per_query']:
        faulty_tuple_pos = True

    if len(train_ref_queries["negatives"]) < config['negatives_per_query']:
        faulty_tuple_neg = True

    return faulty_tuple_pos, faulty_tuple_neg

def print_faulty_tuple_string(faulty_tuple_pos, faulty_tuple_neg, no_other_neg, fpath, batch_num):
    if faulty_tuple_pos:
        log_string('----' + str(batch_num) + '-----', fpath)
        log_string('----' + 'FAULTY TUPLE (not enough positives)' + '-----', fpath)

    if faulty_tuple_neg:
        log_string('----' + str(batch_num) + '-----', fpath)
        log_string('----' + 'FAULTY TUPLE (not enough negatives)' + '-----', fpath)

    if no_other_neg:
        log_string('----' + str(batch_num) + '-----', fpath)
        log_string('----' + 'NO OTHER NEG' + '-----', fpath)

def get_input_data(config, q_tuples, use_dists, use_angle, batch_num, fpath):

    NEGATIVES_PER_QUERY = config['negatives_per_query']
    QUERIES_PER_BATCH = config['queries_per_batch']
    POSITIVES_PER_QUERY = config['positives_per_query']
    
    queries = []
    positives = []
    negatives = []
    other_neg = []
    dists = []
    angles = []

    for k in range(len(q_tuples)):
        queries.append(q_tuples[k][0])
        positives.append(q_tuples[k][1])
        negatives.append(q_tuples[k][2])
        other_neg.append(q_tuples[k][3])
        if use_dists:
            dists.append(q_tuples[k][4])
        if use_angle and not use_dists:
            angles.append(q_tuples[k][4])
        if use_angle and use_dists:
            angles.append(q_tuples[k][5])

    queries = np.array(queries)
    other_neg = np.array(other_neg)
    positives = np.array(positives)
    negatives = np.array(negatives)
    if use_dists:
        dists = np.array(dists)
    if use_angle:
        angles = np.array(angles)

    log_string('----' + str(batch_num) + '-----', fpath)

    positives = np.reshape(positives, (QUERIES_PER_BATCH * POSITIVES_PER_QUERY, positives.shape[2], positives.shape[3], 3))
    negatives = np.reshape(negatives, (QUERIES_PER_BATCH * NEGATIVES_PER_QUERY, negatives.shape[2], negatives.shape[3], 3))

    data = {'queries': queries, 'positives': positives, 'negatives': negatives, 'other_neg': other_neg}
    if use_dists:
        dists = np.reshape(dists, (QUERIES_PER_BATCH, POSITIVES_PER_QUERY))
        data['dists'] = dists
    if use_angle:
        angles = np.reshape(angles, (QUERIES_PER_BATCH, POSITIVES_PER_QUERY))
        data['angles'] = angles

    return data

def get_idx_to_keep(queries, set_size):
    if len(queries) > set_size:
        idxs_to_keep = np.linspace(0, len(queries), num=set_size, endpoint=False, dtype=int)
    else:
        idxs_to_keep = np.arange(len(queries))

    return idxs_to_keep

def downsample_queries(queries, indices):
    new_queries = {new_index: {'x': queries[old_index]['x'],
                               'y': queries[old_index]['y'],
                               'query': queries[old_index]['query'],
                               'a': queries[old_index]['a']}
                   for new_index, old_index in enumerate(indices)}
    return new_queries

def get_d_to_nearest_latent_matrix(q_queries, nearest_latent_indices, query_XY, ref_XY):
    d_to_nearest_latent = np.empty(nearest_latent_indices.shape)
    for i in range(len(q_queries)):
        for j in range(nearest_latent_indices.shape[1]):
            other_index = nearest_latent_indices[i][j]
            d_to_nearest_latent[i, j] = np.linalg.norm(query_XY[i, :] - ref_XY[other_index, :])
        
    return d_to_nearest_latent

def draw_plots(d_to_nearest_latent, nearest_d_dist, out_name, writer, out_dir, mode='train'):
    OUT_DIR = os.path.join(out_dir, mode, 'plots')
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    num_queries, num_neighbours = d_to_nearest_latent.shape
    top_n = np.empty(d_to_nearest_latent.shape)
    for i in range(num_queries):
        for j in range(num_neighbours):
            top_n[i, j] = min(d_to_nearest_latent[i, 0:(j + 1)])

    # Write results to summary
    x = [[] for _ in range(top_n.shape[1] + 1)]
    y = [[] for _ in range(top_n.shape[1] + 1)]

    # Sorting top_n
    for n in range(top_n.shape[1]):
        x[n] = np.sort(top_n[:, n])
        y[n] = np.array(range(num_queries)) / float(num_queries)
    x[-1] = np.sort(np.array(nearest_d_dist).reshape(-1))
    y[-1] = np.array(range(num_queries)) / float(num_queries)

    summary = tf.Summary()
    full_auc = sklearn.metrics.auc(x[0], y[0])
    summary.value.add(tag='full_auc@Top1', simple_value=full_auc)

    x_partial = x  # not actually a copy (x will be changed when modifiying x_partial)
    y_partial = y
    for rad in [100, 50, 30, 25, 10, 4, 1]:

        plt.clf()
        for n in range(top_n.shape[1] + 1):
            x_partial[n] = [p for p in x_partial[n] if p <= rad]
            y_partial[n] = [y_partial[n][k] for k in range(len(x_partial[n]))]
            plt.plot(x_partial[n], y_partial[n])
        plt.legend(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5', 'Optimum'])
        plt.ylabel('Correctly localized')
        plt.xlabel('Tolerance [m]')
        plt.xlim(0, rad)
        plt.title(out_name)

        correct_fraction = len(x_partial[0]) / len(top_n[:, 0])
        summary.value.add(tag='%<{}m@Top1'.format(rad), simple_value=correct_fraction)
        plt.text(0.5 * float(rad), 0.02, '%<{}m@Top1={:7.2f}'.format(rad, correct_fraction))

        plt.savefig(os.path.join(OUT_DIR, mode + '_' + out_name + '_{}.pdf'.format(rad)))

    step = int(re.findall(r'\d+', out_name)[0])
    writer.add_summary(summary, step)

def create_query_img(idx, q_queries):
    # Query image
    file_path = q_queries[idx]['query']
    query_angle = q_queries[idx]['a']

    query_image = load_img(file_path)
    query_image = put_text('Query', 35, query_image)

    return query_image, query_angle

def create_retrieved_img(idx, ref_queries, nearest_latent_indices, ref_latent_vectors, d_to_nearest_latent, nearest_latent_dists, 
                         query_angle, query_latent_vectors, num_dims):
    file_path = ref_queries[nearest_latent_indices[idx][0]]['query']
    retrieved_image_angle = ref_queries[nearest_latent_indices[idx][0]]['a']
    retrieved_image_latent_vec = ref_latent_vectors[nearest_latent_indices[idx][0]]

    retrieved_image = load_img(file_path)
    dist = d_to_nearest_latent[idx][0]
    latent_dist = np.linalg.norm(query_latent_vectors[idx][:num_dims//2] - retrieved_image_latent_vec[:num_dims//2])

    angle = angle_correction(query_angle, retrieved_image_angle)
    latent_angle = np.linalg.norm(query_latent_vectors[idx][num_dims//2:] - retrieved_image_latent_vec[num_dims//2:]) 

    retrieved_image = put_text('Retrieved {:.2f}'.format(dist), 35, retrieved_image)
    retrieved_image = put_text('Latent Dist {:.2f}'.format(latent_dist), 50, retrieved_image)
    retrieved_image = put_text('Angle {:.2f}'.format(angle), 65, retrieved_image)
    retrieved_image = put_text('Latent Angle {:.2f}'.format(latent_angle), 80, retrieved_image)

    return retrieved_image, file_path

def create_optimal_img(idx, ref_queries, ref_latent_vectors, query_angle, query_latent_vectors, nearest_d_dist, 
                       nearest_d_indices, num_dims):
    file_path = ref_queries[nearest_d_indices[idx][0]]['query']
    optimal_angle = ref_queries[nearest_d_indices[idx][0]]['a']
    optimal_image_latent_vec = ref_latent_vectors[nearest_d_indices[idx][0]]

    optimal_image = load_img(file_path)
    dist = nearest_d_dist[idx][0]
    latent_dist = np.linalg.norm(query_latent_vectors[idx][:num_dims//2] - optimal_image_latent_vec[:num_dims//2])
    
    angle = angle_correction(query_angle, optimal_angle)
    latent_angle = np.linalg.norm(query_latent_vectors[idx][num_dims//2:] - optimal_image_latent_vec[num_dims//2:])

    optimal_image = put_text('Optimal {:.2f}'.format(dist), 35, optimal_image)
    optimal_image = put_text('Latent Dist {:.2f}'.format(latent_dist), 50, optimal_image)
    optimal_image = put_text('Angle {:.2f}'.format(angle), 65, optimal_image)
    optimal_image = put_text('Latent Angle {:.2f}'.format(latent_angle), 80, optimal_image)

    return optimal_image


def save_visual_examples(q_queries, query_latent_vectors, ref_queries, ref_latent_vectors, nearest_latent_dists, 
                         nearest_latent_indices, d_to_nearest_latent, nearest_d_dist, nearest_d_indices, 
                         out_dir, mode, out_name):
    num_dims = query_latent_vectors.shape[-1]
    example_img_dir = os.path.join(out_dir, mode, out_name) ## adding latent vectors
    os.makedirs(example_img_dir)
    example_idxs = np.arange(0, len(q_queries))
    np.random.shuffle(example_idxs)
    for i in range(10):
        idx = example_idxs[i]

        # Query image
        query_image, query_angle = create_query_img(idx, q_queries)

        # Retrieved image
        retrieved_image, retr_img_path = create_retrieved_img(idx, ref_queries, nearest_latent_indices, ref_latent_vectors, 
                                               d_to_nearest_latent, nearest_latent_dists, query_angle, query_latent_vectors, num_dims)    

        # Optimal image
        optimal_image = create_optimal_img(idx, ref_queries, ref_latent_vectors, query_angle, query_latent_vectors, nearest_d_dist, 
                                           nearest_d_indices, num_dims)


        merged_images = merge_images(query_image, retrieved_image)
        merged_images = merge_images(merged_images, optimal_image)
        save_img(os.path.join(example_img_dir, str(i)+':'+os.path.basename(q_queries[idx]['query'])), merged_images)
        
        with open(os.path.join(example_img_dir, 'retrieved_img.txt'), 'a') as f:
            f.write(str(i)+': '+os.path.basename(retr_img_path)+'\n')


def localization_accuracy(q_queries, ref_queries, nearest_latent_indices, dist_tol, ang_tol, topn):
    top_correc_count = np.array([0]*topn)
    for query_idx in q_queries:
        query_x, query_y, query_a = get_query_params(q_queries[query_idx])

        nearest_ref_idxs = nearest_latent_indices[query_idx]
        for i,ref_idx in enumerate(nearest_ref_idxs):
            ref_x, ref_y, ref_a = get_query_params(ref_queries[ref_idx])

            geo_dist = np.sqrt((query_x - ref_x)**2 + (query_y - ref_y)**2)
            angle_diff = angle_correction(query_a, ref_a)

            if geo_dist < dist_tol and angle_diff < ang_tol:
                for val in range(i, topn):
                    top_correc_count[val]+=1
                break

    final_accuracy = top_correc_count/len(q_queries)
    return final_accuracy

def get_query_params(query):
    x = query['x']
    y = query['y']
    a = query['a']

    return x,y,a


def update_top5_acc(q_queries, ref_queries, nearest_latent_indices, writer, out_name, mode):
    summary = tf.Summary()
    dist_tol_arr = [1,2]
    ang_tol_arr = [0.523, 1.046]
    for dist_tol in dist_tol_arr:
        for ang_tol in ang_tol_arr:
            final_accuracy = localization_accuracy(q_queries, ref_queries, nearest_latent_indices, dist_tol, ang_tol, 5)
            summary.value.add(tag='{}-{}-Top5_accuracy'.format(dist_tol, ang_tol), simple_value=final_accuracy[-1])
    
    # if mode == 'train':
    step = int(re.findall(r'\d+', out_name)[0])
    writer.add_summary(summary, step)

def get_relevant_queries(queries, pattern):
    new_queries = {}
    for k,v in queries.items():
        if pattern in v['query']:
            new_queries[k] = v
    sorted_queries = {idx: v for idx, (_, v) in enumerate(sorted(new_queries.items(), key=lambda item: item[1]['t']))}
    return sorted_queries

def get_filename_queries(queries):
    
    filename_queries = {v['query']: k for idx, (k, v) in enumerate(sorted(queries.items(), key=lambda item: item[1]['t']))}
    return filename_queries


def smart_sampling(sorted_queries, img_paths, window_size, sample_idx, ANGLE_THRESHOLD):
    
    if window_size == 1:
        sample_idx.append(sorted_queries[img_paths[0]])

    else:
        first_img = img_paths[0]
        last_img = img_paths[-1]
        
        _, _, _, _, first_img_angle = get_t_x_y_a_from_path(first_img)
        _, _, _, _, last_img_angle = get_t_x_y_a_from_path(last_img)
        
        angle_diff = angle_correction(first_img_angle, last_img_angle)
        
        if angle_diff < ANGLE_THRESHOLD:
            sample_idx.append(sorted_queries[last_img])
        else:
            smart_sampling(sorted_queries, img_paths[:window_size//2], window_size//2, sample_idx, ANGLE_THRESHOLD)
            smart_sampling(sorted_queries, img_paths[window_size//2:], window_size//2, sample_idx, ANGLE_THRESHOLD)
            
    return sample_idx    
    
    
def smart_idx_to_keep(sorted_queries, window_size, angle_threshold):
    filenames = list(sorted_queries.keys())
        
    idxs_to_keep = [sorted_queries[filenames[0]]]
    batch_num = len(filenames)//window_size
    for i in range(batch_num):
        sample_idx = []
        sample_idx = smart_sampling(sorted_queries, filenames[i*window_size:(i+1)*window_size], window_size, sample_idx, angle_threshold)
        idxs_to_keep += sample_idx
    
    return idxs_to_keep