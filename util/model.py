import os
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.neighbors import KDTree
import matplotlib as mpl

mpl.use('Agg')


import utils
import nets1
from nets1 import vgg16NetvladPca, vgg16Netvlad
from loss import h_sum_angle, triplet_loss, lazy_quadruplet_dist_loss, h_sum_angle_all_pos, lazy_quad_sum_angle, lazy_quadruplet_loss
from utils import log_string
from loading_touples import get_query_tuple, get_xy
from img_helper import load_imgs

def build_model(config):

    QUERIES_PER_BATCH = config['queries_per_batch']
    POSITIVES_PER_QUERY = config['positives_per_query']
    NEGATIVES_PER_QUERY = config['negatives_per_query']
    USE_WPCA = config['use_wpca']
    DROPOUT = config['dropout_rate']
    USE_DISENTENGLAR = config['use_disentenglar']
    USE_ALL_LAYERS = config['use_vgg_all_layers']
    USE_RESIDUAL = config['use_residual']
    N_LAYERS = config['n_layers']
    
    query = tf.placeholder(dtype=tf.float32, shape=[QUERIES_PER_BATCH * 1, None, None, 3])
    positives = tf.placeholder(dtype=tf.float32, shape=[QUERIES_PER_BATCH * POSITIVES_PER_QUERY, None, None, 3])
    negatives = tf.placeholder(dtype=tf.float32, shape=[QUERIES_PER_BATCH * NEGATIVES_PER_QUERY, None, None, 3])
    other_negatives = tf.placeholder(dtype=tf.float32, shape=[QUERIES_PER_BATCH * 1, None, None, 3])

    is_training_pl = tf.placeholder(tf.bool, shape=())
    epoch_num = tf.placeholder(tf.float32, shape=())

    batch = tf.Variable(0)
    vecs = tf.concat([query, positives, negatives, other_negatives], 0)

    if USE_WPCA:
        out_vecs = vgg16NetvladPca(vecs)
    else:
        if USE_ALL_LAYERS:
            out_vecs = nets1.vgg16Netvlad_all_layers(vecs)
        else:
            out_vecs = vgg16Netvlad(vecs)

    if USE_DISENTENGLAR:
        out_vecs = disentangle_network(out_vecs, is_training_pl, DROPOUT, USE_RESIDUAL, N_LAYERS)
    print('NETVLAD out vec shapes ============================ {}'.format(out_vecs))

    q_vec_flat, pos_vecs_flat, neg_vecs_flat, other_neg_vec_flat = tf.split(out_vecs,
                                                                            [1 * QUERIES_PER_BATCH,
                                                                            POSITIVES_PER_QUERY * QUERIES_PER_BATCH,
                                                                            NEGATIVES_PER_QUERY * QUERIES_PER_BATCH,
                                                                            1 * QUERIES_PER_BATCH], 0)

    q_vec = tf.reshape(q_vec_flat, [QUERIES_PER_BATCH, 1, -1])
    pos_vecs = tf.reshape(pos_vecs_flat, [QUERIES_PER_BATCH, POSITIVES_PER_QUERY, -1])
    neg_vecs = tf.reshape(neg_vecs_flat, [QUERIES_PER_BATCH, NEGATIVES_PER_QUERY, -1])
    other_neg_vec = tf.reshape(other_neg_vec_flat, [QUERIES_PER_BATCH, 1, -1])


    ops = {'query': query,
            'positives': positives,
            'negatives': negatives,
            'other_negatives': other_negatives,
            'is_training_pl': is_training_pl,
            'step': batch,
            'epoch_num': epoch_num,
            'q_vec': q_vec,
            'pos_vecs': pos_vecs,
            'neg_vecs': neg_vecs,
            'other_neg_vec': other_neg_vec,
            'batch': batch}

    return ops

def disentangle_network(out_vecs, is_training, dropout_rate, use_residual=False, n_layers=2):
    
    num_dims = out_vecs.get_shape()[1]
    
    with tf.variable_scope('disentengle'):
        new_vec = tf.layers.dense(out_vecs,512, activation=tf.nn.relu)
        for _ in range(n_layers-1):
            new_vec = tf.layers.dropout(new_vec, rate=dropout_rate, training=is_training)
            new_vec = tf.layers.dense(new_vec,512, activation=tf.nn.relu)

        if use_residual:
            new_vec = tf.layers.dropout(new_vec, rate=dropout_rate, training=is_training)
            new_vec = tf.layers.dense(new_vec,num_dims)
            final_output = new_vec + out_vecs
        else:
            final_output = new_vec
        final_output = tf.math.l2_normalize(final_output, axis=1)
        
    return final_output



def build_loss(config, ops):
    
    QUERIES_PER_BATCH = config['queries_per_batch']
    POSITIVES_PER_QUERY = config['positives_per_query']
    USE_DISTS = config['use_dists']
    MAX_SQUARED_F_FILE = config['max_squared_f_file']
    MAX_ANGLE_F_FILE = config['max_angle_f_file']
    MAX_ANGLE_ACTUAL_FILE = config['max_actual_angle_file']
    LOSS = config['loss']
    MAX_POS_RADIUS = config['max_pos_radius']
    USE_ANGLE = config['use_angle']
    MARGIN_1 = config['margin_1']
    MARGIN_2 = config['margin_2']
    LAMBDA = config['lam']
    LAMBDA2 = config['lam2']


    if USE_DISTS:  # The latent vector dists contain squared distance values (we do not do sqrt in losses, as it is not diffable)
        with open(MAX_SQUARED_F_FILE, 'rb') as handle:
            f_max = float(pickle.load(handle)[0])  # Already squared
        d_max = float(MAX_POS_RADIUS) ** 2
    if USE_ANGLE:
        with open(MAX_ANGLE_F_FILE, 'rb') as handle:
            max_f_angle = float(pickle.load(handle)[0]) 
        with open(MAX_ANGLE_ACTUAL_FILE, 'rb') as handle:
            max_actual_angle = float(pickle.load(handle)[0]) 

    if LOSS == 'triplet':
        loss = triplet_loss(ops['q_vec'], ops['pos_vecs'], ops['neg_vecs'], MARGIN_1)  

    if LOSS == 'lazy_quad':
        loss = lazy_quadruplet_loss(ops['q_vec'], ops['pos_vecs'], ops['neg_vecs'], ops['other_neg_vec'], MARGIN_1, MARGIN_2)
        
    if LOSS == 'h_sum_angle':
        pos_angles = tf.placeholder(dtype=tf.float32, shape=[QUERIES_PER_BATCH, POSITIVES_PER_QUERY])
        pos_dists = tf.placeholder(dtype=tf.float32, shape=[QUERIES_PER_BATCH, POSITIVES_PER_QUERY])
        angle_loss, distance_loss, visual_loss, loss = h_sum_angle(ops['q_vec'], ops['pos_vecs'], ops['neg_vecs'], MARGIN_1, LAMBDA, LAMBDA2, pos_dists, d_max, f_max, pos_angles, max_actual_angle, max_f_angle)
    
    if LOSS == 'h_sum_angle_all_pos':
        pos_angles = tf.placeholder(dtype=tf.float32, shape=[QUERIES_PER_BATCH, POSITIVES_PER_QUERY])
        pos_dists = tf.placeholder(dtype=tf.float32, shape=[QUERIES_PER_BATCH, POSITIVES_PER_QUERY])
        angle_loss, distance_loss, visual_loss, loss = h_sum_angle_all_pos(ops['q_vec'], ops['pos_vecs'], ops['neg_vecs'], MARGIN_1, LAMBDA, LAMBDA2, 
                                                                         pos_dists, d_max, f_max, pos_angles, 
                                                                         max_actual_angle, max_f_angle)

    if LOSS == 'lazy_quad_sum_angle':
        pos_angles = tf.placeholder(dtype=tf.float32, shape=[QUERIES_PER_BATCH, POSITIVES_PER_QUERY])
        pos_dists = tf.placeholder(dtype=tf.float32, shape=[QUERIES_PER_BATCH, POSITIVES_PER_QUERY])
        angle_loss, distance_loss, visual_loss, loss = lazy_quad_sum_angle(ops['q_vec'], ops['pos_vecs'], ops['neg_vecs'], ops['other_neg_vec'], MARGIN_1, MARGIN_2, LAMBDA, LAMBDA2, pos_dists, d_max, f_max, pos_angles, max_actual_angle, max_f_angle)
         
    if LOSS == 'h_sum_angle' or LOSS == 'h_sum_angle_all_pos' or LOSS == 'lazy_quad_sum_angle':
        tf.summary.scalar('angle_loss', angle_loss)
        tf.summary.scalar('distance_loss', distance_loss)
        tf.summary.scalar('Visual_loss', visual_loss)
    tf.summary.scalar('total_loss', loss)

    # Get training operator
    learning_rate = utils.get_learning_rate(ops['epoch_num'], config)
    tf.summary.scalar('learning_rate', learning_rate)
    merged = tf.summary.merge_all()
    
    ops['loss'] = loss
    ops['merged'] = merged
    ops['learning_rate'] = learning_rate

    if USE_DISTS:
        ops['pos_dists'] = pos_dists
    if USE_ANGLE:
        ops['pos_angles'] = pos_angles
    
    return ops


def optimizer(ops, config):

    OPTIMIZER = config['optimizer']
    MOMENTUM = config['momentum']
    learning_rate = ops['learning_rate']
    loss = ops['loss']
    batch = ops['batch']
    
    if OPTIMIZER == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
    elif OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=batch)

    return train_op

def get_savers(args):
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=args.max_to_keep)
    epoch_saver = tf.train.Saver(max_to_keep=0)

    to_restore = {}
    for var in tf.trainable_variables():
        print(var.name)
        if var.name != 'Variable:0' and 'disentengle' not in var.name and 'skip_vgg' not in var.name:
            saved_name = var._shared_name
            to_restore[saved_name] = var
            print(var)

    restoration_saver = tf.train.Saver(to_restore)

    return saver, epoch_saver, restoration_saver

def train_one_epoch(config, sess, ops, train_writer, val_writer, epoch, saver, train_ref_queries, train_query_queries, test_ref_queries, test_query_queries):

    NEGATIVES_PER_QUERY = config['negatives_per_query']
    QUERIES_PER_BATCH = config['queries_per_batch']
    POSITIVES_PER_QUERY = config['positives_per_query']
    VAL_STEPS = config['val_steps']
    LOG_PATH = config['log_path']
    USE_DISTS = config['use_dists']
    USE_ANGLE = config['use_angle']
    OUT_DIR = config['save_dir']
    TRAIN_REF_QUERIES = train_ref_queries
    ANGLE_THRESHOLD = config['angle_threshold']
    WINDOW_SIZE = config['window_size']
    EASY_THRESHOLD = config['easy_threshold']
    USE_HARD_POSITIVES = config['use_hard_positives']
    DATA_NAME = config['data_name']

    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_REF_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)

    batches_in_epoch = len(train_file_idxs) // QUERIES_PER_BATCH
    for i in tqdm(range(batches_in_epoch)):

        log_string('**** BATCH %03d ****' % (i), LOG_PATH)

        if i > 0 and (i % VAL_STEPS == 0 or i == 1):
            save_path = saver.save(sess, os.path.join(OUT_DIR, "checkpoint"), global_step=batches_in_epoch * epoch + i)
            log_string("Model saved in file: %s" % save_path, LOG_PATH)
            evaluate_localization(TRAIN_REF_QUERIES, train_query_queries, os.path.basename(save_path), ops, sess, train_writer, mode='train', out_dir = OUT_DIR, config=config, angle_threshold=ANGLE_THRESHOLD, window_size=WINDOW_SIZE, dataname=DATA_NAME)
            evaluate_localization(test_ref_queries, test_query_queries, os.path.basename(save_path), ops, sess, val_writer, mode='test', out_dir = OUT_DIR, config=config, angle_threshold=ANGLE_THRESHOLD, window_size=WINDOW_SIZE, dataname=DATA_NAME)

        batch_keys = train_file_idxs[i * QUERIES_PER_BATCH:(i + 1) * QUERIES_PER_BATCH]
        q_tuples = []

        no_other_neg = False
        for j in range(QUERIES_PER_BATCH):

            log_string('**** QUERY %03d ****' % (j), LOG_PATH)
            faulty_tuple_pos, faulty_tuple_neg = utils.check_faulty_tuple(TRAIN_REF_QUERIES[batch_keys[j]], config)
            if faulty_tuple_pos or faulty_tuple_neg:
                break

            q_tuples.append(get_query_tuple(TRAIN_REF_QUERIES[batch_keys[j]], POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY,
                            TRAIN_REF_QUERIES, easy_threshold=EASY_THRESHOLD, hard_neg=[], other_neg=True, 
                            add_pos_dists=USE_DISTS, add_pos_angle=USE_ANGLE, use_hard_positives=USE_HARD_POSITIVES))

            if (q_tuples[j][3].shape[2] != 3):
                no_other_neg = True
                break

        # construct query array
        utils.print_faulty_tuple_string(faulty_tuple_pos, faulty_tuple_neg, no_other_neg, LOG_PATH, i)
        if faulty_tuple_pos or faulty_tuple_neg or no_other_neg:
            continue

        input_data = utils.get_input_data(config, q_tuples, USE_DISTS, USE_ANGLE, i, LOG_PATH)
        if len(input_data['queries'].shape) != 4:
            log_string('----' + 'FAULTY QUERY' + '-----', LOG_PATH)
            continue

        feed_dict = {ops['query']: input_data['queries'], ops['positives']: input_data['positives'], ops['negatives']: input_data['negatives'], ops['other_negatives']: input_data['other_neg'], ops['is_training_pl']: is_training, ops['epoch_num']: epoch}

        if USE_DISTS:
            feed_dict[ops['pos_dists']] = input_data['dists']
        if USE_ANGLE:
            feed_dict[ops['pos_angles']] = input_data['angles']
        summary, step, _, loss_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss']],
                                                  feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        log_string('batch loss: %f' % loss_val, LOG_PATH)


def evaluate_localization(ref_queries, q_queries, out_name, ops, sess, writer, mode, out_dir, config, angle_threshold, window_size, dataname):
    if mode == 'train':
        if dataname=='cold':
            ref_pattern = 'freiburg/seq1_cloudy1'
            q_pattern = 'freiburg/seq1_night3'
        else:
            ref_pattern = '2014-11-18-13-20-12'
            q_pattern = '2015-02-13-09-16-26'
    else:
        if dataname == 'cold':
            ref_pattern = 'seq3_cloudy1'
            q_pattern = 'seq3_sunny3'
        else:
            ref_pattern = '2014-11-18-13-20-12'
            q_pattern = '2015-02-13-09-16-26'
    ref_queries = utils.get_relevant_queries(ref_queries, ref_pattern)
    q_queries = utils.get_relevant_queries(q_queries, q_pattern)
    
    ref_filename_queries = utils.get_filename_queries(ref_queries)
    
    ref_idxs_to_keep = utils.smart_idx_to_keep(ref_filename_queries, window_size, angle_threshold)

    q_set_size = 500
    q_idxs_to_keep = utils.get_idx_to_keep(q_queries, q_set_size)

    ref_queries = utils.downsample_queries(ref_queries, ref_idxs_to_keep)
    q_queries = utils.downsample_queries(q_queries, q_idxs_to_keep)

    ref_latent_vectors = get_latent_vectors(sess, ops, ref_queries, config)
    query_latent_vectors = get_latent_vectors(sess, ops, q_queries, config)
    
    ref_XY = get_xy(ref_queries)
    query_XY = get_xy(q_queries)

    ref_latent_tree = KDTree(ref_latent_vectors)
    ref_d_tree = KDTree(ref_XY)

    ## used to get 5 nearest neighbours from query latent vectors
    nearest_latent_dists, nearest_latent_indices = ref_latent_tree.query(query_latent_vectors, k=5)
    nearest_d_dist, nearest_d_indices = ref_d_tree.query(query_XY, k=1)

    d_to_nearest_latent = utils.get_d_to_nearest_latent_matrix(q_queries, nearest_latent_indices, query_XY, ref_XY)
    utils.draw_plots(d_to_nearest_latent, nearest_d_dist, out_name, writer, out_dir, mode=mode)

    # Update top5 accuracy based on distance and angle
    utils.update_top5_acc(q_queries, ref_queries, nearest_latent_indices, writer, out_name, mode=mode)

    # Save visual examples
    utils.save_visual_examples(q_queries, query_latent_vectors, ref_queries, ref_latent_vectors, 
                               nearest_latent_dists, nearest_latent_indices, d_to_nearest_latent, 
                               nearest_d_dist, nearest_d_indices, out_dir, mode, out_name) ## adding num_dims



def get_latent_vectors(sess, ops, dict_to_process, config):
    QUERIES_PER_BATCH = config['queries_per_batch']
    POSITIVES_PER_QUERY = config['positives_per_query']
    NEGATIVES_PER_QUERY = config['negatives_per_query']

    print('--Getting {} latent vectors--'.format(len(dict_to_process.keys())))
    is_training = False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = QUERIES_PER_BATCH * (1 + POSITIVES_PER_QUERY + NEGATIVES_PER_QUERY + 1)
    q_output = []
    for q_index in range(len(train_file_idxs) // batch_num):
        file_indices = train_file_idxs[q_index * batch_num:(q_index + 1) * (batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_imgs(file_names)

        q1 = queries[0:QUERIES_PER_BATCH]

        q2 = queries[QUERIES_PER_BATCH:QUERIES_PER_BATCH * (POSITIVES_PER_QUERY + 1)]

        q3 = queries[QUERIES_PER_BATCH * (POSITIVES_PER_QUERY + 1):QUERIES_PER_BATCH * (
                NEGATIVES_PER_QUERY + POSITIVES_PER_QUERY + 1)]

        q4 = queries[QUERIES_PER_BATCH * (NEGATIVES_PER_QUERY + POSITIVES_PER_QUERY + 1):QUERIES_PER_BATCH * (
                NEGATIVES_PER_QUERY + POSITIVES_PER_QUERY + 2)]

        q2 = np.reshape(q2, (QUERIES_PER_BATCH * POSITIVES_PER_QUERY, q2.shape[1], q2.shape[2], 3))
        q3 = np.reshape(q3, (QUERIES_PER_BATCH * NEGATIVES_PER_QUERY, q3.shape[1], q3.shape[2], 3))

        feed_dict = {ops['query']: q1, ops['positives']: q2, ops['negatives']: q3, ops['other_negatives']: q4,
                     ops['is_training_pl']: is_training}
        o1, o2, o3, o4 = sess.run([ops['q_vec'], ops['pos_vecs'], ops['neg_vecs'], ops['other_neg_vec']],
                                  feed_dict=feed_dict)

        o1 = np.reshape(o1, (-1, o1.shape[-1]))
        o2 = np.reshape(o2, (-1, o2.shape[-1]))
        o3 = np.reshape(o3, (-1, o3.shape[-1]))
        o4 = np.reshape(o4, (-1, o4.shape[-1]))

        out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if len(q_output) != 0:
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        queries = load_imgs([dict_to_process[index]["query"]])

        if QUERIES_PER_BATCH - 1 > 0:
            fake_queries = np.zeros((QUERIES_PER_BATCH - 1, queries.shape[1], queries.shape[2], 3))
            q = np.vstack((queries, fake_queries))
        else:
            q = queries

        fake_pos = np.zeros((QUERIES_PER_BATCH * POSITIVES_PER_QUERY, queries.shape[1], queries.shape[2], 3))
        fake_neg = np.zeros((QUERIES_PER_BATCH * NEGATIVES_PER_QUERY, queries.shape[1], queries.shape[2], 3))
        fake_other_neg = np.zeros((QUERIES_PER_BATCH, queries.shape[1], queries.shape[2], 3))
        feed_dict = {ops['query']: q, ops['positives']: fake_pos, ops['negatives']: fake_neg,
                     ops['other_negatives']: fake_other_neg, ops['is_training_pl']: is_training}
        output = sess.run(ops['q_vec'], feed_dict=feed_dict)
        output = output[0]
        output = np.squeeze(output)
        if q_output.shape[0] != 0:
            q_output = np.vstack((q_output, output))
        else:
            q_output = output
    return q_output
