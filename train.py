import argparse
import os
import sys
import json
import time


import matplotlib as mpl

mpl.use('Agg')
import tensorflow as tf

from util.loading_touples import load_queries
from util.utils import log_string
from util import utils, model

parser = argparse.ArgumentParser()
# Paths
parser.add_argument('--out_dir', default='all_experiments/')
parser.add_argument('--data_name', default='cold')
parser.add_argument('--checkpoint',
                    default='checkpoints/vd16_pitts30k_conv5_3_vlad_preL2_intra_white/vd16_pitts30k_conv5_3_vlad_preL2_intra_white')  # Convert checkpoints from matlab using functionality from 
parser.add_argument('--query_folder', default='cold_pickles')
parser.add_argument('--max_squared_f_file',
                    default='cold_pickles/train_query_vd16_pitts30k_conv5_3_vlad_preL2_intra_white_latent_vectors/max_dist_50.pickle')
parser.add_argument('--max_angle_f_file',
                    default='cold_pickles/train_query_vd16_pitts30k_conv5_3_vlad_preL2_intra_white_latent_vectors/max_f_angle_50.pickle')
parser.add_argument('--max_actual_angle_file',
                    default='cold_pickles/train_query_vd16_pitts30k_conv5_3_vlad_preL2_intra_white_latent_vectors/max_actual_angle_50.pickle')


# Loss
parser.add_argument('--positives_per_query', type=int, default=6,  # 10
                    help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=6,
                    help='Number of definite negatives in each training tuple [default: 18]')
parser.add_argument('--hard_negatives_per_query', type=int, default=3,
                    help='Number of hard negatives mined per query')
parser.add_argument('--sampled_negatives', type=int, default=1000, help='Samples for hard negative mining.')

parser.add_argument('--window_size', type=int, default=10, help='Window size for smart sampling')
parser.add_argument('--angle_threshold', type=float, default=0.3, help='Angle threshold for smart sampling')
parser.add_argument('--message', type=str, default='', help='special message for save directory')

parser.add_argument('--loss', default='h_sum_angle') # Triplet + huber loss
parser.add_argument('--margin_1', type=float, default=0.1, help='Margin for hinge loss [default: 0.5]')  # NetVLAD
parser.add_argument('--margin_2', type=float, default=0.2, help='Margin for hinge loss [default: 0.2]')
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--lam2', type=float, default=1.0)

parser.add_argument('--max_pos_radius', type=float, default=10.0)
parser.add_argument('--min_neg_radius', type=float, default=25.0)

# Training
parser.add_argument('--n_layers', type=int, default=2, help='Number of FC layers in disentelar network')
parser.add_argument('--queries_per_batch', type=int, default=2,
                    help='Batch Size during training')  # Do not increase, will lead to bug
parser.add_argument('--max_epoch', type=int, default=30, help='Epoch to run [default: 20]')
parser.add_argument('--mining_epoch', type=int, default=1, help='First epoch in which hard negatives are mined')
parser.add_argument('--base_lr', type=float, default=float(5e-5), help='Initial learning rate [default: 0.00005]')
parser.add_argument('--minimal_lr', type=float, default=float(5e-12))
parser.add_argument('--lr_down_factor', type=float, default=0.5)
parser.add_argument('--lr_down_frequency', type=float, default=1)  # Reducing LR every frequency epochs by factor
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for disenteglement network')

parser.add_argument('--easy_threshold', type=float, default=0.25, help='Angle threshold for easy positives')
parser.add_argument('--use_hard_positives', action='store_true', help='Use hard positives or not?')

parser.add_argument('--bn_init_decay', type=float, default=0.5)
parser.add_argument('--bn_decay_rate', type=float, default=0.5)
parser.add_argument('--bn_decay_step', type=float, default=200000)
parser.add_argument('--bn_decay_clip', type=float, default=0.99)

parser.add_argument('--val_steps', type=int, default=100)
parser.add_argument('--max_to_keep', default=1)

parser.add_argument('--use_wpca', type=bool, default=False)
parser.add_argument('--use_disentenglar', action='store_true', help='Use Disenteglar on top of netvlad or not?')
parser.add_argument('--use_vgg_all_layers', action='store_true', help='Use all layers of vgg16 netvlad or not?')
parser.add_argument('--use_residual', action='store_true', help='Use residual connection on disentangler or not?')


FLAGS = parser.parse_args()

#OUT_DIR = FLAGS.out_dir
CHECKPOINT = FLAGS.checkpoint
QUERY_FOLDER = FLAGS.query_folder


MAX_TO_KEEP = FLAGS.max_to_keep

POSITIVES_PER_QUERY = FLAGS.positives_per_query
NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
HARD_NEGATIVES_PER_QUERY = FLAGS.hard_negatives_per_query
SAMPLED_NEGATIVES = FLAGS.sampled_negatives

EXPERIMENT_TIMESTAMP = str(int(time.time()))

OUT_DIR = utils.create_save_dir(FLAGS, EXPERIMENT_TIMESTAMP)

LOSS = FLAGS.loss
USE_DISTS = (LOSS == 'lazy_dist' or LOSS == 'simple_dist' or LOSS == 'weak_simple_dist' or LOSS == 'combination' or
             LOSS == 'lazy_quadruplet_dist' or LOSS == 'huber' or LOSS == 'lazy_tukey_dist' or LOSS == 'simple_tukey'
             or LOSS == 'sum' or LOSS == 'unscaled_sum' or LOSS == 'h_sum' or LOSS == 'm_sum' or LOSS == 'hlq' or LOSS == 'nlq'
             or LOSS == 'h_sum_angle' or LOSS=='lazy_quadruplet_dist'or LOSS=='h_sum_angle_all_pos' or LOSS == 'lazy_quad_sum_angle')
USE_ANGLE = (LOSS == 'angle_loss' or LOSS == 'h_sum_angle' or LOSS == 'lazy_quadruplet_dist'or LOSS=='h_sum_angle_all_pos' or LOSS == 'lazy_quad_sum_angle')


MIN_NEG_RADIUS = FLAGS.min_neg_radius

QUERIES_PER_BATCH = FLAGS.queries_per_batch
MAX_EPOCH = FLAGS.max_epoch
MINING_EPOCH = FLAGS.mining_epoch

VAL_STEPS = FLAGS.val_steps


if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
LOG_PATH = os.path.join(OUT_DIR, 'log_train.txt')
ARG_LOG_PATH = os.path.join(OUT_DIR, 'config.json')

log_string(str(FLAGS)+ '\n', LOG_PATH)
utils.save_config(FLAGS, USE_DISTS, USE_ANGLE, LOG_PATH, ARG_LOG_PATH, OUT_DIR)
print('arg file done')

# Build queries
TRAIN_REF_QUERIES = load_queries(QUERY_FOLDER, 'train_ref.pickle')
TRAIN_QUERY_QUERIES = load_queries(QUERY_FOLDER, 'train_query.pickle')
TEST_REF_QUERIES = load_queries(QUERY_FOLDER, 'test_ref.pickle')
TEST_QUERY_QUERIES = load_queries(QUERY_FOLDER, 'test_query.pickle')


def train():
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        print("In Graph")

        config = json.load(open(ARG_LOG_PATH, 'r'))
        print(config)
        ops = model.build_model(config)
        ops = model.build_loss(config, ops)
        train_op = model.optimizer(ops, config)
        ops['train_op'] = train_op

        saver, epoch_saver, restoration_saver = model.get_savers(FLAGS)

        # Create a session
        configs = utils.get_gpu_config()
        sess = tf.Session(config=configs)

        # Initialize a new model
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Initialized")

        restoration_saver.restore(sess, CHECKPOINT)

        # Add summary writers
        train_writer = tf.summary.FileWriter(os.path.join(OUT_DIR, 'log/train'), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(OUT_DIR, 'log/validation'))


        for epoch in range(MAX_EPOCH):
            print(epoch)
            print()
            log_string('**** EPOCH %03d ****' % epoch, LOG_PATH)
            sys.stdout.flush()

            # epoch_saver.save(sess, os.path.join(OUT_DIR, "epoch-checkpoint"), global_step=epoch, latest_filename='epoch-checkpoint')

            model.train_one_epoch(config, sess, ops, train_writer, val_writer, epoch, saver, TRAIN_REF_QUERIES, TRAIN_QUERY_QUERIES, 
                                  TEST_REF_QUERIES, TEST_QUERY_QUERIES)


if __name__ == "__main__":
    train()
