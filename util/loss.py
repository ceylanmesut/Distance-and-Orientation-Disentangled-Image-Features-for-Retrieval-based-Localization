import tensorflow as tf


def angle_loss(q_vec, q2_vec, pos_angles, max_angle, max_f_angle):
    num_dims = q2_vec.get_shape()[2]
    num_pos = q2_vec.get_shape()[1]
    query_copies = tf.tile(q_vec, [1, int(num_pos), 1])  # shape num_pos x output_dim

    squared_f_angle_dists = tf.reduce_sum(tf.squared_difference(q2_vec[:,:,num_dims//2:], query_copies[:,:,num_dims//2:]), 2)

    scaled_angle = tf.div(pos_angles, max_angle)
    scaled_f_angle = tf.div(squared_f_angle_dists, max_f_angle)

    ang_loss = tf.losses.huber_loss(scaled_angle, scaled_f_angle)
    return ang_loss

def h_sum_angle(q_vec, pos_vecs, neg_vecs, margin, lam, lam2, squared_d_dists, d_max_squared, f_max_squared, pos_angles, max_angle, max_f_angle):
    angle_loss_name = angle_loss(q_vec, pos_vecs, pos_angles, max_angle, max_f_angle)
    distance_loss = huber_dist_loss(q_vec, pos_vecs, squared_d_dists, d_max_squared, f_max_squared)
    triplet_loss_name = triplet_loss(q_vec, pos_vecs, neg_vecs, margin)
    total_loss = tf.add(tf.add( triplet_loss_name,tf.multiply(lam, distance_loss)), tf.multiply(lam2, angle_loss_name))
    return angle_loss_name, distance_loss, triplet_loss_name, total_loss

def lazy_quad_sum_angle(q_vec, pos_vecs, neg_vecs, other_neg, margin1, margin2, lam, lam2, squared_d_dists, d_max_squared, f_max_squared, pos_angles, max_angle, max_f_angle):
    angle_loss_name = angle_loss(q_vec, pos_vecs, pos_angles, max_angle, max_f_angle)
    distance_loss = huber_dist_loss(q_vec, pos_vecs, squared_d_dists, d_max_squared, f_max_squared)
    lazy_quad_loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, margin1, margin2)
    total_loss = tf.add(tf.add( lazy_quad_loss,tf.multiply(lam, distance_loss)), tf.multiply(lam2, angle_loss_name))
    return angle_loss_name, distance_loss, lazy_quad_loss, total_loss

def h_sum_angle_all_pos(q_vec, pos_vecs, neg_vecs, margin, lam, lam2, squared_d_dists, d_max_squared, f_max_squared, pos_angles, max_angle, max_f_angle):
    angle_loss_name = angle_loss(q_vec, pos_vecs, pos_angles, max_angle, max_f_angle)
    distance_loss = huber_dist_loss(q_vec, pos_vecs, squared_d_dists, d_max_squared, f_max_squared)
    triplet_loss_name = triplet_loss_all_pos(q_vec, pos_vecs, neg_vecs, margin)
    total_loss = tf.add(tf.add( triplet_loss_name,tf.multiply(lam, distance_loss)), tf.multiply(lam2, angle_loss_name))
    return angle_loss_name, distance_loss, triplet_loss_name, total_loss


def huber_dist_loss(q_vec, q2_vec, squared_d_dists, d_max_squared, f_max_squared):
    num_dims = q2_vec.get_shape()[2]
    num_pos = q2_vec.get_shape()[1]
    batch_size = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_pos), 1])  # shape num_pos x output_dim
    squared_f_dists = tf.reduce_sum(tf.squared_difference(q2_vec[:,:,:num_dims//2], query_copies[:,:,:num_dims//2]), 2)

    d_max_copies = tf.fill([int(batch_size), int(num_pos)], d_max_squared)
    f_max_copies = tf.fill([int(batch_size), int(num_pos)], f_max_squared)

    scaled_d_dists = tf.div(squared_d_dists, d_max_copies)
    scaled_f_dists = tf.div(squared_f_dists, f_max_copies)

    return tf.losses.huber_loss(scaled_d_dists, scaled_f_dists)

##########Losses for PointNetVLAD###########

# Returns average loss across the query tuples in a batch, loss in each is the average loss of the definite negatives against the best positive
def best_pos_distance(query, pos_vecs):
    with tf.name_scope('best_pos_distance') as scope:
        # batch = query.get_shape()[0]
        num_pos = pos_vecs.get_shape()[1]
        query_copies = tf.tile(query, [1, int(num_pos), 1])  # shape num_pos x output_dim
        best_pos = tf.reduce_min(tf.reduce_sum(tf.squared_difference(pos_vecs, query_copies), 2), 1)
        # best_pos=tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
        return best_pos


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
    # ''', end_points, reg_weight=0.001):
    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m = tf.fill([int(batch), int(num_neg)], margin)
    triplet_loss = tf.reduce_mean(tf.reduce_sum(
        tf.maximum(tf.add(m, tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies), 2))),
                   tf.zeros([int(batch), int(num_neg)])), 1))
    return triplet_loss

def compute_feature_distance(query, vecs):
    num = vecs.get_shape()[1]
    query_copies = tf.tile(query, [1, int(num), 1])  # shape queries_per_batch x num x output_dim

    dists = tf.reduce_sum(tf.squared_difference(vecs, query_copies), 2)
    return dists, num

def triplet_loss_all_pos(q_vec, pos_vecs, neg_vecs, margin):
    print('New triplet Loss ===========================================')
    batch = q_vec.get_shape()[0]

    pos_dist, num_pos = compute_feature_distance(q_vec, pos_vecs)
    neg_dist, num_neg = compute_feature_distance(q_vec, neg_vecs)
    pos_dist = tf.tile(pos_dist, [1, int(num_neg)])

    neg_dist = tf.repeat(neg_dist, repeats=num_pos, axis=1)
    m = tf.fill([int(batch), int(num_neg)*int(num_pos)], margin)
    triplet_loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.add(m, tf.subtract(pos_dist, neg_dist)), 
                        tf.zeros([int(batch), int(num_neg)*int(num_pos)])), 1))

    return triplet_loss

def lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m = tf.fill([int(batch), int(num_neg)], margin)
    triplet_loss = tf.reduce_mean(tf.reduce_max(
        tf.maximum(tf.add(m, tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies), 2))),
                   tf.zeros([int(batch), int(num_neg)])), 1))
    return triplet_loss

def lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2):
    trip_loss = lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, m1)

    best_pos = best_pos_distance(q_vec, pos_vecs)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)

    second_loss = tf.reduce_mean(tf.reduce_max(tf.maximum(
        tf.add(m2, tf.subtract(best_pos, tf.reduce_sum(tf.squared_difference(neg_vecs, other_neg_copies), 2))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = trip_loss + second_loss

    return total_loss


def best_pos_d_distance(q_vec, q2_vec, squared_d_dists, d_max_squared, f_max_squared, squared_pos_angles, max_angle, max_f_angle):
    num_dims = q2_vec.get_shape()[2]
    num_pos = q2_vec.get_shape()[1]
    batch_size = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_pos), 1])  # shape num_pos x output_dim
    squared_f_dists = tf.reduce_sum(tf.squared_difference(q2_vec[:,:,:num_dims//2], query_copies[:,:,:num_dims//2]), 2)
    squared_f_angle_dists = tf.reduce_sum(tf.squared_difference(q2_vec[:,:,num_dims//2:], query_copies[:,:,num_dims//2:]), 2)


    d_max_copies = tf.fill([int(batch_size), int(num_pos)], d_max_squared)
    f_max_copies = tf.fill([int(batch_size), int(num_pos)], f_max_squared)
    
    max_angle_copies = tf.fill([int(batch_size), int(num_pos)], max_angle)
    max_f_angle_copies = tf.fill([int(batch_size), int(num_pos)], max_f_angle)

    scaled_d_dists = tf.div(squared_d_dists, d_max_copies)
    scaled_f_dists = tf.div(squared_f_dists, f_max_copies)
    
    scaled_angle_dists = tf.div(squared_pos_angles, max_angle_copies)
    scaled_f_angle_dists = tf.div(squared_f_angle_dists, max_f_angle_copies)
    
    best_d_pos = tf.reduce_min(tf.squared_difference(scaled_f_dists, scaled_d_dists),1)
    best_a_pos = tf.reduce_min(tf.squared_difference(scaled_f_angle_dists, scaled_angle_dists),1)
    best_pos = best_d_pos + best_a_pos

    return best_d_pos, best_a_pos, best_pos


def lazy_triplet_dist_loss(q_vec, pos_vecs, neg_vecs, squared_d_dists, margin, d_max_squared, f_max_squared, squared_pos_angles, max_angle, max_f_angle):
    margin = 0.0  # Ignoring margin
    _,_,best_pos = best_pos_d_distance(q_vec, pos_vecs, squared_d_dists, d_max_squared, f_max_squared, squared_pos_angles, max_angle, max_f_angle)
    num_neg = neg_vecs.get_shape()[1]
    batch_size = q_vec.get_shape()[0]
    query_copies = tf.tile(q_vec, [1, int(num_neg), 1])
    best_pos = tf.tile(tf.reshape(best_pos, (-1, 1)), [1, int(num_neg)])
    m = tf.fill([int(batch_size), int(num_neg)], margin)

    f_max_copies = tf.fill([int(batch_size), int(num_neg)], f_max_squared)

    lazy_dist = tf.reduce_mean(tf.reduce_max(tf.maximum(tf.add(m, tf.subtract(best_pos, tf.div(
        tf.reduce_sum(tf.squared_difference(neg_vecs, query_copies), 2), f_max_copies))),
                                                        tf.zeros([int(batch_size), int(num_neg)])), 1))
    return lazy_dist

def lazy_quadruplet_dist_loss(q_vec, pos_vecs, neg_vecs, other_neg, squared_d_dists, d_max_squared,
                              f_max_squared, squared_pos_angles, max_angle, max_f_angle):
    # Ignoring margins
    m1 = 0.0
    m2 = 0.0
    num_dims = pos_vecs.get_shape()[2]
    trip_loss = lazy_triplet_dist_loss(q_vec, pos_vecs, neg_vecs, squared_d_dists, m1, d_max_squared, f_max_squared, squared_pos_angles, max_angle, max_f_angle)

    best_d_pos, best_a_pos, best_pos = best_pos_d_distance(q_vec, pos_vecs, squared_d_dists, d_max_squared, f_max_squared, squared_pos_angles, max_angle, max_f_angle)
    num_neg = neg_vecs.get_shape()[1]
    batch = q_vec.get_shape()[0]

    other_neg_copies = tf.tile(other_neg, [1, int(num_neg), 1])
    best_d_pos = tf.tile(tf.reshape(best_d_pos, (-1, 1)), [1, int(num_neg)])
    best_a_pos = tf.tile(tf.reshape(best_a_pos, (-1, 1)), [1, int(num_neg)])
    m2 = tf.fill([int(batch), int(num_neg)], m2)
    f_max_copies = tf.fill([int(batch), int(num_neg)], f_max_squared)
    angle_f_max_copies = tf.fill([int(batch), int(num_neg)], max_f_angle)

    second_loss = tf.reduce_mean(tf.reduce_max(tf.maximum(
        tf.add(m2, tf.subtract(best_d_pos, tf.div(tf.reduce_sum(tf.squared_difference(neg_vecs[:,:,:num_dims//2], other_neg_copies[:,:,:num_dims//2]), 2),
                                                f_max_copies))),
        tf.zeros([int(batch), int(num_neg)])), 1))
        
    angle_loss = tf.reduce_mean(tf.reduce_max(tf.maximum(
        tf.add(m2, tf.subtract(best_a_pos, tf.div(tf.reduce_sum(tf.squared_difference(neg_vecs[:,:,num_dims//2:], other_neg_copies[:,:,num_dims//2:]), 2),
                                                angle_f_max_copies))),
        tf.zeros([int(batch), int(num_neg)])), 1))

    total_loss = trip_loss + second_loss + angle_loss

    return angle_loss, second_loss, trip_loss, total_loss
