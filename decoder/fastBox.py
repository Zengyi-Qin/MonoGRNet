#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create the fastbox decoder. For a detailed description see:
https://arxiv.org/abs/1612.07695 ."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random
import pickle
from utils import train_utils
from utils import data_utils


import tensorflow as tf

def _roi_align(hyp, pred_boxes, early_feat, out_channels, crop_size, branch):
    with tf.name_scope('roi_align'):
        grid_size = hyp['grid_width']*hyp['grid_height']
        outer_size = grid_size*hyp['batch_size']
        fine_stride = 8.
        coarse_stride = hyp['region_size']

        batch_ids = []
        x_offsets = []
        y_offsets = []
        for n in range(hyp['batch_size']):
            for i in range(hyp['grid_height']):
                for j in range(hyp['grid_width']):
                    for k in range(hyp['rnn_len']):
                        batch_ids.append(n)
                        x_offsets.append([coarse_stride / 2. + coarse_stride * j])
                        y_offsets.append([coarse_stride / 2. + coarse_stride * i])
        batch_ids = tf.constant(batch_ids)
        x_offsets = tf.reshape(tf.constant(x_offsets), (-1, 1))
        y_offsets = tf.reshape(tf.constant(y_offsets), (-1, 1))

        pred_boxes_r = tf.reshape(pred_boxes, [outer_size * hyp['rnn_len'], 4])
        pred_dx, pred_dy, pred_w, pred_h = tf.split(pred_boxes[:, 0, :], 4, axis=1)
        pred_x = pred_dx + x_offsets
        pred_y = pred_dy + y_offsets

        pred_left = (pred_x - pred_w * 0.5) / hyp['image_width']
        pred_right = (pred_x + pred_w * 0.5) / hyp['image_width']
        pred_top = (pred_y - pred_h * 0.5) / hyp['image_height']
        pred_bottom = (pred_y + pred_h * 0.5) / hyp['image_height']

        pred_boxes_coor = tf.concat([pred_top, pred_left, pred_bottom, pred_right], axis=1)
        roi_align = tf.image.crop_and_resize(early_feat, pred_boxes_coor, batch_ids, [crop_size, crop_size])

        in_channels = early_feat.get_shape().as_list()[-1]
        filt = tf.get_variable('roi_align_conv', shape=(1, 1, in_channels, out_channels))
        conv = tf.nn.conv2d(roi_align, filt, [1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv)
        
        tf.add_to_collection('trainable', filt)
        tf.add_to_collection(branch, filt)
        filt_decay = tf.nn.l2_loss(filt) * 1e-4
        tf.add_to_collection('new_weights', filt_decay)
        tf.add_to_collection(branch, filt_decay)

        assert hyp['rnn_len'] == 1
        return tf.reshape(conv, [outer_size, crop_size * crop_size * out_channels])

def _roi_align_keep_ratio(hyp, pred_boxes, early_feat, out_channels, crop_size, name, branch):
    with tf.name_scope('roi_align'):
        grid_size = hyp['grid_width']*hyp['grid_height']
        outer_size = grid_size*hyp['batch_size']
        fine_stride = 8.
        coarse_stride = hyp['region_size']

        batch_ids = []
        x_offsets = []
        y_offsets = []
        for n in range(hyp['batch_size']):
            for i in range(hyp['grid_height']):
                for j in range(hyp['grid_width']):
                    for k in range(hyp['rnn_len']):
                        batch_ids.append(n)
                        x_offsets.append([coarse_stride / 2. + coarse_stride * j])
                        y_offsets.append([coarse_stride / 2. + coarse_stride * i])
        batch_ids = tf.constant(batch_ids)
        x_offsets = tf.reshape(tf.constant(x_offsets), (-1, 1))
        y_offsets = tf.reshape(tf.constant(y_offsets), (-1, 1))

        pred_boxes_r = tf.reshape(pred_boxes, [outer_size * hyp['rnn_len'], 4])
        pred_dx, pred_dy, pred_w, pred_h = tf.split(pred_boxes_r, 4, axis=1)
        pred_x = pred_dx + x_offsets
        pred_y = pred_dy + y_offsets

        pred_left = (pred_x - pred_w * 0.5) / hyp['image_width']
        pred_right = (pred_x + pred_w * 0.5) / hyp['image_width']
        pred_top = (pred_y - pred_h * 0.5) / hyp['image_height']
        pred_bottom = (pred_y + pred_h * 0.5) / hyp['image_height']

        pred_boxes_coor = tf.concat([pred_top, pred_left, pred_bottom, pred_right], axis=1)
        roi_align = tf.image.crop_and_resize(early_feat, pred_boxes_coor, batch_ids, [crop_size, crop_size])

        in_channels = early_feat.get_shape().as_list()[-1]
        filt = tf.get_variable('roi_align_conv_{}'.format(name), shape=(1, 1, in_channels, out_channels))
        conv = tf.nn.conv2d(roi_align, filt, [1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv)

        tf.add_to_collection('trainable', filt)
        tf.add_to_collection(branch, filt)
        filt_decay = tf.nn.l2_loss(filt) * 1e-4
        tf.add_to_collection('new_weights', filt_decay)
        tf.add_to_collection(branch+'_decay', filt_decay)

        assert hyp['rnn_len'] == 1
        return relu


def _build_corner_regression_layer(hype, pred_boxes, early_feat):
    out_channels = 64
    crop_size = 16
    outer_size = hype['grid_height'] * hype['grid_width'] * hype['batch_size']
    features = _roi_align(hype, pred_boxes, early_feat, out_channels, crop_size, 'corners')
    corner_weights = tf.get_variable('corner_regression', \
                                      shape=[crop_size*crop_size*out_channels, 24])
    tf.add_to_collection('trainable', corner_weights)
    tf.add_to_collection('corners', corner_weights)
    corner_weights_decay = tf.nn.l2_loss(corner_weights) * 1e-4
    tf.add_to_collection('new_weights', corner_weights_decay)
    tf.add_to_collection('corners_decay', corner_weights_decay)

    pred_corners = 3.0 * tf.tanh(tf.matmul(features, corner_weights))
   
    return tf.reshape(pred_corners, (outer_size, 24))

def compute_corners(hypes, dimensions, alpha):
    outer_size = hypes['grid_height'] * hypes['grid_width'] * hypes['batch_size']
    h, w, l = tf.split(tf.reshape(dimensions, (outer_size, 1, 3)), [1, 1, 1], axis=2)
    unrot = tf.concat([tf.concat([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], axis=2),
                       tf.concat([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], axis=2)], axis=1) 
    alpha_r = tf.reshape(alpha, (outer_size, 1))
    zeros = tf.zeros((outer_size, 1), dtype=tf.float32)

    x_rot_vect = tf.reshape(tf.concat([tf.cos(alpha_r), tf.sin(alpha_r)], axis=1), (outer_size, 2, 1))
    x_rot = tf.reduce_sum(unrot * x_rot_vect, axis=1, keep_dims=True)

    z_rot_vect = tf.reshape(tf.concat([-tf.sin(alpha_r), tf.cos(alpha_r)], axis=1), (outer_size, 2, 1))
    z_rot = tf.reduce_sum(unrot * z_rot_vect, axis=1, keep_dims=True)

    zeros = tf.zeros((outer_size, 1, 1), dtype=tf.float32)
    y_rot = tf.concat([zeros, zeros, zeros, zeros, -h, -h, -h, -h], axis=2)

    corners_rot = tf.reshape(tf.concat([x_rot, y_rot, z_rot], axis=1), (outer_size, 24))
    return corners_rot  

def _rezoom(hyp, pred_boxes, early_feat, early_feat_channels,
            w_offsets, h_offsets):
    '''
    Rezoom into a feature map at multiple interpolation points
    in a grid.

    If the predicted object center is at X, len(w_offsets) == 3,
    and len(h_offsets) == 5,
    the rezoom grid will look as follows:

    [o o o]
    [o o o]
    [o X o]
    [o o o]
    [o o o]

    Where each letter indexes into the feature map with bilinear interpolation
    '''
    with tf.name_scope('rezoom'):
        grid_size = hyp['grid_width'] * hyp['grid_height']
        outer_size = grid_size * hyp['batch_size']
        indices = []
        for w_offset in w_offsets:
            for h_offset in h_offsets:
                indices.append(train_utils.bilinear_select(hyp,
                                                           pred_boxes,
                                                           early_feat,
                                                           early_feat_channels,
                                                           w_offset, h_offset))
        interp_indices = tf.concat(axis=0, values=indices)
        rezoom_features = train_utils.interp(early_feat,
                                             interp_indices,
                                             early_feat_channels)
        rezoom_features_r = tf.reshape(rezoom_features,
                                       [len(w_offsets) * len(h_offsets),
                                        outer_size,
                                        hyp['rnn_len'],
                                        early_feat_channels])
        rezoom_features_t = tf.transpose(rezoom_features_r, [1, 2, 0, 3])
        return tf.reshape(rezoom_features_t,
                          [outer_size,
                           hyp['rnn_len'],
                           len(w_offsets) * len(h_offsets) * early_feat_channels])


def _build_inner_layer(hyp, encoded_features, train):
    '''
    Apply an 1x1 convolutions to compute inner features
    The layer consists of 1x1 convolutions implemented as
    matrix multiplication. This makes the layer very fast.
    The layer has "hyp['num_inner_channel']" channels
    '''
    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']

    num_ex = hyp['batch_size'] * hyp['grid_width'] * hyp['grid_height']

    channels = int(encoded_features.shape[-1])
    hyp['cnn_channels'] = channels
    hidden_input = tf.reshape(encoded_features, [num_ex, channels])

    scale_down = hyp['scale_down']

    hidden_input = tf.reshape(
        hidden_input * scale_down, (hyp['batch_size'] * grid_size, channels))

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    model_2D_path = os.path.join(hyp['dirs']['data_dir'], 'model_2D.pkl')
    with open(model_2D_path, 'rb') as file:
        data_dict = pickle.load(file)#, encoding='latin1')    
        file.close()

    with tf.variable_scope('Overfeat'):
        trained_ip = tf.constant_initializer(value=data_dict['ip'])
        w = tf.get_variable(name='ip', initializer=trained_ip, shape=data_dict['ip'].shape)
        output = tf.matmul(hidden_input, w)

    if train:
        # Adding dropout during training
        output = tf.nn.dropout(output, 0.5)
    return output, data_dict


def _build_output_layer(hyp, hidden_output, data_dict, logits, calib_pinv, labels, train):
    '''
    Build an 1x1 conv layer.
    The layer consists of 1x1 convolutions implemented as
    matrix multiplication. This makes the layer very fast.
    The layer has "hyp['num_inner_channel']" channels
    '''

    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']

    width_range = np.arange(hyp['grid_width'])
    height_range = np.arange(hyp['grid_height'])
    width_range, height_range = np.meshgrid(width_range, height_range)
    x_offset = (width_range.flatten() + 0.5)*hyp['region_size']
    y_offset = (height_range.flatten() + 0.5)*hyp['region_size']
    
    xy_offset = np.concatenate([x_offset.reshape(1, grid_size, 1), 
                                y_offset.reshape(1, grid_size, 1)], 
                                axis=2).astype(np.float32)
    xy_offset = np.concatenate([xy_offset for _ in range(hyp['batch_size'])], axis=0)
    xy_offset = np.reshape(xy_offset, (outer_size, 2))

    trained_box_weights = tf.constant_initializer(value=data_dict['box_out'])

    box_weights = tf.get_variable(name='box_out', initializer=trained_box_weights, 
                                  shape=data_dict['box_out'].shape)
    box_weights_decay = tf.nn.l2_loss(box_weights) * 1e-4
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, box_weights_decay)
    

    trained_conf_weights = tf.constant_initializer(value=data_dict['confs_out'])
    conf_weights = tf.get_variable(name='confs_out', initializer=trained_conf_weights,
                                   shape=data_dict['confs_out'].shape)
    conf_weights_decay = tf.nn.l2_loss(conf_weights) * 1e-4
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, conf_weights_decay)

    pred_boxes = tf.reshape(tf.matmul(hidden_output, box_weights) * 50,
                            [outer_size, 1, 4])

    # hyp['rnn_len']
    pred_logits = tf.reshape(tf.matmul(hidden_output, conf_weights),
                             [outer_size, 1, hyp['num_classes']])

    pred_logits_squash = tf.reshape(pred_logits,
                                    [outer_size,
                                     hyp['num_classes']])

    pred_confidences_squash = tf.nn.softmax(pred_logits_squash)
    pred_confidences = tf.reshape(pred_confidences_squash,
                                  [outer_size, hyp['rnn_len'],
                                   hyp['num_classes']])

    depth_deep_feat = tf.reshape(logits['depth_deep_feat'], (outer_size, 128))
    depth_weights = tf.get_variable('depth_out', 
                                    shape=(128, 1))
    tf.add_to_collection('trainable', depth_weights)
    tf.add_to_collection('depth', depth_weights)

    depth_weights_decay = tf.nn.l2_loss(depth_weights) * 1e-5
    tf.add_to_collection('new_weights', depth_weights_decay)
    tf.add_to_collection('depth_decay', depth_weights_decay)

    pred_depths = tf.reshape(tf.matmul(depth_deep_feat, depth_weights),
                             [outer_size, 1])
    
    location_deep_feat = tf.reshape(logits['location_deep_feat'], (outer_size, 128))
    location_weights = tf.get_variable('location_out', 
                                        shape=(128, 2))
    tf.add_to_collection('trainable', location_weights)
    tf.add_to_collection('location', location_weights)

    location_weights_decay = tf.nn.l2_loss(location_weights) * 1e-5
    tf.add_to_collection('new_weights', location_weights_decay)
    tf.add_to_collection('location_decay', location_weights_decay)

    pred_locations_offset = tf.matmul(location_deep_feat, location_weights)

    #pred_locations_offset = tf.reshape(pred_boxes, (outer_size, 4))[:, :2]
    pred_locations_proj = tf.reshape(pred_locations_offset + xy_offset, (outer_size, 2))

    #pred_locations_proj = tf.reshape(xy_offset, (outer_size, 2))


    calib = tf.reshape(labels[3], (outer_size, 3, 4))
    xy_scale = tf.reshape(labels[5], (outer_size, 2))   
    pred_locations_proj = pred_locations_proj / (xy_scale + 1e-7)

    principle_points = tf.reshape(calib[:, :2, 2], (outer_size, 2))
    translations = tf.reshape(calib[:, :2, 3], (outer_size, 2))
    focal_length = tf.reshape(calib[:, 0, 0], (outer_size, 1))
    """
    if train:
        boxes, dimensions, location, alpha = tf.split(labels[1], [4, 3, 3, 1], 3)
        location_x, location_y, depths = tf.split(location, [1, 1, 1], 3)
        use_depths = tf.reshape(depths, (outer_size, 1))
    else:
        use_depths = tf.reshape(pred_depths, (outer_size, 1))
    """


    use_depths = tf.reshape(pred_depths, (outer_size, 1))

    pred_depths = tf.reshape(pred_depths, (outer_size, 1))

    pred_xy = ((pred_locations_proj - principle_points) * use_depths - translations) / (focal_length + 1e-7)
    pred_locations_3d = tf.concat([pred_xy, use_depths], axis=1)

    return pred_boxes, pred_logits, pred_confidences, pred_depths, pred_locations_3d

def _build_rezoom_layer(hyp, rezoom_input, data_dict, logits, train):
    with tf.name_scope('rezoom_layer'):
        grid_size = hyp['grid_width'] * hyp['grid_height']
        outer_size = grid_size * hyp['batch_size']

        pred_boxes, pred_logits, pred_confidences, pred_depths, pred_locations, early_feat, \
            hidden_output = rezoom_input

        early_feat_channels = hyp['early_feat_channels']
        early_feat = early_feat[:, :, :, :early_feat_channels]

        w_offsets = hyp['rezoom_w_coords']
        h_offsets = hyp['rezoom_h_coords']
        num_offsets = len(w_offsets) * len(h_offsets)
        rezoom_features = _rezoom(
            hyp, pred_boxes, early_feat, early_feat_channels,
            w_offsets, h_offsets)
        if train:
            rezoom_features = tf.nn.dropout(rezoom_features, 0.5)

        delta_features = tf.concat(
            axis=1,
            values=[hidden_output,
                    rezoom_features[:, 0, :] / 1000.])
        dim = 128
        shape = [hyp['num_inner_channel'] + 
                 early_feat_channels * num_offsets,
                 dim]

        trained_delta1_weights = tf.constant_initializer(data_dict['delta1'])
        delta_weights1 = tf.get_variable('delta1', initializer=trained_delta1_weights, 
                                         shape=data_dict['delta1'].shape)
        # TODO: maybe adding dropout here?
        ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights1))
        if train:
            ip1 = tf.nn.dropout(ip1, 0.5)        

        trained_delta2_weights = tf.constant_initializer(value=data_dict['delta2'])
        delta_confs_weights = tf.get_variable(
            'delta2', initializer=trained_delta2_weights, shape=[dim, hyp['num_classes']])

        confidence_weights = (delta_weights1, delta_confs_weights)

        trained_delta_boxes_weights = tf.constant_initializer(value=data_dict['delta_boxes'])
        delta_boxes_weights = tf.get_variable('delta_boxes', initializer=trained_delta_boxes_weights, shape=[dim, 4])

        rere_feature = tf.matmul(ip1, delta_boxes_weights) * 5
        pred_boxes_delta = (tf.reshape(rere_feature, [outer_size, 1, 4]))

        scale = hyp.get('rezoom_conf_scale', 50)
        feature2 = tf.matmul(ip1, delta_confs_weights) * scale
        pred_confs_delta = tf.reshape(feature2, [outer_size, 1,
                                      hyp['num_classes']])

        pred_confs_delta = tf.reshape(pred_confs_delta,
                                      [outer_size, hyp['num_classes']])

        pred_confidences_squash = tf.nn.softmax(pred_confs_delta)
        pred_confidences = tf.reshape(pred_confidences_squash,
                                      [outer_size, hyp['rnn_len'],
                                       hyp['num_classes']])

        pred_depths_delta = tf.zeros(shape=(outer_size, 1))
        pred_locations_delta = tf.zeros(shape=(outer_size, 3))

        return pred_boxes, pred_logits, pred_confidences, \
            pred_confs_delta, pred_boxes_delta, pred_depths_delta, pred_locations_delta, confidence_weights

def _bbox_from_corners(hyp, global_corners, calib, xy_scale):
    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']
    global_corners_expand = tf.concat([tf.reshape(global_corners, (outer_size, 3, 8)),
                                       tf.constant(np.ones((outer_size, 1, 8)), dtype=tf.float32)],
                                       axis=1)
    calib_r = tf.reshape(calib, (outer_size, 3, 4))
    xy_scale_r = tf.reshape(xy_scale, (outer_size, 2))
    xz = tf.reduce_sum(tf.reshape(calib_r[:, 0, :], (outer_size, 4, 1)) * global_corners_expand, axis=1)
    yz = tf.reduce_sum(tf.reshape(calib_r[:, 1, :], (outer_size, 4, 1)) * global_corners_expand, axis=1)
    z = tf.reduce_sum(tf.reshape(calib_r[:, 2, :], (outer_size, 4, 1)) * global_corners_expand, axis=1)
    x_scale, y_scale = tf.split(xy_scale_r, [1, 1], axis=1)
    x = tf.clip_by_value(xz * x_scale / (z + 1e-5), 0, hyp['image_width'] - 1) 
    y = tf.clip_by_value(yz * y_scale / (z + 1e-5), 0, hyp['image_height'] - 1)
    left = tf.reduce_min(x, axis=1, keep_dims=True)
    right = tf.reduce_max(x, axis=1, keep_dims=True)
    top = tf.reduce_min(y, axis=1, keep_dims=True)
    bottom = tf.reduce_max(y, axis=1, keep_dims=True)
    coarse_stride = hyp['region_size']
    batch_ids = []
    x_offsets = []
    y_offsets = []
    for n in range(hyp['batch_size']):
        for i in range(hyp['grid_height']):
            for j in range(hyp['grid_width']):
                for k in range(hyp['rnn_len']):
                    batch_ids.append(n)
                    x_offsets.append([coarse_stride / 2. + coarse_stride * j])
                    y_offsets.append([coarse_stride / 2. + coarse_stride * i])
    batch_ids = tf.constant(batch_ids)
    x_offsets = tf.reshape(tf.constant(x_offsets), (outer_size, 1))
    y_offsets = tf.reshape(tf.constant(y_offsets), (outer_size, 1))
    pred_bbox_proj = tf.concat([(left + right) * 0.5 - x_offsets, (top + bottom) * 0.5 - y_offsets,
                                 right - left, bottom - top], axis=1)
    return pred_bbox_proj

def _build_td_confidence_layer(hyp, dlogits, early_feat, hidden_output, confidence_weights, labels, train):
    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']
    corners = tf.reshape(dlogits['pred_corners'], (outer_size, 3, 8))
    locations = tf.reshape(dlogits['pred_locations_new'], (outer_size, 3, 1))
    global_corners = locations + corners
    calib, xy_scale = labels[3], labels[5]
    pred_bbox_proj = _bbox_from_corners(hyp, global_corners, calib, xy_scale)

    early_feat_channels = hyp['early_feat_channels']
    early_feat = early_feat[:, :, :, :early_feat_channels]
    w_offsets = hyp['rezoom_w_coords']
    h_offsets = hyp['rezoom_h_coords']
    num_offsets = len(w_offsets) * len(h_offsets)
    rezoom_features = _rezoom(
            hyp, pred_bbox_proj, early_feat, early_feat_channels,
            w_offsets, h_offsets)
    if train:
        rezoom_features = tf.nn.dropout(rezoom_features, 0.5)
    delta_features = tf.concat(
        axis=1,
        values=[hidden_output,
                rezoom_features[:, 0, :] / 1000.])
    dim = 128
    shape = [hyp['num_inner_channel'] +
             early_feat_channels * num_offsets,
             dim]
    delta_weights1, delta_confs_weights = confidence_weights
    ip1 = tf.nn.relu(tf.matmul(delta_features, delta_weights1))
    if train:
        ip1 = tf.nn.dropout(ip1, 0.5)
    scale = hyp.get('rezoom_conf_scale', 50)
    feature2 = tf.matmul(ip1, delta_confs_weights) * scale
    pred_td_confs_delta = tf.reshape(feature2, [outer_size, hyp['rnn_len'],
                                     hyp['num_classes']])
    pred_td_confidences_squash = tf.nn.softmax(pred_td_confs_delta)
    pred_td_confidences = tf.reshape(pred_td_confidences_squash,
                                    [outer_size, hyp['rnn_len'],
                                     hyp['num_classes']])
    return pred_td_confidences, pred_bbox_proj

def _build_refine_layer(hyp, logits, pred_bbox_proj):
    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']
    out_channels = 256
    crop_size = 16
    depth_feat = _roi_align_keep_ratio(hyp, pred_bbox_proj, logits['depth_deep_feat'], 
                                       out_channels, crop_size, 'refine_depth_feat', 'refine')
    location_feat = _roi_align_keep_ratio(hyp, pred_bbox_proj, logits['location_early_feat'], 
                                          out_channels, crop_size, 'refine_location_feat', 'refine')
    corner_feat = _roi_align_keep_ratio(hyp, pred_bbox_proj, logits['corner_early_feat'], 
                                        out_channels, crop_size, 'refine_corner_feat', 'refine')
    refine_feat = depth_feat + location_feat + corner_feat

    filt_1 = tf.get_variable('refine_filt_1', shape=(3, 3, out_channels, 256))
    bias_1 = tf.get_variable('refine_bias_1', shape=(256, ))
    filt_2 = tf.get_variable('refine_filt_2', shape=(3, 3, 256, 256))
    bias_2 = tf.get_variable('refine_bias_2', shape=(256, ))
    fc_weight = tf.get_variable('refine_fc_weight', shape=(256*(crop_size // 4)**2, 24)) 

    conv_1 = tf.nn.conv2d(refine_feat, filt_1, [1, 2, 2, 1], padding='SAME')
    add_bias_1 = tf.nn.bias_add(conv_1, bias_1)
    relu_1 = tf.nn.relu(add_bias_1)

    conv_2 = tf.nn.conv2d(relu_1, filt_2, [1, 2, 2, 1], padding='SAME')
    add_bias_2 = tf.nn.bias_add(conv_2, bias_2)
    relu_2 = tf.nn.relu(add_bias_2)

    feat_r = tf.reshape(relu_2, (outer_size, -1))
    delta_global_corners = 2.0 * tf.tanh(tf.matmul(feat_r, fc_weight))
    
    vars = [filt_1, bias_1, filt_2, bias_2, fc_weight]
    for var in vars:
        tf.add_to_collection('trainable', var)
        tf.add_to_collection('refine', var)
        var_decay = tf.nn.l2_loss(var) * 1e-4
        tf.add_to_collection('new_weights', var_decay)
        tf.add_to_collection('refine_decay', var_decay)
    return delta_global_corners

def decoder(hyp, logits, labels, train):
    """Apply decoder to the logits.

    Computation which decode CNN boxes.
    The output can be interpreted as bounding Boxes.


    Args:
      logits: Logits tensor, output von encoder

    Return:
      decoded_logits: values which can be interpreted as bounding boxes
    """
    hyp['rnn_len'] = 1
    encoded_features = logits['deep_feat']

    batch_size = hyp['batch_size']
    hyp['solver']['batch_size'] = batch_size
    if not train:
        hyp['batch_size'] = 1


    grid_size = hyp['grid_width']*hyp['grid_height']
    outer_size = grid_size*hyp['batch_size']

    early_feat = logits['early_feat']

    initializer = tf.random_uniform_initializer(-0.1, 0.1)

    with tf.variable_scope('decoder', initializer=initializer):
        with tf.name_scope('inner_layer'):
            # Build inner layer.
            # See https://arxiv.org/abs/1612.07695 fig. 2 for details
            hidden_output, data_dict = _build_inner_layer(hyp, encoded_features, train)

        with tf.name_scope('output_layer'):
            # Build output layer
            # See https://arxiv.org/abs/1612.07695 fig. 2 for details
            calib_pinv = None
            pred_boxes, pred_logits, pred_confidences, pred_depths, pred_locations = _build_output_layer(
                hyp, hidden_output, data_dict, logits, calib_pinv, labels, train)

        # Dictionary filled with return values
        dlogits = {}
        current_pred_boxes = pred_boxes
        if hyp['use_rezoom']:
            rezoom_input = pred_boxes, pred_logits, pred_confidences, pred_depths, pred_locations, \
                early_feat, hidden_output
            # Build rezoom layer
            # See https://arxiv.org/abs/1612.07695 fig. 2 for details
            rezoom_output = _build_rezoom_layer(hyp, rezoom_input, data_dict, logits, train)

            pred_boxes, pred_logits, pred_confidences, \
                pred_confs_deltas, pred_boxes_deltas, pred_depths_deltas, pred_locations_deltas, confidence_weights = rezoom_output

            dlogits['pred_confs_deltas'] = pred_confs_deltas
            dlogits['pred_boxes_deltas'] = pred_boxes_deltas
         
            dlogits['pred_boxes_new'] = pred_boxes + pred_boxes_deltas

            dlogits['pred_depths_deltas'] = pred_depths_deltas

            dlogits['pred_depths_new'] = pred_depths + pred_depths_deltas

            dlogits['pred_locations_deltas'] = pred_locations_deltas

            dlogits['pred_locations_new'] = pred_locations_deltas + pred_locations
 
            current_pred_boxes = dlogits['pred_boxes_new']

        with tf.name_scope('corner_regression_layer'):
            dlogits['pred_corners'] = _build_corner_regression_layer(hyp, current_pred_boxes, logits['corner_early_feat'])

    
        dlogits['pred_td_confidence'], dlogits['pred_bbox_proj'] = _build_td_confidence_layer(hyp, dlogits, early_feat, hidden_output, confidence_weights, labels, train)

        dlogits['delta_global_corners'] = _build_refine_layer(hyp, logits, dlogits['pred_bbox_proj'])

        dlogits['pred_global_corners'] = tf.reshape(tf.reshape(pred_locations, (outer_size, 3, 1)) + tf.reshape(dlogits['pred_corners'], (outer_size, 3, 8)), (outer_size, 24))

        dlogits['refined_global_corners'] = dlogits['pred_global_corners'] + dlogits['delta_global_corners']

    # Fill dict with return values
    dlogits['pred_boxes'] = pred_boxes
    dlogits['pred_logits'] = pred_logits
    dlogits['pred_confidences'] = pred_confidences
    dlogits['pred_depths'] = pred_depths
    dlogits['pred_locations'] = pred_locations

    hyp['batch_size'] = batch_size

    return dlogits


def _add_rezoom_loss_histograms(hypes, pred_boxes_deltas):
    """
    Add some histograms to tensorboard.
    """
    return

def _compute_rezoom_loss(hypes, rezoom_loss_input, slow=False):
    """
    Computes loss for delta output. Only relevant
    if rezoom layers are used.
    """

    grid_size = hypes['grid_width'] * hypes['grid_height']
    outer_size = grid_size * hypes['batch_size']
    head = hypes['solver']['head_weights']
    regression_weights = hypes['solver']['regression_weights']

    perm_truth, depths_truth, locations_truth, pred_boxes, classes, pred_mask, pred_depths, pred_locations, \
        pred_confs_deltas, pred_boxes_deltas, pred_depths_deltas, pred_locations_deltas, mask_r = rezoom_loss_input
    if hypes['rezoom_change_loss'] == 'center':
        error = (perm_truth[:, :, 0:2] - pred_boxes[:, :, 0:2]) \
            / tf.maximum(perm_truth[:, :, 2:4], 1.)
        square_error = tf.reduce_sum(tf.square(error), 2)
        inside = tf.reshape(tf.to_int64(
            tf.logical_and(tf.less(square_error, 0.2**2),
                           tf.greater(classes, 0))), [-1])
    elif hypes['rezoom_change_loss'] == 'iou':
        pred_boxes_flat = tf.reshape(pred_boxes, [-1, 4])
        perm_truth_flat = tf.reshape(perm_truth, [-1, 4])
        iou = train_utils.iou(train_utils.to_x1y1x2y2(pred_boxes_flat),
                              train_utils.to_x1y1x2y2(perm_truth_flat))
        inside = tf.reshape(tf.to_int64(tf.greater(iou, 0.5)), [-1])
    else:
        assert not hypes['rezoom_change_loss']
        inside = tf.reshape(tf.to_int64((tf.greater(classes, 0))), [-1])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred_confs_deltas, labels=inside)

    delta_confs_loss = tf.reduce_sum(cross_entropy*mask_r) \
        / outer_size * hypes['solver']['head_weights'][0] * 0.1

    

    delta_unshaped = perm_truth - (pred_boxes + pred_boxes_deltas)

    delta_residual = tf.reshape(delta_unshaped * pred_mask,
                                [outer_size, hypes['rnn_len'], 4])
    sqrt_delta = tf.minimum(tf.square(delta_residual), 10. ** 2)
    delta_boxes_loss = tf.reduce_sum(sqrt_delta) / outer_size * head[1] * 0.05 
   
    pred_mask = tf.squeeze(pred_mask, 2)

    """
    delta_depths_unshaped = (depths_truth - (pred_depths + pred_depths_deltas)) / (depths_truth + 1e-5)
    delta_depths_residual = tf.reshape(delta_depths_unshaped * pred_mask, 
                                       [outer_size, hypes['rnn_len'], 1])
    sqrt_delta_depths = tf.abs(delta_depths_residual)
    delta_depths_loss = (tf.reduce_sum(sqrt_delta_depths) / (1e-5 + tf.reduce_sum(pred_mask))) * head[1] * 0.1 * regression_weights[0]
    """

    delta_depths_loss = tf.reduce_sum(tf.abs((depths_truth - (pred_depths + pred_depths_deltas))) * tf.reshape(pred_mask, (outer_size, 1))) * 0.05 / outer_size

    delta_locations_loss = tf.reduce_sum(tf.abs((locations_truth - (pred_locations + pred_locations_deltas))) * pred_mask) * 0.05 / outer_size

    return delta_confs_loss, delta_boxes_loss, delta_depths_loss, delta_locations_loss


def loss(hypes, decoded_logits, labels, slow=False):
    """Calculate the loss from the logits and the labels.

    Args:
      decoded_logits: output of decoder
      labels: Labels tensor; Output from data_input

      flags: 0 if object is present 1 otherwise
      confidences: ??
      boxes: encoding of bounding box location

    Returns:
      loss: Loss tensor of type float.
    """

    confidences, boxes, mask, calib, calib_pinv, z_3d_scale = labels
    boxes, dimensions, location, alpha = tf.split(boxes, [4, 3, 3, 1], 3)
    location_x, location_y, depths = tf.split(location, [1, 1, 1], 3)
    true_corners = compute_corners(hypes, dimensions, alpha)

    pred_boxes = decoded_logits['pred_boxes']
    pred_logits = decoded_logits['pred_logits']
    pred_confidences = decoded_logits['pred_confidences']
    

    pred_confs_deltas = decoded_logits['pred_confs_deltas']
    pred_boxes_deltas = decoded_logits['pred_boxes_deltas']

    pred_depths = decoded_logits['pred_depths']
    pred_depths_deltas = decoded_logits['pred_depths_deltas']

    pred_locations = decoded_logits['pred_locations']
    pred_locations_deltas = decoded_logits['pred_locations_deltas']

    pred_corners = decoded_logits['pred_corners']

    refined_global_corners = decoded_logits['refined_global_corners']

    grid_size = hypes['grid_width'] * hypes['grid_height']
    outer_size = grid_size * hypes['batch_size']

    true_global_corners = tf.reshape(tf.reshape(location, (outer_size, 3, 1)) + tf.reshape(true_corners, (outer_size, 3, 8)), (outer_size, 24))


    head = hypes['solver']['head_weights']

    regression_weights = hypes['solver']['regression_weights']

    # Compute confidence loss
    confidences = tf.reshape(confidences, (outer_size, 1))
    true_classes = tf.reshape(tf.cast(tf.greater(confidences, 0), 'int64'),
                              [outer_size])

    pred_classes = tf.reshape(pred_logits, [outer_size, hypes['num_classes']])
    mask_r = tf.reshape(mask, [outer_size])

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=pred_classes, labels=true_classes)

    # ignore don't care areas
    cross_entropy_sum = (tf.reduce_sum(mask_r*cross_entropy))
    confidences_loss = cross_entropy_sum / outer_size * head[0]

    true_boxes = tf.reshape(boxes, (outer_size, hypes['rnn_len'], 4))

    # box loss for background prediction needs to be zerod out
    boxes_mask = tf.reshape(
        tf.cast(tf.greater(confidences, 0), 'float32'), (outer_size, 1, 1))

    # danger zone
    residual = (true_boxes - pred_boxes) * boxes_mask

    boxes_loss = tf.reduce_sum(tf.abs(residual)) / outer_size * head[1]

    true_depths = tf.reshape(depths, (outer_size, 1))
   
    """ 
    depths_residual = ((true_depths - pred_depths) * boxes_mask) / (true_depths + 1e-5)
    depths_loss = tf.reduce_sum(tf.abs(depths_residual)) / (1e-5 + tf.reduce_sum(boxes_mask)) * head[1] * regression_weights[0]
    """
 
    depths_loss = tf.reduce_sum(tf.abs(true_depths - pred_depths) * tf.reshape(boxes_mask, (outer_size, 1))) / outer_size

    true_locations = tf.reshape(location, (outer_size, 3))    

    locations_loss = tf.reduce_sum(tf.abs(true_locations - pred_locations) * tf.reshape(boxes_mask, (outer_size, 1))) / outer_size

    boxes_mask = tf.reshape(boxes_mask, (outer_size, 1))
    corners_loss = tf.reduce_sum(tf.abs(true_corners - pred_corners) * boxes_mask) / outer_size * 0.5


    refine_loss = tf.reduce_sum(tf.abs(refined_global_corners - true_global_corners) * boxes_mask) / outer_size * 0.5

    boxes_mask = tf.reshape(boxes_mask, (outer_size, 1, 1))     

    joint_2d_3d = False
    joint_3d = False
    depth = False
    location = False
    corners = False
    refine = True

    location_refine = False

    reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES

    if hypes['use_rezoom']:
        # add rezoom loss
        rezoom_loss_input = true_boxes, true_depths, true_locations, \
                            pred_boxes, confidences, boxes_mask, pred_depths, pred_locations, \
                            pred_confs_deltas, pred_boxes_deltas, pred_depths_deltas, \
                            pred_locations_deltas, mask_r

        delta_confs_loss, delta_boxes_loss, delta_depths_loss, delta_locations_loss = _compute_rezoom_loss(
            hypes, rezoom_loss_input)

        _add_rezoom_loss_histograms(hypes, pred_boxes_deltas)


        if joint_2d_3d:
            loss = 10 * depths_loss + 10 * delta_depths_loss + 10 * locations_loss + 10 * delta_locations_loss + 10 * corners_loss + \
                   confidences_loss + delta_confs_loss + boxes_loss + delta_boxes_loss + 10 * refine_loss
            weights_loss = 0.1 * tf.add_n(tf.get_collection('new_weights')) + 0.1 * tf.add_n(tf.get_collection(reg_loss_col),
                                                                                     name='reg_loss')

        elif joint_3d:
            loss = depths_loss + delta_depths_loss + locations_loss + delta_locations_loss + corners_loss + refine_loss
            weights_loss = 0.1 * tf.add_n(tf.get_collection('new_weights'))

        elif depth:
            loss = depths_loss + delta_depths_loss
            weights_loss = 0.1 * tf.add_n(tf.get_collection('depth_decay')) 
        elif location:
            loss = locations_loss + delta_locations_loss
            weights_loss = 0.1 * tf.add_n(tf.get_collection('location_decay'))
        elif corners:
            loss = corners_loss
            weights_loss = 0.1 * tf.add_n(tf.get_collection('corners_decay'))

        elif refine:
            loss = refine_loss
            weights_loss = 0.1 * tf.add_n(tf.get_collection('refine_decay'))

        elif location_refine:
            loss = refine_loss + locations_loss + delta_locations_loss
            weights_loss = 0.1 * tf.add_n(tf.get_collection('refine_decay')) + 0.1 * tf.add_n(tf.get_collection('location_decay'))

  
    tf.add_to_collection('total_losses', tf.reduce_sum(loss))  

    total_loss = tf.reduce_sum(weights_loss + loss) 
    losses = {}
    losses['total_loss'] = total_loss
    losses['loss'] = loss
    losses['confidences_loss'] = confidences_loss
    losses['boxes_loss'] = boxes_loss
    losses['weight_loss'] = weights_loss
    losses['depths_loss'] = depths_loss
    losses['locations_loss'] = locations_loss
    losses['corners_loss'] = corners_loss
    losses['refine_loss'] = refine_loss

    if hypes['use_rezoom']:
        losses['delta_boxes_loss'] = delta_boxes_loss
        losses['delta_confs_loss'] = delta_confs_loss
        losses['delta_depths_loss'] = delta_depths_loss
        losses['delta_locations_loss'] = delta_locations_loss
    return losses
def evaluation(hyp, images, labels, decoded_logits, losses, global_step):
    """
    Compute summary metrics for tensorboard
    """

    pred_confidences = decoded_logits['pred_confidences']
    pred_boxes = decoded_logits['pred_boxes']
    pred_depths = decoded_logits['pred_depths_new']
    pred_locations = decoded_logits['pred_locations_new']
    pred_corners = decoded_logits['pred_corners']

    pred_global_corners = decoded_logits['pred_global_corners']
    refined_global_corners = decoded_logits['refined_global_corners']

    # Estimating Accuracy
    grid_size = hyp['grid_width'] * hyp['grid_height']
    outer_size = grid_size * hyp['batch_size']
    confidences, boxes, mask, calib, calib_pinv, z_3d_scale = labels
    boxes_mask = tf.reshape(tf.cast(tf.greater(confidences, 0), 'float32'), (outer_size, 1))
    positive_count = tf.reduce_sum(boxes_mask)

    boxes, dimensions, location, alpha = tf.split(boxes, [4, 3, 3, 1], 3)
    location_x, location_y, depths = tf.split(location, [1, 1, 1], 3)

    new_shape = [hyp['batch_size'], hyp['grid_height'],
                 hyp['grid_width'], hyp['num_classes']]
    pred_confidences_r = tf.reshape(pred_confidences, new_shape)
    # Set up summary operations for tensorboard
    a = tf.equal(tf.cast(confidences, 'int64'),
                 tf.argmax(pred_confidences_r, 3))

    accuracy = tf.reduce_mean(tf.cast(a, 'float32'), name='/accuracy')

    true_depths = tf.reshape(depths, (outer_size, 1))
    pred_depths_r = tf.reshape(pred_depths, (outer_size, 1))
    depths_error = tf.reduce_sum(tf.abs(pred_depths_r - true_depths) * boxes_mask) / (positive_count + 1e-2)

    true_locations = tf.reshape(location, (outer_size, 3))
    pred_locations_r = tf.reshape(pred_locations, (outer_size, 3))
    locations_error = tf.reduce_sum(tf.abs(pred_locations_r - true_locations) * boxes_mask) / (positive_count + 1e-2)

    true_corners = compute_corners(hyp, dimensions, alpha)
    true_global_corners = tf.reshape(tf.reshape(location, (outer_size, 3, 1)) + tf.reshape(true_corners, (outer_size, 3, 8)), (outer_size, 24))
    global_corners_error = tf.reduce_sum(tf.abs(pred_global_corners - true_global_corners) * boxes_mask) / (positive_count * 24 + 1e-2)
    refined_corners_error = tf.reduce_sum(tf.abs(refined_global_corners - true_global_corners) * boxes_mask) / (positive_count * 24 + 1e-2)

    '''
    eval_list = []
    eval_list.append(('Acc.', accuracy))
    eval_list.append(('Conf', losses['confidences_loss']))
    eval_list.append(('Box', losses['boxes_loss']))
    eval_list.append(('Weight', losses['weight_loss']))
    eval_list.append(('Depth', losses['depths_loss']))
    eval_list.append(('Location', losses['locations_loss']))
    eval_list.append(('Corner', losses['corners_loss']))
    if hyp['use_rezoom']:
        eval_list.append(('Delta', tf.reduce_sum(losses['delta_boxes_loss'] + \
                                                 losses['delta_confs_loss'] + \
                                                 losses['delta_depths_loss'] + \
                                                 losses['delta_locations_loss'])))
    '''

    eval_list = []
    eval_list.append(('Weight', losses['weight_loss']))
    eval_list.append(('Boxes', losses['boxes_loss']))
    eval_list.append(('Delta', losses['delta_boxes_loss']))
    eval_list.append(('Confidence', losses['confidences_loss']))
    eval_list.append(('Delta', losses['delta_confs_loss']))
    eval_list.append(('Depth', losses['depths_loss']))
    #eval_list.append(('Delta', losses['delta_depths_loss']))
    eval_list.append(('Error', depths_error))
    eval_list.append(('Location', losses['locations_loss']))
    #eval_list.append(('Delta', losses['delta_locations_loss']))
    eval_list.append(('Error', locations_error))
    eval_list.append(('Corner', losses['corners_loss']))
    eval_list.append(('Error', global_corners_error))
    eval_list.append(('Refine', refined_corners_error))

    # Log Images
    # show ground truth to verify labels are correct
    pred_confidences_r = tf.reshape(
        pred_confidences,
        [hyp['batch_size'], grid_size, hyp['rnn_len'], hyp['num_classes']])

    # show predictions to visualize training progress
    pred_boxes_r = tf.reshape(
        pred_boxes, [hyp['batch_size'], grid_size, hyp['rnn_len'],
                     4])

    pred_corners_r = tf.reshape(pred_corners, [hyp['batch_size'], grid_size, 24])

    pred_depths_r = tf.reshape(pred_depths, [hyp['batch_size'], grid_size, hyp['rnn_len'], 1])
    pred_locations_r = tf.reshape(pred_locations, [hyp['batch_size'], grid_size, hyp['rnn_len'], 3])

    test_pred_confidences = pred_confidences_r[0, :, :, :]
    test_pred_boxes = pred_boxes_r[0, :, :, :]
    test_pred_depths = pred_depths_r[0, :, :, :]
    test_pred_locations = pred_locations_r[0, :, :, :]

    test_pred_corners = pred_corners_r[0, :, :]

    def log_image(np_img, np_confidences, np_boxes, np_depths, np_locations, np_corners, np_global_step, pred_or_true):

        if pred_or_true == 'pred':
            plot_image = train_utils.add_rectangles(
                hyp, np_img, np_confidences, np_boxes, np_depths, np_locations, np_corners, use_stitching=True,
                rnn_len=hyp['rnn_len'])[0]
        else:
            np_mask = np_boxes
            plot_image = data_utils.draw_encoded(
                np_img[0], np_confidences[0], mask=np_mask[0], cell_size=32)

        num_images = 20

        filename = '%s_%s.jpg' % \
            ((np_global_step // hyp['logging']['write_iter'])
                % num_images, pred_or_true)
        img_path = os.path.join(hyp['dirs']['output_dir'], filename)

        scp.misc.imsave(img_path, plot_image)

        return plot_image

    pred_log_img = tf.py_func(log_image,
                              [images, test_pred_confidences,
                               test_pred_boxes, test_pred_depths, test_pred_locations, test_pred_corners,
                               global_step, 'pred'], [tf.float32])

    true_log_img = tf.py_func(log_image,
                              [images, confidences, mask, test_pred_depths, test_pred_locations, test_pred_corners,
                               global_step, 'true'],
                              [tf.uint8])
    return eval_list
