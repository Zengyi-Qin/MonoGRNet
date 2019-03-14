#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the MediSeg model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

import scipy as scp
import scipy.misc

import numpy as np

import tensorflow as tf

import utils.train_utils
import time

import random

from utils.annolist import AnnotationLib as AnnLib

import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_val_dir(hypes, validation=True):
    if validation:
        val_dir = os.path.join(hypes['dirs']['output_dir'], 'val_out')
    else:
        val_dir = os.path.join(hypes['dirs']['output_dir'], 'train_out')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    return val_dir


def make_img_dir(hypes):
    val_dir = os.path.join(hypes['dirs']['output_dir'], 'val_images')
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    return val_dir


def write_rects(rects, filename):
    with open(filename, 'w') as f:
        for rect in rects:
            string = "Car -1 -1 -10 %f %f %f %f %f %f %f %f %f %f %f %f" % \
                (rect.x1, rect.y1, rect.x2, rect.y2,
                 rect.height_3d, rect.width_3d, rect.length_3d, rect.x_3d, rect.y_3d, rect.z_3d, \
                 rect.alpha, rect.score)
            print(string, file=f)


def evaluate(hypes, sess, image_pl, calib_pl, xy_scale_pl, softmax):
    pred_annolist, image_list, dt, dt2 = get_results(
        hypes, sess, image_pl, calib_pl, xy_scale_pl, softmax, True)

    val_path = make_val_dir(hypes)

    eval_list = []

    eval_cmd = os.path.join(hypes['dirs']['base_path'],
                            hypes['data']['eval_cmd'])

    label_dir = os.path.join(hypes['dirs']['data_dir'],
                             hypes['data']['label_dir'])

    try:
        subprocess.check_call([eval_cmd, val_path, label_dir])
    except OSError as error:
        logging.warning("Failed to run run kitti evaluation code.")
        logging.warning("Please run: `cd submodules/KittiObjective2/ && make`")
        logging.warning("For more information see:"
                        "`submodules/KittiObjective2/README.md`")
        exit(1)
        img_dir = make_img_dir(hypes)
        logging.info("Output images have been written to {}.".format(img_dir))
        eval_list.append(('Speed (msec)', 1000*dt))
        eval_list.append(('Speed (fps)', 1/dt))
        eval_list.append(('Post (msec)', 1000*dt2))
        return eval_list, image_list

    res_file = os.path.join(val_path, "stats_car_detection.txt")

    with open(res_file) as f:
        for mode in ['easy', 'medium', 'hard']:
            line = f.readline()
            result = np.array(line.rstrip().split(" ")).astype(float)
            mean = np.mean(result)
            eval_list.append(("val   " + mode, mean))

    pred_annolist, image_list2, dt, dt2 = get_results(
        hypes, sess, image_pl, calib_pl, xy_scale_pl, softmax, False)

    val_path = make_val_dir(hypes, False)
    subprocess.check_call([eval_cmd, val_path, label_dir])
    res_file = os.path.join(val_path, "stats_car_detection.txt")

    with open(res_file) as f:
        for mode in ['easy', 'medium', 'hard']:
            line = f.readline()
            result = np.array(line.rstrip().split(" ")).astype(float)
            mean = np.mean(result)
            eval_list.append(("train   " + mode, mean))

    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))
    eval_list.append(('Post (msec)', 1000*dt2))

    return eval_list, image_list


def get_results(hypes, sess, image_pl, calib_pl, xy_scale_pl, decoded_logits, validation=True):

   
    pred_boxes = decoded_logits['pred_boxes_new']
   
    #pred_boxes = decoded_logits['pred_bbox_proj']
    pred_depths = decoded_logits['pred_depths_new']
    pred_locations = decoded_logits['pred_locations_new']
    pred_confidences = decoded_logits['pred_confidences']
    pred_corners = decoded_logits['pred_corners']

    refined_global_corners = decoded_logits['refined_global_corners']

    # Build Placeholder
    shape = [hypes['image_height'], hypes['image_width'], 3]

    if validation:
        kitti_txt = os.path.join(hypes['dirs']['data_dir'],
                                 hypes['data']['val_file'])
    else:
        kitti_txt = os.path.join(hypes['dirs']['data_dir'],
                                 hypes['data']['train_file'])
    # true_annolist = AnnLib.parse(test_idl)

    val_dir = make_val_dir(hypes, validation)
    img_dir = make_img_dir(hypes)

    image_list = []

    pred_annolist = AnnLib.AnnoList()

    files = [line.rstrip() for line in open(kitti_txt)]
    base_path = os.path.realpath(os.path.dirname(kitti_txt))

    for i, file in enumerate(files):
        image_file = file.split(" ")[0]
        if not validation and random.random() > 0.2:
            continue

        image_file_split = image_file.split('/')

        image_file = os.path.join(base_path, image_file)
        
        index = image_file_split[-1].split('.')[0]
        calib_file = os.path.join(base_path, image_file_split[0], 'calib', index + '.txt')

        orig_img = scp.misc.imread(image_file)[:, :, :3]

        xy_scale = np.reshape([hypes['image_width']*1.0/orig_img.shape[1],
                               hypes['image_height']*1.0/orig_img.shape[0]], (1, 1, 1, 2)).astype(np.float32)
        xy_scale = np.repeat(xy_scale, hypes['grid_height'], axis=1)
        xy_scale = np.repeat(xy_scale, hypes['grid_width'], axis=2)

        img = scp.misc.imresize(orig_img, (hypes["image_height"],
                                           hypes["image_width"]),
                                interp='cubic')
        calibs = [line.rstrip().split(' ') for line in open(calib_file)]
        assert calibs[2][0] == 'P2:'
        calib = np.reshape(calibs[2][1:], (1, 1, 1, 3, 4)).astype(np.float32)
        calib = np.repeat(calib, hypes['grid_height'], axis=1)
        calib = np.repeat(calib, hypes['grid_width'], axis=2)

        feed = {image_pl: img, calib_pl: calib, xy_scale_pl: xy_scale}

        (np_pred_boxes, np_pred_confidences, np_refined_global_corners) = sess.run([pred_boxes, pred_confidences, refined_global_corners], feed_dict=feed)
        """
        depth_map = np.reshape(np_pred_depths, (12, 39))
        depth_map = depth_map / np.amax(depth_map)
        depth_map[depth_map<0] = 0
        depth_map = (depth_map * 255).astype(np.uint8)

        depth_map = scp.misc.imresize(depth_map, (120, 390))
        #scp.misc.imsave('./visualize/kittiBox/{}_depth_map.png'.format(index), depth_map)
        plt.figure(figsize=(12, 4))
        plt.imshow(depth_map, cmap='winter')
        plt.savefig('./visualize/kittiBox/{}_depth_map_pred.png'.format(index))
        plt.close()

        depth_map_gt = tf.get_collection('depth_map_gt')
        np_depth_map_gt = sess.run(depth_map_gt, feed_dict=feed)[0]

        depth_map = np.reshape(np_depth_map_gt, (12, 39))
        depth_map = depth_map / np.amax(depth_map)
        depth_map[depth_map<0] = 0
        depth_map = (depth_map * 255).astype(np.uint8)

        depth_map = scp.misc.imresize(depth_map, (120, 390))
        #scp.misc.imsave('./visualize/kittiBox/{}_depth_map.png'.format(index), depth_map)
        plt.figure(figsize=(12, 4))
        plt.imshow(depth_map, cmap='winter')
        plt.savefig('./visualize/kittiBox/{}_depth_map_true.png'.format(index))
        plt.close()
        """
        
        outer_size = hypes['grid_width'] * hypes['grid_height'] * 1

        np_refined_corners = np_refined_global_corners.reshape((outer_size, 3, 8))

        np_pred_x = np.mean(np_refined_corners[:, 0, :], axis=-1, keepdims=True)
        np_pred_y = np.mean(np_refined_corners[:, 1, :4], axis=-1, keepdims=True)
        np_pred_depths = np.mean(np_refined_corners[:, 2, :], axis=-1, keepdims=True)

        np_pred_locations = np.concatenate([np_pred_x, np_pred_y, np_pred_depths], axis=1)

        np_pred_corners = np.reshape(np_refined_corners - np_pred_locations.reshape(outer_size, 3, 1), (outer_size, 24))

        pred_anno = AnnLib.Annotation()
        pred_anno.imageName = image_file
        new_img, rects = utils.train_utils.add_rectangles(
            hypes, [img], np_pred_confidences,
            np_pred_boxes, np_pred_depths, np_pred_locations, 
            np_pred_corners, show_removed=False,
            use_stitching=True, rnn_len=hypes['rnn_len'],
            min_conf=0.50, tau=hypes['tau'], color_acc=(0, 255, 0))

        if validation and i % 15 == 0:
            image_name = os.path.basename(pred_anno.imageName)
            image_name = os.path.join(img_dir, image_name)
            scp.misc.imsave(image_name, new_img)

        if validation:
            image_name = os.path.basename(pred_anno.imageName)
            image_list.append((image_name, new_img))
        # get name of file to write to
        image_name = os.path.basename(image_file)
        val_file_name = image_name.split('.')[0] + '.txt'
        val_file = os.path.join(val_dir, val_file_name)

        # write rects to file      
        for rect in rects:
            rect.calib = calib

        pred_anno.rects = rects
        pred_anno = utils.train_utils.rescale_boxes((
            hypes["image_height"],
            hypes["image_width"]),
            pred_anno, orig_img.shape[0],
            orig_img.shape[1])

        write_rects(rects, val_file)

        pred_annolist.append(pred_anno)

    start_time = time.time()
    for i in xrange(100):
        (np_pred_boxes, np_pred_confidence, np_pred_depths, np_pred_locations) = \
         sess.run([pred_boxes, pred_confidences, pred_depths, pred_locations], feed_dict=feed)
    dt = (time.time() - start_time)/100

    start_time = time.time()
    for i in xrange(100):
        utils.train_utils.compute_rectangels(
            hypes, np_pred_confidences,
            np_pred_boxes, np_pred_depths, np_pred_locations, show_removed=False,
            use_stitching=True, rnn_len=hypes['rnn_len'],
            min_conf=0.001, tau=hypes['tau'])
    dt2 = (time.time() - start_time)/100

    return pred_annolist, image_list, dt, dt2
