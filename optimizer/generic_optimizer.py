from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import logging
import sys
import tensorflow as tf


def get_learning_rate(hypes, step):
    if "learning_rates" not in hypes['solver']:
        lr = hypes['solver']['learning_rate']
        lr_step = hypes['solver']['learning_rate_step']
        if lr_step is not None:
            adjusted_lr = (lr * 0.5 ** max(0, (step / lr_step) - 2))
            return adjusted_lr
        else:
            return lr

    for i, num in enumerate(hypes['solver']['steps']):
        if step < num:
            return hypes['solver']['learning_rates'][i]

def get_regression_weights(step, init):
    depth_weight = init
    location_weight = init
    corner_weight = init 
    return np.array([depth_weight, location_weight, corner_weight], dtype=np.float32)

def training(hypes, loss, global_step, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.''
    sol = hypes["solver"]
    hypes['tensors'] = {}
    hypes['tensors']['global_step'] = global_step
    total_loss = loss['total_loss']
    with tf.name_scope('training'):

        if sol['opt'] == 'RMS':
            opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                            decay=0.9, epsilon=sol['epsilon'])
        elif sol['opt'] == 'Adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                         epsilon=sol['epsilon'])
        elif sol['opt'] == 'SGD':
            lr = learning_rate
            opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif sol['opt'] == 'Momentum':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

        else:
            raise ValueError('Unrecognized opt type')



        joint_2d_3d = False
        joint_3d = False
        depth = False
        location = False
        corners = False
        refine = True

        refine_location = False

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES

        if joint_2d_3d:
            grads_and_vars = opt.compute_gradients(total_loss)

        elif joint_3d:
            grads_and_vars = opt.compute_gradients(total_loss, tf.get_collection('trainable'))

        elif depth:
            grads_and_vars = opt.compute_gradients(total_loss, tf.get_collection('depth'))

        elif location:
            grads_and_vars = opt.compute_gradients(total_loss, tf.get_collection('location'))

        elif corners:
            grads_and_vars = opt.compute_gradients(total_loss, tf.get_collection('corners'))

        elif refine:
            grads_and_vars = opt.compute_gradients(total_loss, tf.get_collection('refine'))     

        elif refine_location:
            grads_and_vars = opt.compute_gradients(total_loss, tf.get_collection('refine') + tf.get_collection('location'))

        if hypes['clip_norm'] > 0:
            grads, tvars = zip(*grads_and_vars)
            clip_norm = hypes["clip_norm"]
            clipped_grads, norm = tf.clip_by_global_norm(grads, clip_norm)
            grads_and_vars = zip(clipped_grads, tvars)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(grads_and_vars,
                                           global_step=global_step)

    return train_op
