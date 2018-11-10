#!/usr/bin/env python
"""This file contains all the model information: the training steps, the batch size and the model iself."""

import tensorflow as tf


def get_training_steps():
    """Returns the number of batches that will be used to train the CNN.
    It is recommended to change this value."""
    return 50


def get_batch_size():
    """Returns the batch size that will be used by the CNN.
    It is recommended to change this value."""
    return 1


def cnn(features, labels, mode):
    """Returns an EstimatorSpec that is constructed using a CNN that you have to write below."""
    # Input Layer (a batch of images that have 64x64 pixels and are RGB colored (3)
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

    # TODO: Code of the CNN

    if mode == tf.estimator.ModeKeys.PREDICT:
        # TODO: return tf.estimator.EstimatorSpec with prediction values of all classes

    if mode == tf.estimator.ModeKeys.TRAIN:
        # TODO: Let the model train here
        # TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        # The classes variable below exists of an tensor that contains all the predicted classes in a batch
        # TODO: eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=classes)}
        # TODO: return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
