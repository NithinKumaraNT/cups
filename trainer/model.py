#!/usr/bin/env python
"""This file contains all the model information: the training steps, the batch size and the model iself."""

import tensorflow as tf

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

IMAGE_SIZE = 64
NUM_CLASSES = 4
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500
BATCH_SIZE = 50
NUM_TRAINING_STEPS = 20000


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")


def get_training_steps():
    """Returns the number of steps that will be used to train the CNN.
    It is recommended to change this value."""
    return NUM_TRAINING_STEPS


def get_batch_size():
    """Returns the batch size that will be used by the CNN.
    It is recommended to change this value."""
    return BATCH_SIZE


def cnn(features, labels, mode):
    """Returns an EstimatorSpec that is constructed using a CNN that you have to write below."""

    # first convert labels into
    labels = tf.cast(labels, tf.int64)

    # Input Layer (a batch of images that have 64x64 pixels and are RGB colored (3)
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

#    if mode == tf.estimator.ModeKeys.TRAIN:
#        input_layer = augment_input_layer(input_layer)

    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 3, 64],
            stddev=5e-2,
            wd=None)

        conv = tf.nn.conv2d(
            input_layer,
            kernel,
            [1, 1, 1, 1],
            padding='SAME')

        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0, tf.float32))

        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    pool1 = tf.nn.max_pool(
        conv1,
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='pool1')

    # norm1
    norm1 = tf.nn.lrn(
        pool1,
        4,
        bias=1.0,
        alpha=0.001 / 9.0,
        beta=0.75,
        name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay(
            'weights',
            shape=[5, 5, 64, 64],
            stddev=5e-2,
            wd=None)

        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = reduce(lambda x, y: x*y, pool2.get_shape().as_list()[1:])
        reshape = tf.reshape(pool2, [-1, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=None)
        biases = tf.get_variable(
            'biases',
            [NUM_CLASSES],
            initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / get_batch_size()
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    tf.train.get_global_step(),
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=tf.train.get_global_step())

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, tf.train.get_global_step())

    with tf.control_dependencies([apply_gradient_op]):
        train_op = variable_averages.apply(tf.trainable_variables())

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        # The classes variable below exists of an tensor that contains all the predicted classes in a batch
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops)


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


# def _augment_input_layer(input_layer):
#     # Image processing for training the network. Note the many random
#     # distortions applied to the image.
#
#     # Randomly crop a [height, width] section of the image.
#     distorted_image = tf.random_crop(input_layer, [height, width, 3])
#
#     # Randomly flip the image horizontally.
#     distorted_image = tf.image.random_flip_left_right(distorted_image)
#
#     # Because these operations are not commutative, consider randomizing
#     # the order their operation.
#     # NOTE: since per_image_standardization zeros the mean and makes
#     # the stddev unit, this likely has no effect see tensorflow#1458.
#     distorted_image = tf.image.random_brightness(distorted_image,
#                                                  max_delta=63)
#     distorted_image = tf.image.random_contrast(distorted_image,
#                                                lower=0.2, upper=1.8)
#
#     # Subtract off the mean and divide by the variance of the pixels.
#     float_image = tf.image.per_image_standardization(distorted_image)
#
#     # Set the shapes of tensors.
#     float_image.set_shape([height, width, 3])
#     read_input.label.set_shape([1])
