#!/usr/bin/env python
"""This file trains the model upon all data with the arguments it got via the gcloud command"""

from functools import partial
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

import trainer.data as data
import trainer.model as model


def json_serving_input_fn():
    """This function is used to do predictions on Google Cloud when receiving a json file."""
    input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
    images = tf.map_fn(partial(tf.image.decode_jpeg, channels=3), input_ph, dtype=tf.uint8)
    images = tf.cast(images, tf.float32) / 255.
    images.set_shape([None, 64, 64, 3])

    return tf.estimator.export.ServingInputReceiver({"x": images}, {'bytes': input_ph})


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn
}


def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters set."""

    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
            'index' in tf_config['task']):
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=[
                '/job:ps',
                '/job:worker/task:%d' % tf_config['task']['index']
            ])
    return None


def train_model(params):
    """The function gets the training data from the training folder and the test folder.
    The CNN from the model.py file is trained with this training data.
    The evaluation in this method is not important since all data was already used to train."""
    (train_data, train_labels) = data.create_data_with_labels("data/train/")
    (eval_data, eval_labels) = data.create_data_with_labels("data/test/")

    train_data = np.append(train_data, eval_data, axis=0)
    train_labels = np.append(train_labels, eval_labels, axis=0)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=model.get_batch_size(),
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    exporter = tf.estimator.FinalExporter('exported', SERVING_FUNCTIONS[params.export_format])
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=model.get_training_steps())
    eval_spec = tf.estimator.EvalSpec(eval_input_fn,
                                      steps=params.eval_steps,
                                      exporters=[exporter],
                                      name='exported_eval')

    run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var())
    run_config = run_config.replace(model_dir=params.job_dir)

    estimator = tf.estimator.Estimator(model_fn=model.cnn, config=run_config)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--job-dir',
        type=str,
        default='output',
        help='directory to store checkpointed models'
    )
    PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=1,
        type=int
    )
    PARSER.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='JSON'
    )

    ARGS = PARSER.parse_args()
    tf.logging.set_verbosity('INFO')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__['INFO'] / 10)

    HPARAMS = hparam.HParams(**ARGS.__dict__)
    train_model(HPARAMS)
