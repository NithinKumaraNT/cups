#!/usr/bin/env python
"""This file trains the model upon the training data and evaluates it with the test data.
It uses the arguments it got via the gcloud command."""
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

import trainer.data as data
import trainer.model_adams as model


def train_model(params):

    """The function gets the training data from the training folder,
    the evaluation data from the test folder and trains the CNN from the model.py file with it."""
    (train_data, train_labels) = data.create_data_with_labels("data/train/")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=model.get_batch_size(),
        num_epochs=None,
        shuffle=True)

    (eval_data, eval_labels) = data.create_data_with_labels("data/test/")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    estimator = tf.estimator.Estimator(model_fn=model.cnn)

    steps_per_eval = int(model.get_training_steps() / params.eval_steps)

    res = []

    for _ in range(params.eval_steps):
        estimator.train(train_input_fn, steps=steps_per_eval)
        res.append(estimator.evaluate(eval_input_fn))

    f1 = plt.figure(1)
    plt.plot(list(item["loss"] for item in res))
    plt.ylabel('loss')
    f1.show()

    f2 = plt.figure(2)
    plt.plot(list(item["accuracy"] for item in res))
    plt.ylabel('accuracy')
    f2.show()

    raw_input('Press enter to continue: ')


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--eval-steps',
        help='Number of steps to run evaluation for at each checkpoint',
        default=1,
        type=int
    )

    ARGS = PARSER.parse_args()
    tf.logging.set_verbosity('INFO')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.logging.__dict__['INFO'] / 10)

    HPARAMS = hparam.HParams(**ARGS.__dict__)
    train_model(HPARAMS)
