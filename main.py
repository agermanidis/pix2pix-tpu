from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
from functools import partial

from model import get_loss
from dataset import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch-size", type=int, default=8, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])

parser.add_argument("--use-tpu", action="store_true", help="use TPU")
parser.add_argument('--tpu', help="The Cloud TPU to use for training")
parser.add_argument('--tpu-zone', default=None, help="GCE zone where the Cloud TPU is located in.")
parser.add_argument('--gcp-project', default=None, help="Project name for the Cloud TPU-enabled project.")
parser.add_argument('--num-shards', default=8, type=int)
parser.add_argument('--model-dir', required=True, help="Estimator model dir")
parser.add_argument('--iterations', type=int, default=500, help="Estimator model dir")


args = parser.parse_args()


   
def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        raise RuntimeError("mode {} is not supported yet".format(mode))

    loss = get_loss(features, labels, args)

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            args.lr,
            tf.train.get_global_step(),
            decay_steps=100000,
            decay_rate=0.96)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        if args.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_global_step()))


def set_random_seed():
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def main(argv):
    set_random_seed()
    tf.logging.set_verbosity(tf.logging.INFO)

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        args.tpu,
        zone=args.tpu_zone,
        project=args.gcp_project
    )

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=args.model_dir,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(args.iterations, args.num_shards),
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=args.use_tpu,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        params={"data_dir": args.data_dir},
        config=run_config)

    estimator.train(input_fn=partial(load_dataset, args), max_steps=args.max_steps)


if __name__ == "__main__":
    tf.app.run()
