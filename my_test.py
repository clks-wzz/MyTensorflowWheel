# coding: utf-8
import pandas as pd
import numpy as np
import tensorflow as tf
from tfrecorder import TFrecorder
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from tfrecorder import TFrecorder
from my_generate_data import input_fn_maker
from my_generate_model import model_fn

# log info setting
tf.logging.set_verbosity(tf.logging.INFO)

# data fn
padding_info = ({'image':[28,28,1],'label':[]})
test_input_fn = input_fn_maker('./mnist_tfrecord/test/',  './mnist_tfrecord/data_info.csv',batch_size = 8,
                               padding = padding_info)
train_eval_fn = input_fn_maker('./mnist_tfrecord/train/',  './mnist_tfrecord/data_info.csv', batch_size = 8,
                               padding = padding_info)

# model fn
model_fn_this=model_fn

tensors_to_log = {"step": "global_ss", "loss": "loss", "accuracy": "lacc"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10) # every_n_iter is important

# # create estimator
this_config=tf.estimator.RunConfig(
    save_summary_steps=10,
    save_checkpoints_steps=100,
    keep_checkpoint_max=5,
    log_step_count_steps=100000
)
mnist_classifier = tf.estimator.Estimator(
    model_fn=model_fn, config=this_config, model_dir="mnist_model_cnn")

eval_results = mnist_classifier.evaluate(input_fn=train_eval_fn, hooks=[logging_hook])
print('train set')
print(eval_results)

# # evaluate test set 
eval_results = mnist_classifier.evaluate(input_fn=test_input_fn, hooks=[logging_hook])
print('test set')
print(eval_results)