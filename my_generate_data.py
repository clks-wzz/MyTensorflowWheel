import numpy as np
import tensorflow as tf
from tfrecorder import TFrecorder

tfr = TFrecorder()

def input_fn_maker(path, data_info_path, shuffle=False, batch_size = 1, epoch = 1, padding = None):
    def input_fn():
        filenames = tfr.get_filenames(path=path, shuffle=shuffle)
        dataset=tfr.get_dataset(paths=filenames, data_info=data_info_path, shuffle = shuffle, 
                            batch_size = batch_size, epoch = epoch, padding =padding)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn

padding_info = ({'image':[28,28,1],'label':[]})
test_input_fn = input_fn_maker('./mnist_tfrecord/test/',  './mnist_tfrecord/data_info.csv',batch_size = 8,
                               padding = padding_info)
train_input_fn = input_fn_maker('./mnist_tfrecord/train/',  './mnist_tfrecord/data_info.csv', shuffle=True, batch_size = 8,
                               padding = padding_info)
train_eval_fn = input_fn_maker('./mnist_tfrecord/train/',  './mnist_tfrecord/data_info.csv', batch_size = 8,
                               padding = padding_info)