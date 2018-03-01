# coding: utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
from tfrecorder import TFrecorder
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")

# info of data
df = pd.DataFrame({'name':['image','label'],
                  'type':['float32','int64'],
                  'shape':[(784,),()],
                  'isbyte':[False,False],
                  "length_type":['fixed','fixed'],
                  "default":[np.NaN,np.NaN]})

# # write
tfr = TFrecorder()
# In[7]:
dataset = mnist.train
path = 'mnist_tfrecord/train/train'
num_examples_per_file = 1000
num_so_far = 0

writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(path, num_so_far, num_examples_per_file))
# write mutilple examples
for i in np.arange(dataset.num_examples):
    features = {}
    # write image of one example
    tfr.feature_writer(df.iloc[0], dataset.images[i], features)
    # write label of one example
    tfr.feature_writer(df.iloc[1], dataset.labels[i], features)
    
    tf_features = tf.train.Features(feature= features)
    tf_example = tf.train.Example(features = tf_features)
    tf_serialized = tf_example.SerializeToString()
    writer.write(tf_serialized)
    
    if i%num_examples_per_file ==0 and i!=0:
        writer.close()
        num_so_far = i
        writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(path, num_so_far, i+num_examples_per_file))
writer.close()
data_info_path = 'mnist_tfrecord/data_info.csv'
df.to_csv(data_info_path,index=False)



# 用该方法写测试集的tfrecord文件
dataset = mnist.test
# 写在哪里
path = 'mnist_tfrecord/test/test'
# 把test_set写成符合要求的examples
test_set = []
for i in np.arange(dataset.num_examples):
    # 一个样本
    features = {}
    # 样本中的第一个feature
    features['image'] = dataset.images[i]
    # 样本中的第二个feature
    features['label'] = dataset.labels[i].astype('int64')
    test_set.append(features)
# 直接写入，每个tfrecord中写1000个样本
# 由于测试集里有10000个样本，所以最终会写出10个tfrecord文件
tfr.writer(path, test_set, num_examples_per_file = 1000)

# # import function
# In[9]:
'''
tfr = TFrecorder()
def input_fn_maker(path, data_info_path, shuffle=False, batch_size = 1, epoch = 1, padding = None):
    def input_fn():
        filenames = tfr.get_filenames(path=path, shuffle=shuffle)
        dataset=tfr.get_dataset(paths=filenames, data_info=data_info_path, shuffle = shuffle, 
                            batch_size = batch_size, epoch = epoch, padding =padding)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn
# In[10]:


padding_info = ({'image':[28,28,1],'label':[]})
test_input_fn = input_fn_maker('mnist_tfrecord/test/',  'mnist_tfrecord/data_info.csv',
                               padding = padding_info)
train_input_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', shuffle=True, batch_size = 512,
                               padding = padding_info)
train_eval_fn = input_fn_maker('mnist_tfrecord/train/',  'mnist_tfrecord/data_info.csv', batch_size = 512,
                               padding = padding_info)
test_inputs = test_input_fn()

# In[11]:


sess =tf.InteractiveSession()
print(test_inputs['image'].eval().shape)
plt.imshow(test_inputs['image'].eval()[0,:,:,0],cmap=plt.cm.gray)
'''
