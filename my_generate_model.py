import numpy as np
import tensorflow as tf
from tfrecorder import TFrecorder

def model_fn(features, mode):
    # shape: [None,28,28,1]
    conv1 = tf.layers.conv2d(
        inputs=features['image'],
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name = 'conv1')
    # shape: [None,28,28,32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name= 'pool1')
    # shape: [None,14,14,32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name = 'conv2')
    # shape: [None,14,14,64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name= 'pool2')
    # shape: [None,7,7,64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64], name= 'pool2_flat')
    # shape: [None,3136]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name= 'dense1')
    # shape: [None,1024]
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # shape: [None,1024]
    logits = tf.layers.dense(inputs=dropout, units=10, name= 'output')
    # shape: [None,10]
    predictions = {
        "image":features['image'],
        "conv1_out":conv1,
        "pool1_out":pool1,
        "conv2_out":conv2,
        "pool2_out":pool2,
        "pool2_flat_out":pool2_flat,
        "dense1_out":dense1,
        "logits":logits,
        "classes": tf.argmax(input=logits, axis=1),
        "labels": features['label'],
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    # predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=features['label'], logits=logits)
    loss = tf.reduce_sum(loss, name='loss')    

    global_count= tf.Variable(tf.constant(0), dtype=tf.int32, name="global_ss")  
    global_count_=tf.add(global_count,1)
    global_count_op=tf.assign(global_count, global_count_)

    lacc=tf.reduce_mean(tf.cast(tf.equal(features['label'], predictions["classes"]), tf.float32), name='lacc')
    tf.summary.scalar('accuracy', lacc)

    acc=tf.metrics.accuracy(labels=features['label'], predictions=predictions["classes"], name='accuracy')
    eval_metric_ops = {"accuracy": acc}

    # train
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        with tf.control_dependencies([global_count_op]):
            train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # evaluate
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)